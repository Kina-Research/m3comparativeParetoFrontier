"""Joint-Embedding Predictive Architecture (JEPA) for text sequences.

Following LeCun (2022) and Assran et al. (CVPR 2023), Text-JEPA avoids generative
token-level reconstruction and the computational burden of a massive vocabulary
softmax. Instead, it predicts target representations directly in latent embedding space:

    s_x = E_theta(x)               [Context Encoder]
    s_y = E_phi(y)                 [Target Encoder with EMA updates]
    s_hat_y = P_psi(s_x, queries)  [Latent Transformer Predictor]
    L_JEPA = D(s_hat_y, s_y)       [Representation distance in R^d]

Key Architectural Advantages for Low-Resource Semitic Modeling:
1. Vocabulary-head elimination: eliminates the d x |V| matrix multiplication
   and softmax (which requires 250,000 x 768 = ~192M parameters in mT5/BLOOMZ).
2. Non-autoregressive latent prediction: predicts entire target sequence representations
   in a single forward pass (O(1) generation vs. O(T) autoregressive steps).
3. Robustness to morphological fragmentation: operates in dense semantic feature
   space rather than penalizing surface-level subword boundary mismatches.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """Narrow transformer predictor operating strictly in latent space R^d."""

    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, context_repr: torch.Tensor,
                target_pos_embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context_repr: (B, T_ctx, d_model) from Context Encoder
            target_pos_embeddings: (B, T_tgt, d_model) positional queries for target tokens
            mask: optional attention mask
        Returns:
            predicted_target_repr: (B, T_tgt, d_model)
        """
        # Form sequence: [context_tokens, query_tokens]
        queries = self.query_proj(target_pos_embeddings)
        seq = torch.cat([context_repr, queries], dim=1)
        h = self.transformer(seq, src_key_padding_mask=mask)
        h = self.norm(h)
        # Extract predictions corresponding to query positions
        t_ctx = context_repr.size(1)
        predicted_target_repr = h[:, t_ctx:, :]
        return predicted_target_repr


class TextJEPA(nn.Module):
    """Complete Text-JEPA Architecture: Context Encoder, EMA Target Encoder, Latent Predictor."""

    def __init__(
        self,
        context_encoder: nn.Module,
        d_model: int = 768,
        predictor_layers: int = 4,
        predictor_heads: int = 8,
        predictor_dim_ff: int = 2048,
        ema_decay: float = 0.996,
        loss_type: str = "cosine",
    ):
        super().__init__()
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.loss_type = loss_type

        # Context Encoder (online, updated via gradient descent)
        self.context_encoder = context_encoder

        # Target Encoder (EMA twin network, no gradient)
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Latent Predictor
        self.predictor = LatentPredictor(
            d_model=d_model,
            nhead=predictor_heads,
            num_layers=predictor_layers,
            dim_feedforward=predictor_dim_ff,
        )

        # Learned target positional query embeddings (up to max_seq_len 512)
        self.pos_queries = nn.Embedding(512, d_model)

    @torch.no_grad()
    def update_target_encoder(self):
        """Exponential Moving Average update: phi = tau * phi + (1 - tau) * theta."""
        for p_online, p_target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(self.ema_decay).add_(
                p_online.data, alpha=1.0 - self.ema_decay
            )

    def encode_context(self, input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    @torch.no_grad()
    def encode_target(self, target_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.target_encoder(input_ids=target_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def forward(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing latent prediction loss."""
        # 1. Encode context representation
        s_x = self.encode_context(context_ids, attention_mask=context_mask)

        # 2. Encode target representation (stop gradient)
        s_y = self.encode_target(target_ids, attention_mask=target_mask)

        # 3. Predict target representation in latent space
        batch_size, t_tgt = target_ids.size()
        pos_indices = torch.arange(t_tgt, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
        target_pos_emb = self.pos_queries(pos_indices)

        s_hat_y = self.predictor(s_x, target_pos_emb)

        # 4. Compute latent distance loss
        if self.loss_type == "cosine":
            # 1 - cosine_similarity
            s_hat_norm = F.normalize(s_hat_y, p=2, dim=-1)
            s_y_norm = F.normalize(s_y, p=2, dim=-1)
            cos_sim = (s_hat_norm * s_y_norm).sum(dim=-1)
            if target_mask is not None:
                cos_sim = cos_sim * target_mask.float()
                loss = 1.0 - (cos_sim.sum() / target_mask.float().sum().clamp(min=1.0))
            else:
                loss = 1.0 - cos_sim.mean()
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(s_hat_y, s_y)
        else:
            loss = F.mse_loss(s_hat_y, s_y)

        # Pseudo-perplexity / representation uncertainty metric: exp(loss)
        pseudo_ppl = torch.exp(loss.detach())

        return {
            "loss": loss,
            "pseudo_ppl": pseudo_ppl,
            "s_hat_y": s_hat_y,
            "s_y": s_y,
        }

    def predict_latent(self, context_ids: torch.Tensor, target_len: int,
                       context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Non-autoregressive one-step latent representation generation."""
        s_x = self.encode_context(context_ids, attention_mask=context_mask)
        batch_size = context_ids.size(0)
        pos_indices = torch.arange(target_len, device=context_ids.device).unsqueeze(0).expand(batch_size, -1)
        target_pos_emb = self.pos_queries(pos_indices)
        return self.predictor(s_x, target_pos_emb)
