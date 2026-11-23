"""Computational complexity and FLOPs counter for Transformer and JEPA paradigms.

Provides exact theoretical and empirical Floating Point Operations (FLOPs)
and Multiply-Accumulate operations (MACs) for:
1. Encoder-Decoder Transformers (e.g., mT5)
2. Decoder-Only Transformers (e.g., BLOOMZ)
3. Joint-Embedding Predictive Architecture (Text-JEPA)

Distinguishes:
- Self-attention (MHSA): Q, K, V projections, QK^T scores, attention*V, out_proj
- Feed-forward network (FFN): standard MLP vs gated SwiGLU/GEGLU
- Cross-attention (in encoder-decoder)
- Vocabulary projection head (generative models) vs. Latent prediction (JEPA)
- Forward pass vs. Backward pass (2x forward for gradients)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ArchitectureSpecs:
    name: str
    paradigm: str  # 'encoder_decoder', 'decoder_only', 'jepa'
    num_layers: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    vocab_size: int
    # Encoder-decoder specific
    num_decoder_layers: Optional[int] = None
    # JEPA specific
    num_predictor_layers: Optional[int] = None
    is_gated_ffn: bool = False


# Canonical model specifications in the Amharic benchmark
MODEL_SPECS: Dict[str, ArchitectureSpecs] = {
    "mt5_small": ArchitectureSpecs(
        name="mT5-small",
        paradigm="encoder_decoder",
        num_layers=8,  # 8 encoder layers
        hidden_dim=512,
        num_heads=6,
        ffn_dim=1024,
        vocab_size=250112,
        num_decoder_layers=8,  # 8 decoder layers
        is_gated_ffn=True,  # GEGLU
    ),
    "bloomz_560m": ArchitectureSpecs(
        name="BLOOMZ-560M",
        paradigm="decoder_only",
        num_layers=24,
        hidden_dim=1024,
        num_heads=16,
        ffn_dim=4096,
        vocab_size=250880,
        is_gated_ffn=False,
    ),
    "text_jepa_base": ArchitectureSpecs(
        name="Text-JEPA (mBERT Backbone + Predictor)",
        paradigm="jepa",
        num_layers=12,  # 12 context encoder layers
        hidden_dim=768,
        num_heads=12,
        ffn_dim=3072,
        vocab_size=119547,  # Used only for input embedding lookup, NO output projection
        num_predictor_layers=4,  # 4 latent predictor layers
        is_gated_ffn=False,
    ),
}


class FLOPsCalculator:
    """Computes exact FLOPs per sequence for Transformer and JEPA architectures."""

    @staticmethod
    def mhsa_flops(batch_size: int, seq_len: int, hidden_dim: int) -> int:
        """FLOPs for Multi-Head Self-Attention (MHSA).
        Q, K, V proj: 3 * 2 * B * T * d^2
        QK^T: 2 * B * T^2 * d
        Softmax * V: 2 * B * T^2 * d
        Output proj: 2 * B * T * d^2
        Total = 8 * B * T * d^2 + 4 * B * T^2 * d
        """
        b, t, d = batch_size, seq_len, hidden_dim
        proj_flops = 8 * b * t * (d ** 2)
        attn_matrix_flops = 4 * b * (t ** 2) * d
        return proj_flops + attn_matrix_flops

    @staticmethod
    def ffn_flops(batch_size: int, seq_len: int, hidden_dim: int, ffn_dim: int, is_gated: bool = False) -> int:
        """FLOPs for Feed-Forward Network.
        Standard MLP: 2 linear layers (d -> ffn_dim -> d) = 2 * 2 * B * T * d * ffn_dim
        Gated MLP (SwiGLU/GEGLU): 3 linear layers (gate, up, down) = 3 * 2 * B * T * d * ffn_dim
        """
        b, t, d = batch_size, seq_len, hidden_dim
        factor = 3 if is_gated else 2
        return factor * 2 * b * t * d * ffn_dim

    @staticmethod
    def cross_attn_flops(batch_size: int, dec_len: int, enc_len: int, hidden_dim: int) -> int:
        """FLOPs for Cross-Attention in Encoder-Decoder.
        Q from decoder: 2 * B * T_dec * d^2
        K, V from encoder: 4 * B * T_enc * d^2
        Cross attention QK^T: 2 * B * T_dec * T_enc * d
        Attention * V: 2 * B * T_dec * T_enc * d
        Output proj: 2 * B * T_dec * d^2
        """
        b, td, te, d = batch_size, dec_len, enc_len, hidden_dim
        q_proj = 2 * b * td * (d ** 2)
        kv_proj = 4 * b * te * (d ** 2)
        scores = 4 * b * td * te * d
        out_proj = 2 * b * td * (d ** 2)
        return q_proj + kv_proj + scores + out_proj

    @staticmethod
    def vocab_projection_flops(batch_size: int, seq_len: int, hidden_dim: int, vocab_size: int) -> int:
        """FLOPs for final linear projection to vocabulary: 2 * B * T * d * |V|."""
        return 2 * batch_size * seq_len * hidden_dim * vocab_size

    @classmethod
    def compute_forward_flops(
        cls,
        specs: ArchitectureSpecs,
        seq_len: int = 128,
        batch_size: int = 1,
        dec_seq_len: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculates total forward FLOPs and component breakdown.
        
        Returns dictionary with components in GFLOPs (10^9 FLOPs).
        """
        b = batch_size
        t = seq_len
        d = specs.hidden_dim
        ffn_d = specs.ffn_dim

        if specs.paradigm == "decoder_only":
            # Decoder-only (e.g. BLOOMZ): L layers of (MHSA + FFN) + Vocab Proj
            layer_mhsa = cls.mhsa_flops(b, t, d)
            layer_ffn = cls.ffn_flops(b, t, d, ffn_d, is_gated=specs.is_gated_ffn)
            backbone_flops = specs.num_layers * (layer_mhsa + layer_ffn)
            vocab_proj_flops = cls.vocab_projection_flops(b, t, d, specs.vocab_size)
            total_flops = backbone_flops + vocab_proj_flops

            return {
                "backbone_gflops": backbone_flops / 1e9,
                "attention_gflops": (specs.num_layers * layer_mhsa) / 1e9,
                "ffn_gflops": (specs.num_layers * layer_ffn) / 1e9,
                "head_gflops": vocab_proj_flops / 1e9,
                "total_gflops": total_flops / 1e9,
                "head_pct": 100.0 * vocab_proj_flops / total_flops,
            }

        elif specs.paradigm == "encoder_decoder":
            # Encoder-Decoder (e.g. mT5):
            # Encoder: L_enc * (MHSA + FFN)
            # Decoder: L_dec * (Causal MHSA + Cross-Attn + FFN) + Vocab Proj
            td = dec_seq_len if dec_seq_len is not None else t
            te = t

            enc_mhsa = cls.mhsa_flops(b, te, d)
            enc_ffn = cls.ffn_flops(b, te, d, ffn_d, is_gated=specs.is_gated_ffn)
            enc_total = specs.num_layers * (enc_mhsa + enc_ffn)

            dec_mhsa = cls.mhsa_flops(b, td, d)
            dec_cross = cls.cross_attn_flops(b, td, te, d)
            dec_ffn = cls.ffn_flops(b, td, d, ffn_d, is_gated=specs.is_gated_ffn)
            num_dec = specs.num_decoder_layers or specs.num_layers
            dec_total = num_dec * (dec_mhsa + dec_cross + dec_ffn)

            vocab_proj_flops = cls.vocab_projection_flops(b, td, d, specs.vocab_size)
            total_flops = enc_total + dec_total + vocab_proj_flops

            return {
                "encoder_gflops": enc_total / 1e9,
                "decoder_backbone_gflops": dec_total / 1e9,
                "attention_gflops": (specs.num_layers * enc_mhsa + num_dec * (dec_mhsa + dec_cross)) / 1e9,
                "ffn_gflops": (specs.num_layers * enc_ffn + num_dec * dec_ffn) / 1e9,
                "head_gflops": vocab_proj_flops / 1e9,
                "total_gflops": total_flops / 1e9,
                "head_pct": 100.0 * vocab_proj_flops / total_flops,
            }

        elif specs.paradigm == "jepa":
            # Text-JEPA:
            # Context Encoder: L_enc * (MHSA + FFN)
            # Latent Predictor: L_pred * (MHSA + FFN)
            # NO vocabulary projection! Target is predicted in latent space R^d!
            enc_mhsa = cls.mhsa_flops(b, t, d)
            enc_ffn = cls.ffn_flops(b, t, d, ffn_d, is_gated=specs.is_gated_ffn)
            enc_total = specs.num_layers * (enc_mhsa + enc_ffn)

            # Predictor processes concatenated [context + target_queries]
            t_pred = t + (dec_seq_len if dec_seq_len is not None else (t // 2))
            num_pred = specs.num_predictor_layers or 4
            pred_mhsa = cls.mhsa_flops(b, t_pred, d)
            pred_ffn = cls.ffn_flops(b, t_pred, d, ffn_d, is_gated=specs.is_gated_ffn)
            pred_total = num_pred * (pred_mhsa + pred_ffn)

            # Latent distance loss: 2 * B * T * d operations (negligible)
            loss_flops = 2 * b * (dec_seq_len or (t // 2)) * d
            total_flops = enc_total + pred_total + loss_flops

            return {
                "encoder_gflops": enc_total / 1e9,
                "predictor_gflops": pred_total / 1e9,
                "attention_gflops": (specs.num_layers * enc_mhsa + num_pred * pred_mhsa) / 1e9,
                "ffn_gflops": (specs.num_layers * enc_ffn + num_pred * pred_ffn) / 1e9,
                "head_gflops": loss_flops / 1e9,
                "total_gflops": total_flops / 1e9,
                "head_pct": 100.0 * loss_flops / total_flops,
            }

        raise ValueError(f"Unknown paradigm: {specs.paradigm}")

    @classmethod
    def scaling_curve(
        cls, specs: ArchitectureSpecs, seq_lens: List[int] = [64, 128, 256, 512, 1024]
    ) -> List[Dict[str, float]]:
        """Generates scaling curve of GFLOPs across sequence lengths."""
        curve = []
        for seq_len in seq_lens:
            res = cls.compute_forward_flops(specs, seq_len=seq_len, batch_size=1)
            res["seq_len"] = seq_len
            curve.append(res)
        return curve
