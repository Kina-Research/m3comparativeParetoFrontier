"""Perplexity and representation energy evaluation across Transformer and JEPA paradigms.

Formulates unified language modeling and representation quality metrics:
1. Decoder-Only (Causal LM):
   L_AR = - (1 / N) * sum_{t=1}^N log P(x_t | x_{<t})
   PPL = exp(L_AR)

2. Encoder-Decoder (Seq2Seq LM):
   L_Seq2Seq = - (1 / M) * sum_{t=1}^M log P(y_t | y_{<t}, x)
   PPL_cond = exp(L_Seq2Seq)

3. Joint-Embedding Predictive Architecture (JEPA):
   E_JEPA = (1 / T_tgt) * sum_{t=1}^T_tgt (1 - cos(s_hat_{y,t}, s_{y,t}))
   PPL_latent = exp(E_JEPA)  [Predictive uncertainty in embedding manifold]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_causal_lm_perplexity(model, input_ids: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Calculates cross-entropy loss and token perplexity for Decoder-Only model."""
    model.eval()
    with torch.no_grad():
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.item() if hasattr(outputs, "loss") else float("nan")
        ppl = math.exp(min(loss, 100.0))

    return {
        "cross_entropy_loss": loss,
        "perplexity": ppl,
        "bits_per_character": loss / math.log(2),
    }


def compute_seq2seq_perplexity(model, input_ids: torch.Tensor,
                              decoder_input_ids: torch.Tensor,
                              labels: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Calculates conditional cross-entropy loss and perplexity for Encoder-Decoder model."""
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss = outputs.loss.item() if hasattr(outputs, "loss") else float("nan")
        ppl = math.exp(min(loss, 100.0))

    return {
        "conditional_loss": loss,
        "conditional_perplexity": ppl,
        "token_accuracy": float(
            (outputs.logits.argmax(dim=-1) == labels)[labels != -100].float().mean()
        ) if hasattr(outputs, "logits") else float("nan"),
    }


def compute_jepa_latent_energy(jepa_model, context_ids: torch.Tensor,
                              target_ids: torch.Tensor,
                              context_mask: Optional[torch.Tensor] = None,
                              target_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Calculates latent prediction energy and representation pseudo-perplexity for JEPA."""
    jepa_model.eval()
    with torch.no_grad():
        res = jepa_model(
            context_ids=context_ids,
            target_ids=target_ids,
            context_mask=context_mask,
            target_mask=target_mask,
        )
        energy = res["loss"].item()
        pseudo_ppl = math.exp(min(energy, 20.0))

    return {
        "latent_energy_loss": energy,
        "pseudo_perplexity": pseudo_ppl,
        "representation_alignment": 1.0 - energy,
    }
