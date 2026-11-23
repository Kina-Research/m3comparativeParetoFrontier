"""Detailed parameter accounting and efficiency breakdown for Transformer and JEPA paradigms.

Calculates:
- Total parameter count N_total
- Embedding table footprint: N_embed = |V| * d
- Transformer backbone footprint: N_backbone (Layers * (Attention + FFN + Norm))
- Projection / Predictor head footprint: N_head
- LoRA parameter efficiency: N_trainable under rank r=8, alpha=16
- Memory footprint in fp32, fp16, and 4-bit quantization
"""
from __future__ import annotations

from typing import Dict, List, Optional
from src.efficiency.flops_counter import MODEL_SPECS, ArchitectureSpecs


def analyze_parameter_distribution(specs: ArchitectureSpecs, lora_r: int = 8) -> Dict[str, float]:
    """Computes exact parameter breakdown for a given architectural specification."""
    d = specs.hidden_dim
    v = specs.vocab_size
    ffn_d = specs.ffn_dim

    # 1. Embedding parameters
    # Input embeddings: v * d. (Positional embeddings: 512 * d or RoPE with 0 learnable params)
    embed_params = v * d

    # 2. Backbone parameters per layer
    # MHSA: Q, K, V, O projections -> 4 * (d * d) + 4 * d biases
    mhsa_params_per_layer = 4 * (d * d) + 4 * d
    
    # FFN:
    # Standard: 2 * (d * ffn_d) + (ffn_d + d) biases
    # Gated (GEGLU/SwiGLU): 3 * (d * ffn_d) + (2*ffn_d + d) biases
    if specs.is_gated_ffn:
        ffn_params_per_layer = 3 * (d * ffn_d)
    else:
        ffn_params_per_layer = 2 * (d * ffn_d) + (ffn_d + d)
        
    # LayerNorms / RMSNorms: 2 per layer (or 3 for cross-attn), each with 2*d (gamma, beta)
    norm_params_per_layer = 4 * d

    if specs.paradigm == "decoder_only":
        # Backbone = num_layers * (mhsa + ffn + norm)
        backbone_params = specs.num_layers * (mhsa_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
        # Head: tied or untied lm_head (d * v)
        head_params = d * v
        total_params = embed_params + backbone_params + head_params

        # LoRA on attention (Q, K, V, O): 4 * (2 * d * r) per layer
        lora_params = specs.num_layers * 4 * (2 * d * lora_r)

    elif specs.paradigm == "encoder_decoder":
        num_enc = specs.num_layers
        num_dec = specs.num_decoder_layers or specs.num_layers

        enc_backbone = num_enc * (mhsa_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
        
        # Decoder has cross-attention: extra (4 * d * d) + extra norm (2 * d)
        cross_attn_params = 4 * (d * d) + 4 * d
        dec_backbone = num_dec * (mhsa_params_per_layer + cross_attn_params + ffn_params_per_layer + 6 * d)

        backbone_params = enc_backbone + dec_backbone
        head_params = d * v  # Tied or untied
        total_params = embed_params + backbone_params + head_params

        # LoRA on encoder + decoder attention:
        enc_lora = num_enc * 4 * (2 * d * lora_r)
        dec_lora = num_dec * (4 + 4) * (2 * d * lora_r)  # self-attn + cross-attn
        lora_params = enc_lora + dec_lora

    elif specs.paradigm == "jepa":
        # Context Encoder (12 layers)
        context_enc_params = specs.num_layers * (mhsa_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
        backbone_params = context_enc_params

        # Latent Predictor (e.g. 4 layers, operates in latent space d)
        num_pred = specs.num_predictor_layers or 4
        pred_params = num_pred * (mhsa_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
        
        # Positional queries: 512 * d
        pos_queries_params = 512 * d
        head_params = pred_params + pos_queries_params

        # NO lm_head! Output vocabulary projection is zero!
        total_params = embed_params + backbone_params + head_params

        # LoRA on context encoder attention:
        lora_params = specs.num_layers * 4 * (2 * d * lora_r)

    trainable_pct = (lora_params / total_params) * 100.0

    return {
        "model_name": specs.name,
        "paradigm": specs.paradigm,
        "total_params_m": total_params / 1e6,
        "embed_params_m": embed_params / 1e6,
        "backbone_params_m": backbone_params / 1e6,
        "head_params_m": head_params / 1e6,
        "embed_fraction_pct": 100.0 * embed_params / total_params,
        "lora_trainable_params_m": lora_params / 1e6,
        "lora_trainable_pct": trainable_pct,
        "memory_fp32_mb": (total_params * 4) / (1024 * 1024),
        "memory_fp16_mb": (total_params * 2) / (1024 * 1024),
        "memory_int4_mb": (total_params * 0.5) / (1024 * 1024),
    }


def compare_all_paradigms() -> List[Dict[str, float]]:
    """Generates comparative parameter table across all evaluated models."""
    results = []
    for key, spec in MODEL_SPECS.items():
        results.append(analyze_parameter_distribution(spec))
    return results
