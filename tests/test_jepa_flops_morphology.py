"""Unit tests for JEPA architecture, FLOPs calculation, parameter breakdown, and Semitic morphology."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.models.jepa import LatentPredictor, TextJEPA
from src.efficiency.flops_counter import FLOPsCalculator, MODEL_SPECS, ArchitectureSpecs
from src.models.parameter_breakdown import analyze_parameter_distribution, compare_all_paradigms
from src.evaluation.morphological_analyzer import (
    AMHARIC_MORPHOLOGICAL_BENCHMARK,
    SemiticMorphologyAnalyzer,
)
from src.efficiency.benchmark_inference import compute_generation_latency_breakdown


class MockContextEncoder(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        self.layer = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        return self.layer(self.embedding(input_ids))


def test_latent_predictor_forward():
    d_model = 64
    predictor = LatentPredictor(d_model=d_model, nhead=2, num_layers=2, dim_feedforward=128)
    context_repr = torch.randn(2, 10, d_model)
    target_pos = torch.randn(2, 5, d_model)
    out = predictor(context_repr, target_pos)
    assert out.shape == (2, 5, d_model)


def test_text_jepa_forward_and_ema():
    d_model = 64
    encoder = MockContextEncoder(d_model=d_model)
    jepa = TextJEPA(
        context_encoder=encoder,
        d_model=d_model,
        predictor_layers=2,
        predictor_heads=2,
        predictor_dim_ff=128,
        ema_decay=0.99,
        loss_type="cosine",
    )

    context_ids = torch.randint(0, 1000, (2, 8))
    target_ids = torch.randint(0, 1000, (2, 4))

    res = jepa(context_ids, target_ids)
    assert "loss" in res
    assert "pseudo_ppl" in res
    assert res["loss"].item() >= 0.0
    assert res["pseudo_ppl"].item() >= 1.0
    assert res["s_hat_y"].shape == (2, 4, d_model)

    # Test EMA update
    initial_target_weight = jepa.target_encoder.layer.weight.data.clone()
    jepa.context_encoder.layer.weight.data.add_(1.0)
    jepa.update_target_encoder()
    updated_target_weight = jepa.target_encoder.layer.weight.data
    assert not torch.equal(initial_target_weight, updated_target_weight)


def test_flops_calculator_all_paradigms():
    for name, spec in MODEL_SPECS.items():
        res = FLOPsCalculator.compute_forward_flops(spec, seq_len=128, batch_size=1)
        assert "total_gflops" in res
        assert res["total_gflops"] > 0
        assert "attention_gflops" in res
        assert "ffn_gflops" in res

        # JEPA should have near-zero head FLOPs compared to generative vocab heads
        if spec.paradigm == "jepa":
            assert res["head_pct"] < 5.0  # Head is negligible in latent space
        else:
            assert res["head_pct"] > 5.0  # Vocabulary projection has noticeable FLOPs


def test_parameter_breakdown():
    comparisons = compare_all_paradigms()
    assert len(comparisons) == 3
    for entry in comparisons:
        assert entry["total_params_m"] > 0
        assert entry["embed_params_m"] > 0
        assert entry["lora_trainable_pct"] < 2.0  # LoRA restricts trainable parameters to < 2%


def test_semitic_morphology_benchmark():
    assert len(AMHARIC_MORPHOLOGICAL_BENCHMARK) >= 20
    tiers = {m.complexity_tier for m in AMHARIC_MORPHOLOGICAL_BENCHMARK}
    assert tiers == {1, 2, 3}
    types = {m.root_type for m in AMHARIC_MORPHOLOGICAL_BENCHMARK}
    assert "triliteral_strong" in types
    assert "triliteral_weak" in types
    assert "quadriliteral" in types


def test_generation_latency_breakdown():
    # Test autoregressive decoding vs JEPA non-autoregressive
    ar_res = compute_generation_latency_breakdown(prompt_len=128, gen_len=64, is_autoregressive=True)
    jepa_res = compute_generation_latency_breakdown(prompt_len=128, gen_len=64, is_autoregressive=False)

    assert ar_res["total_latency_ms"] > jepa_res["total_latency_ms"]
    assert jepa_res["tokens_per_sec"] > ar_res["tokens_per_sec"]
