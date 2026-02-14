"""Tests for GemmaFamily VRAM estimation.

Validates that GemmaFamily inherits LlamaFamily's activation memory
formula and overrides kv_cache_eval for alternating sliding window
attention (Gemma 2).
"""

import pytest

from fitcheck.models.profiles import (
    HardwareSpec,
    LoRAConfig,
    ModelProfile,
    TrainingMethod,
)
from fitcheck.profilers.vram.engine import VRAMEstimator
from fitcheck.profilers.vram.families.gemma import GemmaFamily
from fitcheck.profilers.vram.families.llama import LlamaFamily


def _make_gemma2_9b() -> ModelProfile:
    """Real Gemma 2 9B dimensions with sliding window."""
    return ModelProfile(
        model_id="google/gemma-2-9b",
        architecture="Gemma2ForCausalLM",
        family="gemma",
        hidden_size=3584,
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=256000,
        total_params=9_240_000_000,
        total_params_b=9.24,
        sliding_window=4096,
        max_position_embeddings=8192,
    )


def _make_gemma2_no_window() -> ModelProfile:
    """Gemma-like model without sliding window (e.g., Gemma 1)."""
    return ModelProfile(
        model_id="google/gemma-7b",
        architecture="GemmaForCausalLM",
        family="gemma",
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=16,
        num_kv_heads=16,
        intermediate_size=24576,
        vocab_size=256000,
        total_params=8_540_000_000,
        total_params_b=8.54,
        sliding_window=None,
    )


def _make_llama_equivalent() -> ModelProfile:
    """LlamaFamily model with same dimensions as Gemma 2 for comparison."""
    return ModelProfile(
        model_id="llama-equiv",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=3584,
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=256000,
        total_params=9_240_000_000,
        total_params_b=9.24,
    )


def _make_3090() -> HardwareSpec:
    return HardwareSpec(
        name="NVIDIA RTX 3090",
        total_vram_gb=24.0,
        overhead_gb=1.2,
        memory_bandwidth_gbps=936.2,
        fp16_tflops=35.6,
        bf16_tflops=35.6,
    )


# ---------------------------------------------------------------------------
# Activation memory (inherited from LlamaFamily)
# ---------------------------------------------------------------------------


class TestGemmaActivationMemory:
    """GemmaFamily activation_memory delegates to LlamaFamily."""

    def test_matches_llama_for_same_dimensions(self):
        """GemmaFamily and LlamaFamily should produce identical activation memory
        for models with the same architecture dimensions."""
        gemma_family = GemmaFamily()
        llama_family = LlamaFamily()
        gemma_model = _make_gemma2_9b()
        llama_model = _make_llama_equivalent()

        gemma_act = gemma_family.activation_memory(gemma_model, 2, 1024, False)
        llama_act = llama_family.activation_memory(llama_model, 2, 1024, False)
        assert gemma_act.bytes == llama_act.bytes

    def test_scales_linearly_with_batch_size(self):
        family = GemmaFamily()
        model = _make_gemma2_9b()
        bs1 = family.activation_memory(model, 1, 1024, False)
        bs4 = family.activation_memory(model, 4, 1024, False)
        ratio = bs4.bytes / bs1.bytes
        assert 3.9 < ratio < 4.1

    def test_grad_checkpointing_reduces_memory(self):
        family = GemmaFamily()
        model = _make_gemma2_9b()
        without = family.activation_memory(model, 1, 1024, False)
        with_ckpt = family.activation_memory(model, 1, 1024, True)
        assert with_ckpt.bytes < without.bytes


# ---------------------------------------------------------------------------
# KV-cache with sliding window
# ---------------------------------------------------------------------------


class TestGemmaSlidingWindowKVCache:
    """KV-cache accounts for alternating sliding window attention."""

    def test_sliding_window_reduces_kv_cache(self):
        """At seq=8192 with window=4096, sliding window layers use half the
        KV-cache of full attention layers. Total should be about 75% of
        all-full-attention."""
        family = GemmaFamily()
        model = _make_gemma2_9b()

        # With sliding window
        with_window = family.kv_cache_eval(model, 1, 8192)

        # Without sliding window (full attention on all layers)
        llama_family = LlamaFamily()
        llama_model = _make_llama_equivalent()
        full_attn = llama_family.kv_cache_eval(llama_model, 1, 8192)

        # 42 layers: 21 full + 21 sliding (at half seq)
        # Expected ratio: (21 * 8192 + 21 * 4096) / (42 * 8192) = 0.75
        ratio = with_window.bytes / full_attn.bytes
        assert 0.70 < ratio < 0.80, (
            f"Sliding window KV-cache should be ~75% of full, got {ratio:.2%}"
        )

    def test_short_seq_within_window_no_effect(self):
        """If eval_seq_len <= sliding_window, all layers use full seq_len,
        so sliding window has no effect."""
        gemma_family = GemmaFamily()
        llama_family = LlamaFamily()
        gemma_model = _make_gemma2_9b()  # window=4096
        llama_model = _make_llama_equivalent()

        # seq=2048 <= window=4096
        gemma_kv = gemma_family.kv_cache_eval(gemma_model, 1, 2048)
        llama_kv = llama_family.kv_cache_eval(llama_model, 1, 2048)
        assert gemma_kv.bytes == llama_kv.bytes

    def test_no_window_falls_back_to_full_attention(self):
        """If sliding_window is None, behaves like LlamaFamily."""
        gemma_family = GemmaFamily()
        llama_family = LlamaFamily()
        model_no_window = _make_gemma2_no_window()

        # Build a llama equivalent with same dims
        llama_equiv = ModelProfile(
            model_id="llama-equiv-2",
            architecture="LlamaForCausalLM",
            family="llama",
            hidden_size=model_no_window.hidden_size,
            num_layers=model_no_window.num_layers,
            num_attention_heads=model_no_window.num_attention_heads,
            num_kv_heads=model_no_window.num_kv_heads,
            intermediate_size=model_no_window.intermediate_size,
            vocab_size=model_no_window.vocab_size,
            total_params=model_no_window.total_params,
            total_params_b=model_no_window.total_params_b,
        )

        gemma_kv = gemma_family.kv_cache_eval(model_no_window, 1, 8192)
        llama_kv = llama_family.kv_cache_eval(llama_equiv, 1, 8192)
        assert gemma_kv.bytes == llama_kv.bytes

    def test_kv_cache_scales_with_seq_len(self):
        """KV-cache should still grow with sequence length."""
        family = GemmaFamily()
        model = _make_gemma2_9b()
        short = family.kv_cache_eval(model, 1, 4096)  # within window
        long = family.kv_cache_eval(model, 1, 8192)  # exceeds window
        assert long.bytes > short.bytes

    def test_description_mentions_sliding_window(self):
        """When sliding window is active, description should mention it."""
        family = GemmaFamily()
        model = _make_gemma2_9b()
        est = family.kv_cache_eval(model, 1, 8192)
        assert "sliding" in est.description.lower()


# ---------------------------------------------------------------------------
# End-to-end via VRAMEstimator
# ---------------------------------------------------------------------------


class TestGemmaEndToEnd:
    """Full pipeline: VRAMEstimator with Gemma model."""

    def test_gemma_qlora_fits_on_3090(self):
        """Gemma 2 9B is similar size to Llama 8B, should fit on 3090."""
        estimator = VRAMEstimator()
        hw = _make_3090()
        breakdown = estimator.estimate(
            model=_make_gemma2_9b(),
            hardware=hw,
            method=TrainingMethod.QLORA,
            batch_size=2,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        total_gb = breakdown.steady_state_gb
        assert total_gb < hw.usable_vram_gb, (
            f"Gemma 2 9B QLoRA should fit on 3090. "
            f"{total_gb:.1f}GB > {hw.usable_vram_gb}GB"
        )

    def test_large_vocab_logits_buffer(self):
        """Gemma has vocab=256k -- logits buffer should be substantial."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_gemma2_9b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=1,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        logits_gb = breakdown.logits_buffer.gb
        # 1 * 1024 * 256000 * 4 bytes = ~1 GB
        assert logits_gb > 0.8, (
            f"Gemma 256k vocab logits at bs=1 seq=1024 should be ~1GB, got {logits_gb:.2f}GB"
        )

    def test_eval_spike_with_sliding_window(self):
        """Eval KV-cache spike should be computed and reflect sliding window."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_gemma2_9b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=1,
            seq_len=1024,
            eval_seq_len=8192,
            lora_config=LoRAConfig(rank=16),
        )
        assert breakdown.eval_kv_spike is not None
        assert breakdown.eval_kv_spike.bytes > 0
        assert "sliding" in breakdown.eval_kv_spike.description.lower()
