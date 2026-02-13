"""Tests for LlamaFamily activation memory estimation.

These tests verify that activation memory scales correctly with
batch size, sequence length, and gradient checkpointing -- and
that the absolute values are in the right ballpark for known
model configurations.
"""

import pytest
from fitcheck.models.profiles import ModelProfile
from fitcheck.profilers.vram.families.llama import LlamaFamily


def _make_llama_8b() -> ModelProfile:
    return ModelProfile(
        model_id="meta-llama/Llama-3.1-8B",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        total_params=8_030_000_000,
        total_params_b=8.03,
    )


def _make_llama_70b() -> ModelProfile:
    return ModelProfile(
        model_id="meta-llama/Llama-3.1-70B",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        total_params=70_550_000_000,
        total_params_b=70.55,
    )


class TestActivationMemoryScaling:
    """Activation memory must scale correctly with each dimension."""

    def test_scales_linearly_with_batch_size(self):
        family = LlamaFamily()
        model = _make_llama_8b()
        bs1 = family.activation_memory(model, batch_size=1, seq_len=1024, grad_checkpointing=False)
        bs4 = family.activation_memory(model, batch_size=4, seq_len=1024, grad_checkpointing=False)
        # Not exactly 4x because attention scores scale quadratically with seq_len
        # but linearly with batch size -- so should be very close to 4x
        ratio = bs4.bytes / bs1.bytes
        assert 3.9 < ratio < 4.1, f"Expected ~4x scaling with batch, got {ratio:.2f}x"

    def test_scales_linearly_with_sequence_length(self):
        """With flash attention, activation memory scales linearly with seq_len
        (no materialized s*s attention matrix). 4x seq_len -> ~4x memory."""
        family = LlamaFamily()
        model = _make_llama_8b()
        s512 = family.activation_memory(model, 1, seq_len=512, grad_checkpointing=False)
        s2048 = family.activation_memory(model, 1, seq_len=2048, grad_checkpointing=False)
        ratio = s2048.bytes / s512.bytes
        assert 3.9 < ratio < 4.1, f"Expected ~4x scaling with 4x seq_len, got {ratio:.2f}x"

    def test_grad_checkpointing_reduces_memory(self):
        family = LlamaFamily()
        model = _make_llama_8b()
        no_ckpt = family.activation_memory(model, 1, 1024, grad_checkpointing=False)
        with_ckpt = family.activation_memory(model, 1, 1024, grad_checkpointing=True)
        # Grad checkpointing should reduce to roughly sqrt(32)/32 = ~18% for 32 layers
        ratio = with_ckpt.bytes / no_ckpt.bytes
        assert ratio < 0.35, f"Grad ckpt should reduce to <35%, got {ratio:.1%}"

    def test_larger_model_more_activation_memory(self):
        family = LlamaFamily()
        small = family.activation_memory(_make_llama_8b(), 1, 1024, False)
        large = family.activation_memory(_make_llama_70b(), 1, 1024, False)
        # 70B has 2x wider hidden, 2x wider intermediate, 2.5x more layers
        # = exactly 5x for the dominant terms (MLP intermediates + attention IO)
        assert large.bytes >= small.bytes * 5


class TestActivationMemoryAbsolute:
    """Sanity-check absolute values against known ranges."""

    def test_8b_bs4_seq1024_in_expected_range(self):
        """Llama 8B, batch 4, seq 1024 should be roughly 3-8 GB of activations."""
        family = LlamaFamily()
        est = family.activation_memory(_make_llama_8b(), 4, 1024, False)
        gb = est.bytes / (1024**3)
        assert 2.0 < gb < 12.0, f"Expected 3-8GB activations, got {gb:.2f}GB"

    def test_8b_bs1_seq512_is_small(self):
        """Small batch + short seq should be well under 1GB."""
        family = LlamaFamily()
        est = family.activation_memory(_make_llama_8b(), 1, 512, False)
        gb = est.bytes / (1024**3)
        assert gb < 2.0, f"Expected <2GB for bs=1 seq=512, got {gb:.2f}GB"


class TestKVCacheEval:
    """KV-cache for evaluation spike estimation."""

    def test_kv_cache_scales_with_seq_len(self):
        family = LlamaFamily()
        model = _make_llama_8b()
        short = family.kv_cache_eval(model, eval_batch_size=1, eval_seq_len=1024)
        long = family.kv_cache_eval(model, eval_batch_size=1, eval_seq_len=4096)
        assert long.bytes == pytest.approx(short.bytes * 4, rel=0.01)

    def test_gqa_reduces_kv_cache(self):
        """GQA (8 KV heads vs 32 Q heads) should make KV-cache smaller
        than it would be with full MHA."""
        family = LlamaFamily()
        model = _make_llama_8b()  # 8 KV heads
        est = family.kv_cache_eval(model, 1, 4096)
        # With GQA: 2 * 1 * 4096 * (8 * 128) * 2 * 32 layers
        # = 2 * 4096 * 1024 * 2 * 32 = ~512MB
        # Without GQA (32 KV heads): would be 4x larger
        gb = est.bytes / (1024**3)
        assert gb < 1.0, f"GQA KV-cache should be <1GB for 8B at seq=4096, got {gb:.2f}GB"
