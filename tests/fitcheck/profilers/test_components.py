"""Tests for architecture-independent VRAM component calculators.

Each test validates *behavior* -- that the computed memory scales correctly
with the inputs and produces numbers in the expected range -- not just that
a function returns non-None.
"""

import pytest
from fitcheck.models.profiles import ModelProfile, TrainingMethod, LoRAConfig
from fitcheck.profilers.vram.components import (
    weight_memory,
    optimizer_memory,
    gradient_memory,
    logits_buffer_memory,
    count_lora_params,
    get_trainable_params,
)


def _make_llama_8b() -> ModelProfile:
    """Llama-3.1-8B profile for testing."""
    return ModelProfile(
        model_id="meta-llama/Llama-3.1-8B",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,  # GQA
        intermediate_size=14336,
        vocab_size=128256,
        total_params=8_030_000_000,  # ~8.03B
        total_params_b=8.03,
    )


class TestWeightMemory:
    """Weight memory scales correctly with method and dtype."""

    def test_full_finetune_bf16(self):
        model = _make_llama_8b()
        est = weight_memory(model, TrainingMethod.FULL, "bfloat16")
        # 8.03B params * 2 bytes = ~16.06 GB
        gb = est.bytes / (1024**3)
        assert 14.5 < gb < 16.5, f"Expected ~16GB for full bf16, got {gb:.2f}GB"

    def test_full_finetune_fp32_is_double_bf16(self):
        model = _make_llama_8b()
        bf16 = weight_memory(model, TrainingMethod.FULL, "bfloat16")
        fp32 = weight_memory(model, TrainingMethod.FULL, "float32")
        assert fp32.bytes == pytest.approx(bf16.bytes * 2, rel=0.01)

    def test_qlora_base_much_smaller_than_full(self):
        """QLoRA NF4 base should be roughly 1/4 the size of bf16."""
        model = _make_llama_8b()
        full = weight_memory(model, TrainingMethod.FULL, "bfloat16")
        qlora = weight_memory(model, TrainingMethod.QLORA, "bfloat16")
        ratio = qlora.bytes / full.bytes
        # NF4 is 0.55 bytes/param vs 2 bytes/param = ~0.275 + small adapter overhead
        assert 0.2 < ratio < 0.4, f"QLoRA/Full ratio should be ~0.28, got {ratio:.3f}"

    def test_lora_adapters_add_small_overhead(self):
        """LoRA adapters should add <2% to base model weight memory."""
        model = _make_llama_8b()
        lora_cfg = LoRAConfig(rank=16)
        est = weight_memory(model, TrainingMethod.LORA, "bfloat16", lora_cfg)
        base_only = model.total_params * 2
        overhead_pct = (est.bytes - base_only) / base_only * 100
        assert overhead_pct < 2.0, f"LoRA overhead should be <2%, got {overhead_pct:.2f}%"


class TestLoRAParamCount:
    """LoRA parameter counting scales with rank and targets."""

    def test_rank_16_default_targets(self):
        model = _make_llama_8b()
        lora_cfg = LoRAConfig(rank=16)
        params = count_lora_params(model, lora_cfg)
        # 7 target modules * 32 layers * rank * (in + out) features
        # With GQA (k,v projections use kv_dim=1024 not hidden=4096),
        # rank 16 on Llama 8B produces ~42M adapter params
        assert 30_000_000 < params < 50_000_000

    def test_higher_rank_means_more_params(self):
        model = _make_llama_8b()
        r16 = count_lora_params(model, LoRAConfig(rank=16))
        r64 = count_lora_params(model, LoRAConfig(rank=64))
        # Params scale linearly with rank
        assert r64 == pytest.approx(r16 * 4, rel=0.01)

    def test_fewer_targets_means_fewer_params(self):
        model = _make_llama_8b()
        all_targets = LoRAConfig(rank=16)  # default: all 7 projections
        attn_only = LoRAConfig(rank=16, target_modules=["q_proj", "v_proj"])
        all_params = count_lora_params(model, all_targets)
        attn_params = count_lora_params(model, attn_only)
        assert attn_params < all_params


class TestOptimizerMemory:
    """Optimizer state memory depends on trainable params and optimizer type."""

    def test_adamw_8_bytes_per_param(self):
        est = optimizer_memory(1_000_000, "adamw")
        assert est.bytes == 8_000_000

    def test_8bit_adam_much_smaller(self):
        adamw = optimizer_memory(1_000_000, "adamw")
        adam8 = optimizer_memory(1_000_000, "paged_adamw_8bit")
        assert adam8.bytes == adamw.bytes / 4

    def test_full_finetune_optimizer_dominates(self):
        """For full fine-tune, optimizer states should be ~16GB for an 8B model."""
        params = 8_000_000_000
        est = optimizer_memory(params, "adamw")
        gb = est.bytes / (1024**3)
        assert 55 < gb < 65, f"Expected ~60GB for 8B AdamW states, got {gb:.2f}GB"


class TestGradientMemory:
    def test_gradients_match_dtype(self):
        est_bf16 = gradient_memory(1_000_000, "bfloat16")
        est_fp32 = gradient_memory(1_000_000, "float32")
        assert est_bf16.bytes == 2_000_000
        assert est_fp32.bytes == 4_000_000


class TestLogitsBuffer:
    """Logits buffer is the surprise OOM -- must be computed accurately."""

    def test_128k_vocab_single_sample(self):
        """Llama 3.x with 128k vocab: 1 * 4096 * 128000 * 4 = ~2GB."""
        est = logits_buffer_memory(batch_size=1, seq_len=4096, vocab_size=128000)
        gb = est.bytes / (1024**3)
        assert 1.8 < gb < 2.2, f"Expected ~2GB for 128k vocab, got {gb:.2f}GB"

    def test_scales_linearly_with_batch(self):
        bs1 = logits_buffer_memory(1, 1024, 128000)
        bs4 = logits_buffer_memory(4, 1024, 128000)
        assert bs4.bytes == bs1.bytes * 4

    def test_small_vocab_is_negligible(self):
        """32k vocab with short seqs should be small."""
        est = logits_buffer_memory(4, 512, 32000)
        gb = est.bytes / (1024**3)
        assert gb < 0.3


class TestTrainableParams:
    def test_full_finetune_all_params_trainable(self):
        model = _make_llama_8b()
        assert get_trainable_params(model, TrainingMethod.FULL) == model.total_params

    def test_lora_only_adapter_params_trainable(self):
        model = _make_llama_8b()
        trainable = get_trainable_params(model, TrainingMethod.LORA)
        assert trainable < model.total_params * 0.02  # <2% for rank 16
