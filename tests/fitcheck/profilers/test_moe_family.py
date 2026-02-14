"""Tests for MoEFamily VRAM estimation.

Validates that MoE-specific activation memory scales with
experts_per_token (not num_experts), that weight descriptions
annotate all-experts-on-GPU, and that LoRA param counting
correctly handles per-expert MLP modules.
"""

import pytest

from fitcheck.models.profiles import (
    HardwareSpec,
    LoRAConfig,
    ModelProfile,
    TrainingMethod,
)
from fitcheck.models.results import ComponentEstimate
from fitcheck.profilers.vram import components
from fitcheck.profilers.vram.engine import VRAMEstimator
from fitcheck.profilers.vram.families.moe import MoEFamily


def _make_mixtral_8x7b() -> ModelProfile:
    """Real Mixtral 8x7B dimensions."""
    return ModelProfile(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        architecture="MixtralForCausalLM",
        family="moe",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        total_params=46_700_000_000,
        total_params_b=46.7,
        num_experts=8,
        num_experts_per_token=2,
    )


def _make_dense_same_dims() -> ModelProfile:
    """A dense model with same hidden/layer dims as one Mixtral expert.

    Used as a comparison baseline: Mixtral attention is shared like a
    dense model, and MLP activations scale with experts_per_token.
    """
    return ModelProfile(
        model_id="dense-comparison",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        total_params=7_000_000_000,
        total_params_b=7.0,
    )


def _make_a100_80gb() -> HardwareSpec:
    return HardwareSpec(
        name="NVIDIA A100 80GB",
        total_vram_gb=80.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=2039.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
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
# ModelProfile.active_params
# ---------------------------------------------------------------------------


class TestActiveParams:
    """active_params correctly computes MoE active parameter count."""

    def test_mixtral_active_params_approximately_13b(self):
        """Mixtral 8x7B with top-2 routing: ~12-14B active params."""
        model = _make_mixtral_8x7b()
        active_b = model.active_params / 1e9
        assert 12.0 < active_b < 14.0, (
            f"Mixtral active params should be ~13B, got {active_b:.1f}B"
        )

    def test_dense_model_active_equals_total(self):
        """Dense models: active_params == total_params."""
        model = _make_dense_same_dims()
        assert model.active_params == model.total_params

    def test_active_less_than_total_for_moe(self):
        """MoE active_params must be strictly less than total_params."""
        model = _make_mixtral_8x7b()
        assert model.active_params < model.total_params


# ---------------------------------------------------------------------------
# Activation memory
# ---------------------------------------------------------------------------


class TestMoEActivationMemory:
    """MoE activation memory scales with experts_per_token, not num_experts."""

    def test_moe_activations_less_than_all_experts_active(self):
        """Top-2 of 8 experts: MLP activations should be far less than 8x.

        Compare MoE activation to what a dense model with 8x the MLP
        would produce (same attention, 8x intermediate_size).
        """
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        moe_act = family.activation_memory(model, batch_size=1, seq_len=1024, grad_checkpointing=False)

        # If all 8 experts were active, MLP cost would be 4x higher (8 vs 2).
        # With top-2, MLP is 25% of all-experts cost.
        # Verify total is well below the 4x-MLP scenario.
        from fitcheck.profilers.vram.families.llama import LlamaFamily
        llama = LlamaFamily()
        dense_act = llama.activation_memory(
            _make_dense_same_dims(), batch_size=1, seq_len=1024, grad_checkpointing=False,
        )
        # MoE with top-2 has 2x the MLP of a single expert.
        # Dense comparison has 1x MLP. So MoE should be roughly 1.5-2.5x dense
        # (MLP is part of total alongside attention which is the same).
        ratio = moe_act.bytes / dense_act.bytes
        assert 1.3 < ratio < 3.0, (
            f"MoE (top-2/8) should have ~1.5-2.5x activations vs dense, got {ratio:.2f}x"
        )

    def test_scales_linearly_with_batch_size(self):
        """Doubling batch size should roughly double activation memory."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        bs1 = family.activation_memory(model, 1, 1024, grad_checkpointing=False)
        bs4 = family.activation_memory(model, 4, 1024, grad_checkpointing=False)
        ratio = bs4.bytes / bs1.bytes
        assert 3.9 < ratio < 4.1, f"Expected ~4x with 4x batch, got {ratio:.2f}x"

    def test_scales_linearly_with_seq_len(self):
        """With flash attention, 4x seq_len -> ~4x activation memory."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        s512 = family.activation_memory(model, 1, 512, grad_checkpointing=False)
        s2048 = family.activation_memory(model, 1, 2048, grad_checkpointing=False)
        ratio = s2048.bytes / s512.bytes
        assert 3.9 < ratio < 4.1, f"Expected ~4x with 4x seq_len, got {ratio:.2f}x"

    def test_router_overhead_is_small(self):
        """Router gating logits should be < 1% of total activations."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        est = family.activation_memory(model, 4, 1024, grad_checkpointing=False)

        # Router cost per layer: batch * seq * num_experts * 4 bytes
        b, s, n_experts, n_layers = 4, 1024, 8, 32
        router_total = b * s * n_experts * 4 * n_layers
        router_pct = router_total / est.bytes * 100
        assert router_pct < 1.0, f"Router should be <1% of activations, got {router_pct:.2f}%"

    def test_grad_checkpointing_reduces_memory(self):
        """Gradient checkpointing: sqrt(32)/32 = ~18% for 32 layers."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        without = family.activation_memory(model, 1, 1024, grad_checkpointing=False)
        with_ckpt = family.activation_memory(model, 1, 1024, grad_checkpointing=True)
        ratio = with_ckpt.bytes / without.bytes
        assert ratio < 0.35, f"Grad ckpt should reduce to <35%, got {ratio:.1%}"

    def test_description_mentions_expert_routing(self):
        """The description should annotate the MoE routing behavior."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        est = family.activation_memory(model, 1, 1024, grad_checkpointing=False)
        assert "top-2" in est.description
        assert "8 experts" in est.description


# ---------------------------------------------------------------------------
# KV-cache eval
# ---------------------------------------------------------------------------


class TestMoEKVCache:
    """KV-cache for MoE uses shared attention, not per-expert."""

    def test_kv_cache_scales_with_seq_len(self):
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        short = family.kv_cache_eval(model, 1, 1024)
        long = family.kv_cache_eval(model, 1, 4096)
        assert long.bytes == pytest.approx(short.bytes * 4, rel=0.01)

    def test_kv_cache_same_as_dense_equivalent(self):
        """MoE KV-cache should match dense model with same attention config."""
        from fitcheck.profilers.vram.families.llama import LlamaFamily
        moe_family = MoEFamily()
        llama_family = LlamaFamily()
        moe_model = _make_mixtral_8x7b()
        dense_model = _make_dense_same_dims()

        moe_kv = moe_family.kv_cache_eval(moe_model, 1, 4096)
        dense_kv = llama_family.kv_cache_eval(dense_model, 1, 4096)
        assert moe_kv.bytes == dense_kv.bytes, "KV-cache is in shared attention, not per-expert"


# ---------------------------------------------------------------------------
# Weight memory (via shared components)
# ---------------------------------------------------------------------------


class TestMoEWeightMemory:
    """Weight memory for MoE includes all experts on GPU."""

    def test_qlora_weights_too_large_for_3090(self):
        """Mixtral NF4 weights: 46.7B * 0.55 = ~25.7 GB > 22.8 GB usable on 3090."""
        model = _make_mixtral_8x7b()
        est = components.weight_memory(model, TrainingMethod.QLORA, "bfloat16")
        weight_gb = est.bytes / (1024**3)
        hw = _make_3090()
        assert weight_gb > hw.usable_vram_gb, (
            f"Mixtral QLoRA weights ({weight_gb:.1f} GB) should exceed "
            f"3090 usable VRAM ({hw.usable_vram_gb} GB)"
        )

    def test_qlora_weights_fit_on_a100(self):
        """Mixtral NF4 weights should fit comfortably on A100 80GB."""
        model = _make_mixtral_8x7b()
        est = components.weight_memory(model, TrainingMethod.QLORA, "bfloat16")
        weight_gb = est.bytes / (1024**3)
        hw = _make_a100_80gb()
        assert weight_gb < hw.usable_vram_gb

    def test_description_mentions_all_experts(self):
        """Weight description should annotate MoE expert count and active params."""
        model = _make_mixtral_8x7b()
        est = components.weight_memory(model, TrainingMethod.QLORA, "bfloat16")
        assert "all 8 experts" in est.description.lower()
        assert "active per forward pass" in est.description.lower()

    def test_moe_weight_memory_returns_none(self):
        """MoEFamily.weight_memory() returns None, deferring to shared default."""
        family = MoEFamily()
        model = _make_mixtral_8x7b()
        assert family.weight_memory(model, TrainingMethod.QLORA, "bfloat16") is None


# ---------------------------------------------------------------------------
# LoRA param counting for MoE
# ---------------------------------------------------------------------------


class TestMoELoRAParams:
    """LoRA param counting accounts for per-expert MLP modules."""

    def test_mlp_targets_multiply_by_num_experts(self):
        """MLP LoRA targets on MoE should produce num_experts times more params
        than the same targets on a dense model with identical dimensions."""
        moe = _make_mixtral_8x7b()
        dense = _make_dense_same_dims()
        cfg = LoRAConfig(rank=16, target_modules=["gate_proj", "up_proj", "down_proj"])

        moe_params = components.count_lora_params(moe, cfg)
        dense_params = components.count_lora_params(dense, cfg)
        ratio = moe_params / dense_params
        assert ratio == pytest.approx(8.0, rel=0.01), (
            f"MLP LoRA on 8-expert MoE should be 8x dense, got {ratio:.2f}x"
        )

    def test_attn_only_targets_same_as_dense(self):
        """Attention-only LoRA targets are shared, so MoE == dense count."""
        moe = _make_mixtral_8x7b()
        dense = _make_dense_same_dims()
        cfg = LoRAConfig(rank=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

        moe_params = components.count_lora_params(moe, cfg)
        dense_params = components.count_lora_params(dense, cfg)
        assert moe_params == dense_params, "Attention LoRA is shared, not per-expert"

    def test_default_targets_large_adapter_on_moe(self):
        """Default 7 targets on Mixtral: 4 attn + 3 MLP*8 = much larger adapter."""
        moe = _make_mixtral_8x7b()
        dense = _make_dense_same_dims()
        cfg = LoRAConfig(rank=16)  # default targets

        moe_params = components.count_lora_params(moe, cfg)
        dense_params = components.count_lora_params(dense, cfg)
        # 4 shared attn + 3 * 8 MLP = 28 target modules per layer vs 7 for dense
        # Ratio should be between 2x and 5x (depends on relative sizes of attn vs MLP)
        ratio = moe_params / dense_params
        assert ratio > 2.0, (
            f"MoE default LoRA should be >2x dense due to per-expert MLPs, got {ratio:.2f}x"
        )


# ---------------------------------------------------------------------------
# End-to-end via VRAMEstimator
# ---------------------------------------------------------------------------


class TestMoEEndToEnd:
    """Full pipeline: VRAMEstimator with MoE model."""

    def test_mixtral_qlora_fits_on_a100(self):
        """Mixtral QLoRA on A100 80GB should fit with headroom."""
        estimator = VRAMEstimator()
        hw = _make_a100_80gb()
        breakdown = estimator.estimate(
            model=_make_mixtral_8x7b(),
            hardware=hw,
            method=TrainingMethod.QLORA,
            batch_size=1,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        total_gb = breakdown.steady_state_gb
        assert total_gb < hw.usable_vram_gb, (
            f"Mixtral QLoRA on A100 should fit. {total_gb:.1f}GB > {hw.usable_vram_gb}GB"
        )
        # Sanity: should be in 30-55 GB range
        assert 25.0 < total_gb < 55.0, (
            f"Mixtral QLoRA total should be ~30-50GB, got {total_gb:.1f}GB"
        )

    def test_mixtral_qlora_does_not_fit_on_3090(self):
        """Mixtral QLoRA on 3090: NF4 weights alone exceed usable VRAM."""
        estimator = VRAMEstimator()
        hw = _make_3090()
        breakdown = estimator.estimate(
            model=_make_mixtral_8x7b(),
            hardware=hw,
            method=TrainingMethod.QLORA,
            batch_size=1,
            seq_len=512,
            lora_config=LoRAConfig(rank=16),
        )
        total_gb = breakdown.steady_state_gb
        assert total_gb > hw.usable_vram_gb, (
            f"Mixtral QLoRA should NOT fit on 3090. "
            f"{total_gb:.1f}GB should exceed {hw.usable_vram_gb}GB"
        )

    def test_all_components_present(self):
        """All VRAM components should have positive byte counts."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_mixtral_8x7b(),
            hardware=_make_a100_80gb(),
            method=TrainingMethod.QLORA,
            batch_size=2,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        assert breakdown.weights.bytes > 0
        assert breakdown.optimizer.bytes > 0
        assert breakdown.gradients.bytes > 0
        assert breakdown.activations.bytes > 0
        assert breakdown.logits_buffer.bytes > 0
