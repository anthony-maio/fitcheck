"""End-to-end tests for VRAMEstimator.

These are the integration tests that verify the full estimation pipeline
produces numbers in the right ballpark for real-world configurations.
"""

import pytest
from fitcheck.models.profiles import (
    ModelProfile,
    HardwareSpec,
    TrainingMethod,
    LoRAConfig,
)
from fitcheck.profilers.vram.engine import VRAMEstimator


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


def _make_3090() -> HardwareSpec:
    return HardwareSpec(
        name="NVIDIA RTX 3090",
        total_vram_gb=24.0,
        overhead_gb=1.2,
        memory_bandwidth_gbps=936.2,
        fp16_tflops=35.6,
        bf16_tflops=35.6,
    )


class TestQLoRAOnRTX3090:
    """The canonical example: QLoRA Llama 8B on a 3090."""

    def test_fits_comfortably(self):
        """QLoRA 8B with bs=4 seq=1024 should fit on a 3090 with headroom."""
        estimator = VRAMEstimator()
        hw = _make_3090()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=hw,
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        total_gb = breakdown.steady_state_gb
        assert total_gb < hw.usable_vram_gb, (
            f"QLoRA 8B bs=4 should fit on 3090. "
            f"Estimated {total_gb:.1f}GB > usable {hw.usable_vram_gb}GB"
        )

    def test_qlora_base_weights_are_small(self):
        """NF4 base weights for 8B should be ~4.4GB."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=1,
            seq_len=512,
            lora_config=LoRAConfig(rank=16),
        )
        weight_gb = breakdown.weights.gb
        assert 3.5 < weight_gb < 5.0, f"QLoRA weights should be ~4.4GB, got {weight_gb:.2f}GB"

    def test_logits_buffer_is_significant(self):
        """With 128k vocab, logits buffer should be non-trivial."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
            lora_config=LoRAConfig(rank=16),
        )
        logits_gb = breakdown.logits_buffer.gb
        assert logits_gb > 1.0, (
            f"128k vocab logits buffer should be >1GB at bs=4 seq=1024, got {logits_gb:.2f}GB"
        )

    def test_confidence_range_is_narrow(self):
        """Dynamic margin should be a small fraction of total."""
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
        )
        range_width = breakdown.range_high_gb - breakdown.range_low_gb
        total = breakdown.steady_state_gb
        pct = range_width / total * 100
        assert pct < 20, f"Dynamic margin should be <20% of total, got {pct:.1f}%"


class TestFullFineTuneDoesNotFit:
    """Full fine-tune of 8B in bf16 should NOT fit on a 3090."""

    def test_exceeds_3090_vram(self):
        """Full FT needs ~16GB weights + ~60GB optimizer = way over 24GB."""
        estimator = VRAMEstimator()
        hw = _make_3090()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=hw,
            method=TrainingMethod.FULL,
            batch_size=1,
            seq_len=512,
        )
        total_gb = breakdown.steady_state_gb
        assert total_gb > hw.usable_vram_gb, (
            f"Full FT 8B should NOT fit on 3090. "
            f"Estimated {total_gb:.1f}GB should exceed {hw.usable_vram_gb}GB"
        )


class TestEvalSpike:
    """Eval KV-cache spike is computed when eval_seq_len is provided."""

    def test_eval_spike_present(self):
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
            eval_seq_len=4096,
        )
        assert breakdown.eval_kv_spike is not None
        assert breakdown.eval_kv_spike.bytes > 0

    def test_eval_spike_absent_when_not_requested(self):
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
        )
        assert breakdown.eval_kv_spike is None

    def test_peak_includes_eval_spike(self):
        estimator = VRAMEstimator()
        breakdown = estimator.estimate(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
            eval_seq_len=4096,
        )
        assert breakdown.peak_bytes > breakdown.steady_state_bytes


class TestGradCheckpointingEffect:
    """Gradient checkpointing should meaningfully reduce total VRAM."""

    def test_reduces_total_vram(self):
        estimator = VRAMEstimator()
        kwargs = dict(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
        )
        without = estimator.estimate(**kwargs, grad_checkpointing=False)
        with_gc = estimator.estimate(**kwargs, grad_checkpointing=True)
        assert with_gc.steady_state_bytes < without.steady_state_bytes

    def test_only_activations_change(self):
        """Grad checkpointing should only affect activation memory."""
        estimator = VRAMEstimator()
        kwargs = dict(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            batch_size=4,
            seq_len=1024,
        )
        without = estimator.estimate(**kwargs, grad_checkpointing=False)
        with_gc = estimator.estimate(**kwargs, grad_checkpointing=True)

        assert with_gc.weights.bytes == without.weights.bytes
        assert with_gc.optimizer.bytes == without.optimizer.bytes
        assert with_gc.gradients.bytes == without.gradients.bytes
        assert with_gc.activations.bytes < without.activations.bytes
