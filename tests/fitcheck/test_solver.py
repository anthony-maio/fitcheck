"""Behavioral tests for ConfigSolver.

These test real solver behavior on real model/hardware configs — not mocks.
The solver calls VRAMEstimator which computes real VRAM from real dimensions.
"""

from fitcheck.models.profiles import (
    ModelProfile,
    HardwareSpec,
    TrainingMethod,
    LoRAConfig,
)
from fitcheck.solver import ConfigSolver


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


def _make_h100() -> HardwareSpec:
    return HardwareSpec(
        name="NVIDIA H100 80GB",
        total_vram_gb=80.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=3350.0,
        fp16_tflops=267.6,
        bf16_tflops=267.6,
    )


class TestQLoRA8BOnRTX3090:
    """The canonical use case: QLoRA Llama 8B on consumer hardware."""

    def test_recommended_config_fits_with_headroom(self):
        """Solver should find a config that fits with >= 15% headroom."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
            lora_config=LoRAConfig(rank=16),
        )
        rec = result.recommended
        assert rec.vram_breakdown is not None

        usable = _make_3090().usable_vram_gb
        high = rec.vram_breakdown.range_high_gb
        headroom_pct = (usable - high) / usable * 100
        assert headroom_pct >= 15, (
            f"Recommended config should have >= 15% headroom, got {headroom_pct:.0f}%"
        )

    def test_recommended_uses_8bit_optimizer(self):
        """QLoRA should default to paged_adamw_8bit, not full AdamW."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
        )
        assert result.recommended.optimizer == "paged_adamw_8bit"

    def test_batch_size_greater_than_one(self):
        """QLoRA 8B on 3090 should fit more than bs=1."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        assert result.recommended.micro_batch_size > 1, (
            "QLoRA 8B on 3090 should easily fit bs > 1 at seq_len=512"
        )

    def test_effective_batch_equals_micro_times_accum(self):
        """effective_batch_size = micro_batch_size * gradient_accumulation_steps."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
        )
        rec = result.recommended
        assert rec.effective_batch_size == (rec.micro_batch_size * rec.gradient_accumulation_steps)

    def test_no_gradient_checkpointing_needed(self):
        """QLoRA 8B at seq=512 should NOT need grad checkpointing on 3090."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        assert not result.recommended.gradient_checkpointing, (
            "QLoRA 8B seq=512 shouldn't need grad checkpointing on 3090"
        )


class TestFullFineTuneOnRTX3090:
    """Full fine-tune of 8B should not fit on a 3090."""

    def test_does_not_fit(self):
        """Full FT 8B needs ~80GB — way over 24GB."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.FULL,
        )
        assert any("does not fit" in w.lower() for w in result.warnings), (
            f"Expected 'does not fit' warning, got: {result.warnings}"
        )

    def test_uses_adamw_optimizer(self):
        """Full fine-tune should default to AdamW, not 8-bit."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.FULL,
        )
        assert result.recommended.optimizer == "adamw"


class TestLoRA8BOnRTX3090:
    """LoRA (not QLoRA) uses bf16 base weights -- tighter fit than QLoRA."""

    def test_lora_uses_8bit_optimizer(self):
        """LoRA should default to paged_adamw_8bit like QLoRA."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.LORA,
            seq_len=512,
        )
        assert result.recommended.optimizer == "paged_adamw_8bit"

    def test_lora_uses_more_vram_than_qlora(self):
        """LoRA keeps base weights in bf16, QLoRA quantizes to NF4."""
        solver = ConfigSolver()
        lora_result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.LORA,
            seq_len=512,
        )
        qlora_result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        lora_vram = lora_result.recommended.vram_breakdown.steady_state_gb
        qlora_vram = qlora_result.recommended.vram_breakdown.steady_state_gb
        assert lora_vram > qlora_vram, (
            f"LoRA ({lora_vram:.1f} GB) should use more VRAM than QLoRA ({qlora_vram:.1f} GB)"
        )


class TestQLoRA8BOnH100:
    """Same model on much bigger GPU — should get larger batch sizes."""

    def test_larger_batch_than_3090(self):
        """H100 has 78.5 GB usable — should fit much larger batches."""
        solver = ConfigSolver()
        result_3090 = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        result_h100 = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_h100(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        assert (
            result_h100.recommended.micro_batch_size > result_3090.recommended.micro_batch_size
        ), (
            f"H100 batch ({result_h100.recommended.micro_batch_size}) should be "
            f"larger than 3090 batch ({result_3090.recommended.micro_batch_size})"
        )


class TestAggressiveVsRecommended:
    """Aggressive config should be equal or more aggressive than recommended."""

    def test_aggressive_batch_gte_recommended(self):
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        if result.aggressive is not None:
            assert result.aggressive.micro_batch_size >= result.recommended.micro_batch_size


class TestFallbackChain:
    """Fallbacks should be ordered by decreasing quality."""

    def test_fallbacks_are_more_conservative(self):
        """Each fallback should use less VRAM than the recommended config."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        rec_bs = result.recommended.micro_batch_size
        for fb in result.fallbacks:
            # Fallbacks should have smaller batch, or rank, or enable ckpt
            is_smaller = fb.micro_batch_size < rec_bs
            has_ckpt = fb.gradient_checkpointing and not result.recommended.gradient_checkpointing
            has_lower_rank = (
                fb.lora_rank is not None
                and result.recommended.lora_rank is not None
                and fb.lora_rank < result.recommended.lora_rank
            )
            assert is_smaller or has_ckpt or has_lower_rank, (
                f"Fallback bs={fb.micro_batch_size}, ckpt={fb.gradient_checkpointing}, "
                f"rank={fb.lora_rank} is not more conservative than recommended "
                f"bs={rec_bs}, ckpt={result.recommended.gradient_checkpointing}, "
                f"rank={result.recommended.lora_rank}"
            )

    def test_fallbacks_all_fit(self):
        """Every fallback in the chain should actually fit in VRAM."""
        solver = ConfigSolver()
        hw = _make_3090()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=hw,
            method=TrainingMethod.QLORA,
            seq_len=512,
        )
        for fb in result.fallbacks:
            assert fb.vram_breakdown is not None
            assert fb.vram_breakdown.range_high_gb <= hw.usable_vram_gb, (
                f"Fallback bs={fb.micro_batch_size} doesn't fit: "
                f"{fb.vram_breakdown.range_high_gb:.1f} GB > {hw.usable_vram_gb:.1f} GB"
            )


class TestGradCheckpointingOnlyWhenNeeded:
    """Solver should only enable grad checkpointing when necessary."""

    def test_qlora_short_seq_no_checkpointing(self):
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=256,
        )
        assert not result.recommended.gradient_checkpointing

    def test_longer_seq_may_need_checkpointing(self):
        """At very long seq lengths, even QLoRA might need checkpointing."""
        solver = ConfigSolver()
        result = solver.solve(
            model=_make_llama_8b(),
            hardware=_make_3090(),
            method=TrainingMethod.QLORA,
            seq_len=4096,
        )
        # At seq_len=4096, the solver should still find something that fits.
        # It may or may not need checkpointing — just verify it produces a result.
        assert result.recommended.vram_breakdown is not None
        assert result.recommended.vram_breakdown.range_high_gb <= _make_3090().usable_vram_gb
