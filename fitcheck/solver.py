"""ConfigSolver: finds optimal training configs via constrained batch-size sweep.

Given a model, hardware, and training method, the solver calls VRAMEstimator
in a loop to find the largest batch size that fits, then produces recommended,
aggressive, and fallback configurations.
"""

from __future__ import annotations

import math

from fitcheck.models.profiles import (
    ModelProfile,
    HardwareSpec,
    TrainingMethod,
    LoRAConfig,
)
from fitcheck.models.results import (
    TrainingConfig,
    SolverResult,
    VRAMBreakdown,
)
from fitcheck.profilers.vram.engine import VRAMEstimator

# Headroom fraction for the "recommended" config.
# We want range_high_gb to leave at least this much of usable VRAM free.
_RECOMMENDED_HEADROOM = 0.15
RECOMMENDED_HEADROOM_PCT = _RECOMMENDED_HEADROOM * 100  # public, used by formatter

# Default optimizer per method — QLoRA/LoRA use paged 8-bit Adam,
# full fine-tune uses standard AdamW.
_DEFAULT_OPTIMIZERS: dict[TrainingMethod, str] = {
    TrainingMethod.FULL: "adamw",
    TrainingMethod.LORA: "paged_adamw_8bit",
    TrainingMethod.QLORA: "paged_adamw_8bit",
}


class ConfigSolver:
    """Find optimal training configurations via constrained batch-size sweep."""

    def __init__(self, estimator: VRAMEstimator | None = None):
        self._estimator = estimator or VRAMEstimator()

    def solve(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        seq_len: int = 512,
        lora_config: LoRAConfig | None = None,
        optimizer: str | None = None,
        target_effective_batch: int = 16,
        training_dtype: str = "bfloat16",
        eval_seq_len: int | None = None,
    ) -> SolverResult:
        """Find the best training config that fits in VRAM.

        Returns a SolverResult with recommended, aggressive, and fallback
        configurations, plus warnings about anything concerning.
        """
        opt = optimizer or _DEFAULT_OPTIMIZERS[method]
        lora_cfg = lora_config or LoRAConfig()
        usable_gb = hardware.usable_vram_gb

        if usable_gb <= 0:
            raise ValueError(
                f"Hardware '{hardware.name}' has no usable VRAM "
                f"(total={hardware.total_vram_gb} GB, overhead={hardware.overhead_gb} GB)"
            )

        # Step 1: Can we fit at all with bs=1, no grad checkpointing?
        bs1_breakdown = self._estimate(
            model,
            hardware,
            method,
            1,
            seq_len,
            lora_cfg,
            opt,
            grad_checkpointing=False,
            training_dtype=training_dtype,
            eval_seq_len=eval_seq_len,
        )

        grad_ckpt_needed = False

        if bs1_breakdown.range_high_gb > usable_gb:
            # Try with gradient checkpointing
            bs1_breakdown = self._estimate(
                model,
                hardware,
                method,
                1,
                seq_len,
                lora_cfg,
                opt,
                grad_checkpointing=True,
                training_dtype=training_dtype,
                eval_seq_len=eval_seq_len,
            )
            grad_ckpt_needed = True

            if bs1_breakdown.range_high_gb > usable_gb:
                # Does not fit even at bs=1 with grad checkpointing
                return self._does_not_fit_result(
                    model,
                    hardware,
                    method,
                    seq_len,
                    lora_cfg,
                    opt,
                    training_dtype,
                    target_effective_batch,
                    eval_seq_len,
                    bs1_breakdown,
                )

        # Step 2: Find max batch size via doubling
        max_recommended_bs = self._find_max_batch(
            model,
            hardware,
            method,
            seq_len,
            lora_cfg,
            opt,
            grad_ckpt_needed,
            training_dtype,
            eval_seq_len,
            headroom=_RECOMMENDED_HEADROOM,
        )

        # If bs=1 doesn't meet headroom target, try grad checkpointing
        if max_recommended_bs == 1 and not grad_ckpt_needed:
            bs1_with_ckpt = self._estimate(
                model,
                hardware,
                method,
                1,
                seq_len,
                lora_cfg,
                opt,
                grad_checkpointing=True,
                training_dtype=training_dtype,
                eval_seq_len=eval_seq_len,
            )
            threshold = usable_gb * (1 - _RECOMMENDED_HEADROOM)
            if bs1_with_ckpt.range_high_gb <= threshold:
                grad_ckpt_needed = True
                max_recommended_bs = self._find_max_batch(
                    model,
                    hardware,
                    method,
                    seq_len,
                    lora_cfg,
                    opt,
                    True,
                    training_dtype,
                    eval_seq_len,
                    headroom=_RECOMMENDED_HEADROOM,
                )

        max_aggressive_bs = self._find_max_batch(
            model,
            hardware,
            method,
            seq_len,
            lora_cfg,
            opt,
            grad_ckpt_needed,
            training_dtype,
            eval_seq_len,
            headroom=0.0,  # aggressive = fits within usable VRAM
        )

        # Step 3: Build recommended config
        rec_breakdown = self._estimate(
            model,
            hardware,
            method,
            max_recommended_bs,
            seq_len,
            lora_cfg,
            opt,
            grad_checkpointing=grad_ckpt_needed,
            training_dtype=training_dtype,
            eval_seq_len=eval_seq_len,
        )
        rec_accum = _grad_accum_steps(max_recommended_bs, target_effective_batch)
        recommended = TrainingConfig(
            micro_batch_size=max_recommended_bs,
            gradient_accumulation_steps=rec_accum,
            effective_batch_size=max_recommended_bs * rec_accum,
            seq_len=seq_len,
            gradient_checkpointing=grad_ckpt_needed,
            optimizer=opt,
            lora_rank=lora_cfg.rank if method != TrainingMethod.FULL else None,
            lora_targets=lora_cfg.target_modules if method != TrainingMethod.FULL else None,
            vram_breakdown=rec_breakdown,
            reasoning=self._build_reasoning(
                rec_breakdown,
                usable_gb,
                max_recommended_bs,
                grad_ckpt_needed,
                "recommended",
            ),
        )

        # Step 4: Build aggressive config (only if different from recommended)
        aggressive = None
        if max_aggressive_bs > max_recommended_bs:
            agg_breakdown = self._estimate(
                model,
                hardware,
                method,
                max_aggressive_bs,
                seq_len,
                lora_cfg,
                opt,
                grad_checkpointing=grad_ckpt_needed,
                training_dtype=training_dtype,
                eval_seq_len=eval_seq_len,
            )
            agg_accum = _grad_accum_steps(max_aggressive_bs, target_effective_batch)
            aggressive = TrainingConfig(
                micro_batch_size=max_aggressive_bs,
                gradient_accumulation_steps=agg_accum,
                effective_batch_size=max_aggressive_bs * agg_accum,
                seq_len=seq_len,
                gradient_checkpointing=grad_ckpt_needed,
                optimizer=opt,
                lora_rank=lora_cfg.rank if method != TrainingMethod.FULL else None,
                lora_targets=lora_cfg.target_modules if method != TrainingMethod.FULL else None,
                vram_breakdown=agg_breakdown,
                reasoning=self._build_reasoning(
                    agg_breakdown,
                    usable_gb,
                    max_aggressive_bs,
                    grad_ckpt_needed,
                    "aggressive",
                ),
            )

        # Step 5: Build fallback chain
        fallbacks = self._build_fallbacks(
            model,
            hardware,
            method,
            seq_len,
            lora_cfg,
            opt,
            training_dtype,
            target_effective_batch,
            eval_seq_len,
            max_recommended_bs,
            grad_ckpt_needed,
        )

        # Step 6: Warnings
        warnings = self._generate_warnings(
            rec_breakdown,
            usable_gb,
            grad_ckpt_needed,
            eval_seq_len,
        )

        return SolverResult(
            recommended=recommended,
            aggressive=aggressive,
            fallbacks=fallbacks,
            warnings=warnings,
        )

    def estimate_fixed(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        batch_size: int,
        seq_len: int = 512,
        lora_config: LoRAConfig | None = None,
        optimizer: str | None = None,
        target_effective_batch: int = 16,
        training_dtype: str = "bfloat16",
        eval_seq_len: int | None = None,
    ) -> SolverResult:
        """Estimate VRAM for a fixed batch size without running the solver search."""
        opt = optimizer or _DEFAULT_OPTIMIZERS[method]
        lora_cfg = lora_config or LoRAConfig()

        breakdown = self._estimate(
            model,
            hardware,
            method,
            batch_size,
            seq_len,
            lora_cfg,
            opt,
            grad_checkpointing=False,
            training_dtype=training_dtype,
            eval_seq_len=eval_seq_len,
        )
        accum = _grad_accum_steps(batch_size, target_effective_batch)
        config = TrainingConfig(
            micro_batch_size=batch_size,
            gradient_accumulation_steps=accum,
            effective_batch_size=batch_size * accum,
            seq_len=seq_len,
            optimizer=opt,
            lora_rank=lora_cfg.rank if method != TrainingMethod.FULL else None,
            lora_targets=(lora_cfg.target_modules if method != TrainingMethod.FULL else None),
            vram_breakdown=breakdown,
        )
        return SolverResult(recommended=config)

    def _estimate(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        batch_size: int,
        seq_len: int,
        lora_config: LoRAConfig,
        optimizer: str,
        grad_checkpointing: bool,
        training_dtype: str,
        eval_seq_len: int | None,
    ) -> VRAMBreakdown:
        return self._estimator.estimate(
            model=model,
            hardware=hardware,
            method=method,
            batch_size=batch_size,
            seq_len=seq_len,
            lora_config=lora_config,
            optimizer=optimizer,
            grad_checkpointing=grad_checkpointing,
            training_dtype=training_dtype,
            eval_seq_len=eval_seq_len,
        )

    def _find_max_batch(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        seq_len: int,
        lora_config: LoRAConfig,
        optimizer: str,
        grad_checkpointing: bool,
        training_dtype: str,
        eval_seq_len: int | None,
        headroom: float,
    ) -> int:
        """Find the largest batch size that fits with the given headroom.

        Uses doubling (1→2→4→8→...) to find the upper bound, then binary
        search to refine. Returns power-of-2 batch sizes since those are
        what people actually use.
        """
        usable_gb = hardware.usable_vram_gb
        threshold = usable_gb * (1 - headroom)

        # Doubling phase: find the first power-of-2 that doesn't fit
        bs = 1
        last_fit = 1
        while bs <= 256:  # sane upper bound
            breakdown = self._estimate(
                model,
                hardware,
                method,
                bs,
                seq_len,
                lora_config,
                optimizer,
                grad_checkpointing=grad_checkpointing,
                training_dtype=training_dtype,
                eval_seq_len=eval_seq_len,
            )
            if breakdown.range_high_gb <= threshold:
                last_fit = bs
                bs *= 2
            else:
                break
        else:
            # Everything up to 256 fits — return 256
            return last_fit

        # last_fit is the largest power-of-2 that fits
        return last_fit

    def _does_not_fit_result(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        seq_len: int,
        lora_config: LoRAConfig,
        optimizer: str,
        training_dtype: str,
        target_effective_batch: int,
        eval_seq_len: int | None,
        breakdown: VRAMBreakdown,
    ) -> SolverResult:
        """Build a SolverResult when the config doesn't fit at all."""
        accum = _grad_accum_steps(1, target_effective_batch)
        config = TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=accum,
            effective_batch_size=accum,
            seq_len=seq_len,
            gradient_checkpointing=True,
            optimizer=optimizer,
            lora_rank=lora_config.rank if method != TrainingMethod.FULL else None,
            lora_targets=(lora_config.target_modules if method != TrainingMethod.FULL else None),
            vram_breakdown=breakdown,
            reasoning={
                "verdict": "does_not_fit",
                "detail": (
                    f"Requires {breakdown.range_high_gb:.1f} GB but only "
                    f"{hardware.usable_vram_gb:.1f} GB usable"
                ),
            },
        )
        return SolverResult(
            recommended=config,
            warnings=[
                f"This configuration does not fit on {hardware.name}. "
                f"Estimated {breakdown.range_high_gb:.1f} GB needed, "
                f"{hardware.usable_vram_gb:.1f} GB available."
            ],
        )

    def _build_fallbacks(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        seq_len: int,
        lora_config: LoRAConfig,
        optimizer: str,
        training_dtype: str,
        target_effective_batch: int,
        eval_seq_len: int | None,
        current_bs: int,
        current_grad_ckpt: bool,
    ) -> list[TrainingConfig]:
        """Build a fallback chain of progressively more conservative configs.

        Chain: halve batch → enable grad ckpt → reduce rank → bs=1 + rank=8 + ckpt
        Only includes fallbacks that are different from the recommended config.
        """
        fallbacks = []
        seen: set[tuple[int, bool, int | None]] = set()

        # Track what recommended already uses
        rec_key = (
            current_bs,
            current_grad_ckpt,
            lora_config.rank if method != TrainingMethod.FULL else None,
        )
        seen.add(rec_key)

        candidates = self._fallback_candidates(
            current_bs,
            current_grad_ckpt,
            lora_config,
            method,
        )

        for bs, grad_ckpt, rank in candidates:
            key = (bs, grad_ckpt, rank)
            if key in seen:
                continue
            seen.add(key)

            fb_lora = (
                LoRAConfig(
                    rank=rank,
                    target_modules=lora_config.target_modules,
                )
                if rank is not None
                else lora_config
            )

            breakdown = self._estimate(
                model,
                hardware,
                method,
                bs,
                seq_len,
                fb_lora,
                optimizer,
                grad_checkpointing=grad_ckpt,
                training_dtype=training_dtype,
                eval_seq_len=eval_seq_len,
            )

            if breakdown.range_high_gb > hardware.usable_vram_gb:
                continue  # skip fallbacks that still don't fit

            accum = _grad_accum_steps(bs, target_effective_batch)
            fallbacks.append(
                TrainingConfig(
                    micro_batch_size=bs,
                    gradient_accumulation_steps=accum,
                    effective_batch_size=bs * accum,
                    seq_len=seq_len,
                    gradient_checkpointing=grad_ckpt,
                    optimizer=optimizer,
                    lora_rank=rank,
                    lora_targets=(
                        fb_lora.target_modules if method != TrainingMethod.FULL else None
                    ),
                    vram_breakdown=breakdown,
                    reasoning=self._build_reasoning(
                        breakdown,
                        hardware.usable_vram_gb,
                        bs,
                        grad_ckpt,
                        "fallback",
                    ),
                )
            )

        return fallbacks

    def _fallback_candidates(
        self,
        current_bs: int,
        current_grad_ckpt: bool,
        lora_config: LoRAConfig,
        method: TrainingMethod,
    ) -> list[tuple[int, bool, int | None]]:
        """Generate fallback candidates in order of decreasing quality."""
        rank = lora_config.rank if method != TrainingMethod.FULL else None
        candidates: list[tuple[int, bool, int | None]] = []

        # Halve batch size
        if current_bs > 1:
            candidates.append((max(1, current_bs // 2), current_grad_ckpt, rank))

        # Enable grad checkpointing (if not already)
        if not current_grad_ckpt:
            candidates.append((current_bs, True, rank))
            if current_bs > 1:
                candidates.append((max(1, current_bs // 2), True, rank))

        # Reduce rank (LoRA/QLoRA only)
        if rank is not None and rank > 8:
            candidates.append((current_bs, current_grad_ckpt, rank // 2))
            candidates.append((current_bs, True, rank // 2))

        # Nuclear option: bs=1, rank=8, grad ckpt
        if rank is not None:
            candidates.append((1, True, 8))
        else:
            candidates.append((1, True, rank))

        return candidates

    def _build_reasoning(
        self,
        breakdown: VRAMBreakdown,
        usable_gb: float,
        batch_size: int,
        grad_ckpt: bool,
        tier: str,
    ) -> dict[str, str]:
        headroom_gb = usable_gb - breakdown.range_high_gb
        headroom_pct = headroom_gb / usable_gb * 100

        reasoning: dict[str, str] = {
            "tier": tier,
            "batch_size": str(batch_size),
            "vram_used": f"{breakdown.steady_state_gb:.1f} GB",
            "headroom": f"{headroom_gb:.1f} GB ({headroom_pct:.0f}%)",
        }
        if grad_ckpt:
            reasoning["gradient_checkpointing"] = "enabled to reduce activation memory"

        return reasoning

    def _generate_warnings(
        self,
        breakdown: VRAMBreakdown,
        usable_gb: float,
        grad_ckpt_needed: bool,
        eval_seq_len: int | None,
    ) -> list[str]:
        warnings = []

        headroom_pct = (usable_gb - breakdown.range_high_gb) / usable_gb * 100
        if headroom_pct < 10:
            warnings.append(
                f"Tight VRAM headroom ({headroom_pct:.0f}%). "
                "OOM possible with longer sequences or larger batches."
            )

        if grad_ckpt_needed:
            warnings.append("Gradient checkpointing required -- expect ~30% slower training.")

        if eval_seq_len is not None and breakdown.eval_kv_spike is not None:
            spike_gb = breakdown.eval_kv_spike.gb
            if spike_gb > 1.0:
                warnings.append(
                    f"Eval KV-cache spike of {spike_gb:.1f} GB at seq_len={eval_seq_len}. "
                    "Reduce eval batch size or max eval length if OOM during eval."
                )

        if breakdown.logits_buffer.gb > 2.0:
            warnings.append(
                f"Logits buffer is {breakdown.logits_buffer.gb:.1f} GB "
                "(128k vocab). Consider chunked cross-entropy loss."
            )

        return warnings


def _grad_accum_steps(micro_batch: int, target_effective: int) -> int:
    """Compute gradient accumulation steps to approximate target effective batch."""
    return max(1, math.ceil(target_effective / micro_batch))
