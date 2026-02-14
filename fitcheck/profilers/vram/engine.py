"""VRAMEstimator: top-level orchestrator for VRAM breakdown computation.

Combines architecture-independent component calculators with
architecture-specific family implementations to produce a complete
VRAMBreakdown for any supported training configuration.
"""

from __future__ import annotations

from fitcheck.models.profiles import (
    ModelProfile,
    HardwareSpec,
    TrainingMethod,
    LoRAConfig,
)
from fitcheck.models.results import VRAMBreakdown
from fitcheck.profilers.vram import components
from fitcheck.profilers.vram.families.base import ArchitectureFamily
from fitcheck.profilers.vram.families.gemma import GemmaFamily
from fitcheck.profilers.vram.families.llama import LlamaFamily
from fitcheck.profilers.vram.families.moe import MoEFamily

# Family registry: maps family identifier to implementation
_FAMILIES: dict[str, ArchitectureFamily] = {
    "llama": LlamaFamily(),
    "gemma": GemmaFamily(),
    "moe": MoEFamily(),
}


def _get_family(model: ModelProfile) -> ArchitectureFamily:
    """Look up the architecture family for a model."""
    family = _FAMILIES.get(model.family)
    if family is None:
        supported = sorted(_FAMILIES.keys())
        raise ValueError(
            f"No architecture family registered for '{model.family}' "
            f"(model: {model.model_id}). Supported families: {supported}"
        )
    return family


# Dynamic margin as a fraction of total steady-state VRAM.
# Accounts for memory fragmentation, transient optimizer peaks,
# cuDNN workspace buffers.  Empirically ~5-8% on most hardware.
_DYNAMIC_MARGIN_FRACTION = 0.065


class VRAMEstimator:
    """Compute a complete VRAM breakdown for a training configuration.

    This is the core of fitcheck.  Given a model profile, hardware spec,
    and training configuration, it computes each VRAM component and
    produces a VRAMBreakdown with confidence bounds.
    """

    def estimate(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        method: TrainingMethod,
        batch_size: int,
        seq_len: int,
        lora_config: LoRAConfig | None = None,
        optimizer: str = "adamw",
        grad_checkpointing: bool = False,
        training_dtype: str = "bfloat16",
        eval_seq_len: int | None = None,
        eval_batch_size: int = 1,
    ) -> VRAMBreakdown:
        """Produce a full VRAM breakdown.

        Args:
            model: Resolved model profile from HF Hub.
            hardware: Target GPU specification.
            method: Training method (full/lora/qlora).
            batch_size: Micro batch size (not effective batch).
            seq_len: Sequence length for VRAM estimation (typically p95).
            lora_config: LoRA configuration (required for lora/qlora).
            optimizer: Optimizer name for state size calculation.
            grad_checkpointing: Whether gradient checkpointing is enabled.
            training_dtype: Training precision (bfloat16, float16, float32).
            eval_seq_len: Max sequence length during eval (for KV-cache spike).
                          If None, no eval spike is computed.
            eval_batch_size: Batch size during evaluation.

        Returns:
            VRAMBreakdown with all components, margins, and confidence bounds.
        """
        family = _get_family(model)
        lora_cfg = lora_config or LoRAConfig()

        # --- Component 1: Model weights ---
        # Check if family has a custom weight calculation (MoE needs this)
        custom_weights = family.weight_memory(model, method, training_dtype)
        if custom_weights is not None:
            weight_est = custom_weights
        else:
            weight_est = components.weight_memory(model, method, training_dtype, lora_cfg)

        # --- Component 2: Optimizer states ---
        trainable = components.get_trainable_params(model, method, lora_cfg)
        optimizer_est = components.optimizer_memory(trainable, optimizer)

        # --- Component 3: Gradients ---
        gradient_est = components.gradient_memory(trainable, training_dtype)

        # --- Component 4: Activations (architecture-specific) ---
        activation_est = family.activation_memory(model, batch_size, seq_len, grad_checkpointing)

        # --- Component 5: Logits buffer ---
        logits_est = components.logits_buffer_memory(batch_size, seq_len, model.vocab_size)

        # --- Component 6: Eval KV-cache spike (optional) ---
        eval_spike = None
        if eval_seq_len is not None:
            eval_spike = family.kv_cache_eval(model, eval_batch_size, eval_seq_len)

        # --- Margins ---
        steady_state = (
            weight_est.bytes
            + optimizer_est.bytes
            + gradient_est.bytes
            + activation_est.bytes
            + logits_est.bytes
        )
        dynamic_margin = int(steady_state * _DYNAMIC_MARGIN_FRACTION)

        return VRAMBreakdown(
            weights=weight_est,
            optimizer=optimizer_est,
            gradients=gradient_est,
            activations=activation_est,
            logits_buffer=logits_est,
            eval_kv_spike=eval_spike,
            systematic_margin_bytes=int(hardware.overhead_gb * (1024**3)),
            dynamic_margin_bytes=dynamic_margin,
        )
