"""Architecture family protocol for VRAM estimation.

Each architecture family (LLaMA, Gemma, MoE) implements this protocol
with family-specific formulas for activation memory, KV-cache sizing,
and optionally custom weight memory calculations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fitcheck.models.profiles import ModelProfile, TrainingMethod
from fitcheck.models.results import ComponentEstimate


@runtime_checkable
class ArchitectureFamily(Protocol):
    """Protocol for architecture-specific VRAM computations.

    Each family must implement activation_memory and kv_cache_eval.
    weight_memory is optional -- return None to use the shared
    default from components.py.
    """

    def activation_memory(
        self,
        config: ModelProfile,
        batch_size: int,
        seq_len: int,
        grad_checkpointing: bool,
    ) -> ComponentEstimate:
        """Compute activation memory for the forward + backward pass.

        This is the architecture-specific component that depends on
        attention patterns, MLP structure, and normalization layers.
        """
        ...

    def kv_cache_eval(
        self,
        config: ModelProfile,
        eval_batch_size: int,
        eval_seq_len: int,
    ) -> ComponentEstimate:
        """Compute KV-cache memory during evaluation steps.

        This is the periodic spike above training steady-state that
        causes surprise OOMs on the first eval checkpoint.
        """
        ...

    def weight_memory(
        self,
        config: ModelProfile,
        method: TrainingMethod,
        training_dtype: str,
    ) -> ComponentEstimate | None:
        """Optional override for architecture-specific weight memory.

        MoE models need this to handle CPU offloading of inactive experts.
        Return None to use the default shared calculation.
        """
        ...
