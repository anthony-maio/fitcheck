"""Output models for the fitcheck solver pipeline.

These models represent the results of VRAM estimation and config
optimization -- the data behind the CLI report.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComponentEstimate(BaseModel):
    """A single VRAM component's estimate with explanation."""

    name: str
    bytes: int
    description: str = ""

    @property
    def gb(self) -> float:
        return self.bytes / (1024**3)

    @property
    def display(self) -> str:
        return f"{self.gb:.2f} GB"


class VRAMBreakdown(BaseModel):
    """Complete VRAM decomposition for a training configuration."""

    # The six components
    weights: ComponentEstimate
    optimizer: ComponentEstimate
    gradients: ComponentEstimate
    activations: ComponentEstimate
    logits_buffer: ComponentEstimate
    eval_kv_spike: ComponentEstimate | None = None  # periodic, not steady-state

    # Margins
    systematic_margin_bytes: int = 0  # framework overhead (already in usable VRAM)
    dynamic_margin_bytes: int = 0  # fragmentation, transient peaks

    @property
    def steady_state_bytes(self) -> int:
        """Total VRAM during training (excludes eval spike)."""
        return (
            self.weights.bytes
            + self.optimizer.bytes
            + self.gradients.bytes
            + self.activations.bytes
            + self.logits_buffer.bytes
        )

    @property
    def steady_state_gb(self) -> float:
        return self.steady_state_bytes / (1024**3)

    @property
    def peak_bytes(self) -> int:
        """Peak VRAM including eval spike if present."""
        spike = self.eval_kv_spike.bytes if self.eval_kv_spike else 0
        return self.steady_state_bytes + spike

    @property
    def range_low_bytes(self) -> int:
        return self.steady_state_bytes - self.dynamic_margin_bytes

    @property
    def range_high_bytes(self) -> int:
        return self.steady_state_bytes + self.dynamic_margin_bytes

    @property
    def range_low_gb(self) -> float:
        return self.range_low_bytes / (1024**3)

    @property
    def range_high_gb(self) -> float:
        return self.range_high_bytes / (1024**3)

    @property
    def components(self) -> list[ComponentEstimate]:
        """All steady-state components for iteration."""
        return [
            self.weights,
            self.optimizer,
            self.gradients,
            self.activations,
            self.logits_buffer,
        ]


class TrainingConfig(BaseModel):
    """A concrete training configuration the solver recommends."""

    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    seq_len: int
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"
    lora_rank: int | None = None
    lora_targets: list[str] | None = None

    # Computed VRAM for this config
    vram_breakdown: VRAMBreakdown | None = None

    # Reasoning annotations
    reasoning: dict[str, str] = Field(default_factory=dict)


class SolverResult(BaseModel):
    """Output of the ConfigSolver: three tiers plus warnings."""

    recommended: TrainingConfig
    aggressive: TrainingConfig | None = None
    fallbacks: list[TrainingConfig] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PlanReport(BaseModel):
    """Complete fitcheck plan output -- everything needed to render the CLI report."""

    model_id: str
    architecture_summary: str
    total_params_b: float
    vocab_size: int
    num_layers: int

    dataset_source: str
    dataset_rows: int
    dataset_format: str
    seq_len_stats: dict[str, float] | None = None
    seq_len_used: int
    seq_len_reasoning: str

    hardware_name: str
    total_vram_gb: float
    overhead_gb: float
    usable_vram_gb: float

    method: str
    trainable_params: int
    trainable_pct: float
    samples_per_epoch: float

    solver_result: SolverResult

    throughput_range: tuple[float, float] | None = None  # tok/sec low, high
    estimated_time_range: tuple[float, float] | None = None  # minutes low, high
    cost_local: float = 0.0
    cost_cloud: dict[str, float] | None = None  # {"A10G": 0.85, "A100": 2.40}
