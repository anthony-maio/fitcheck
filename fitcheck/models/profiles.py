"""Input profiles for the fitcheck solver pipeline.

These models represent the three inputs to any VRAM estimation:
the model being trained, the dataset, and the target hardware.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TrainingMethod(str, Enum):
    """Supported fine-tuning methods."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class LoRAConfig(BaseModel):
    """LoRA/QLoRA adapter configuration."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class ModelProfile(BaseModel):
    """Normalized model architecture metadata.

    Resolved from a HuggingFace config.json.  Contains everything
    needed to compute parameter counts and memory requirements.
    """

    model_id: str
    architecture: str  # e.g. "LlamaForCausalLM"
    family: str  # e.g. "llama", "gemma", "moe"

    # Core dimensions
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int  # for GQA; equals num_attention_heads if MHA
    intermediate_size: int
    vocab_size: int

    # Computed totals
    total_params: int  # total parameter count
    total_params_b: float  # total_params / 1e9 for display

    # MoE-specific (None for dense models)
    num_experts: int | None = None
    num_experts_per_token: int | None = None  # top-k routing

    # Metadata
    max_position_embeddings: int = 4096
    torch_dtype: str = "bfloat16"
    sliding_window: int | None = None  # Gemma 2: alternating layers use sliding window

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def active_params(self) -> int:
        """Parameters active per forward pass.  Equals total_params for dense
        models.  For MoE, excludes inactive expert MLP parameters."""
        if not self.is_moe or self.num_experts_per_token is None or self.num_experts is None:
            return self.total_params
        # gate + up + down projections, no bias (standard for modern SwiGLU MoE)
        mlp_per_expert = 3 * self.hidden_size * self.intermediate_size
        inactive_experts = self.num_experts - self.num_experts_per_token
        inactive_mlp_params = inactive_experts * mlp_per_expert * self.num_layers
        return self.total_params - inactive_mlp_params


class SeqLenStats(BaseModel):
    """Sequence length distribution from dataset analysis."""

    min: int
    mean: float
    p50: int
    p95: int
    p99: int
    max: int


class DatasetProfile(BaseModel):
    """Dataset metadata relevant to VRAM estimation."""

    source: str  # file path or HF dataset ID
    num_rows: int
    detected_format: Literal["alpaca", "sharegpt", "raw_text", "unknown"] = "unknown"
    seq_len_stats: SeqLenStats | None = None

    @property
    def effective_seq_len(self) -> int:
        """p95 sequence length, or a conservative default."""
        if self.seq_len_stats is not None:
            return self.seq_len_stats.p95
        return 512  # conservative default when no analysis available


class HardwareSpec(BaseModel):
    """GPU hardware specification with real-world overhead margins."""

    name: str  # e.g. "RTX 3090"
    total_vram_gb: float
    overhead_gb: float  # CUDA context + framework reservation
    memory_bandwidth_gbps: float
    fp16_tflops: float
    bf16_tflops: float

    @property
    def usable_vram_gb(self) -> float:
        return self.total_vram_gb - self.overhead_gb

    @property
    def usable_vram_bytes(self) -> int:
        return int(self.usable_vram_gb * (1024**3))
