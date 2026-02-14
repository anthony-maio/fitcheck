"""Architecture-independent VRAM component calculators.

These functions compute memory for components that depend on parameter
counts and dtypes but NOT on architecture-specific details like
attention patterns.  Architecture-specific components (activations,
KV-cache) are computed by the family classes.
"""

from __future__ import annotations

from fitcheck.models.profiles import TrainingMethod, LoRAConfig, ModelProfile
from fitcheck.models.results import ComponentEstimate

# Bytes per parameter for each dtype
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    # NF4: 4-bit with quantization metadata overhead ~10%
    "nf4": 0.55,
}


def _bytes_per_param(dtype: str) -> float:
    key = dtype.lower().replace("torch.", "")
    if key not in _DTYPE_BYTES:
        raise ValueError(f"Unknown dtype '{dtype}'. Supported: {list(_DTYPE_BYTES.keys())}")
    return _DTYPE_BYTES[key]


_MLP_MODULES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}


def count_lora_params(
    model: ModelProfile,
    lora_config: LoRAConfig,
) -> int:
    """Count total trainable LoRA adapter parameters.

    For each target module in each layer:
        A matrix: (in_features, rank) = in_features * rank
        B matrix: (rank, out_features) = rank * out_features

    For MoE models, MLP modules (gate_proj, up_proj, down_proj) exist
    per-expert.  LoRA applied to these targets trains adapters on ALL
    experts, so MLP adapter params scale with num_experts.
    """
    rank = lora_config.rank

    # Split targets into attention (shared) and MLP (per-expert in MoE)
    attn_targets = [t for t in lora_config.target_modules if t not in _MLP_MODULES]
    mlp_targets = [t for t in lora_config.target_modules if t in _MLP_MODULES]

    attn_dims = _get_module_dimensions(model, attn_targets)
    mlp_dims = _get_module_dimensions(model, mlp_targets)

    per_layer_attn = sum((in_f * rank) + (rank * out_f) for in_f, out_f in attn_dims)
    per_layer_mlp = sum((in_f * rank) + (rank * out_f) for in_f, out_f in mlp_dims)

    # MoE: MLP adapters exist per expert
    num_experts = model.num_experts if model.is_moe else 1

    return (per_layer_attn + per_layer_mlp * num_experts) * model.num_layers


def _get_module_dimensions(
    model: ModelProfile,
    target_modules: list[str],
) -> list[tuple[int, int]]:
    """Map target module names to (in_features, out_features) pairs.

    Handles standard transformer naming conventions.
    """
    h = model.hidden_size
    kv_dim = model.num_kv_heads * model.head_dim
    inter = model.intermediate_size

    # Standard module dimension lookup
    # Covers LLaMA, Mistral, Qwen, Phi, Gemma naming conventions
    module_map: dict[str, tuple[int, int]] = {
        # Attention projections
        "q_proj": (h, h),
        "k_proj": (h, kv_dim),
        "v_proj": (h, kv_dim),
        "o_proj": (h, h),
        # MLP projections (LLaMA-style SwiGLU)
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
        # Alternate naming (GPT-2, some custom models)
        "c_attn": (h, 3 * h),  # fused QKV
        "c_proj": (h, h),
        "w1": (h, inter),
        "w2": (inter, h),
    }

    dims = []
    for name in target_modules:
        if name in module_map:
            dims.append(module_map[name])
        # Skip unknown modules silently -- user might have model-specific targets
    return dims


def weight_memory(
    model: ModelProfile,
    method: TrainingMethod,
    training_dtype: str = "bfloat16",
    lora_config: LoRAConfig | None = None,
) -> ComponentEstimate:
    """Compute VRAM for model weights.

    - Full fine-tune: all params in training dtype
    - LoRA: base model in training dtype + adapters in training dtype
    - QLoRA: base model in NF4 + adapters in training dtype
    """
    if method == TrainingMethod.FULL:
        bpp = _bytes_per_param(training_dtype)
        total_bytes = int(model.total_params * bpp)
        desc = f"{model.total_params_b:.2f}B params in {training_dtype}"

    elif method == TrainingMethod.LORA:
        bpp = _bytes_per_param(training_dtype)
        base_bytes = int(model.total_params * bpp)
        lora_cfg = lora_config or LoRAConfig()
        adapter_params = count_lora_params(model, lora_cfg)
        adapter_bytes = int(adapter_params * bpp)
        total_bytes = base_bytes + adapter_bytes
        desc = (
            f"Base: {model.total_params_b:.2f}B in {training_dtype} "
            f"+ adapters: {adapter_params / 1e6:.1f}M in {training_dtype}"
        )

    elif method == TrainingMethod.QLORA:
        base_bytes = int(model.total_params * _bytes_per_param("nf4"))
        lora_cfg = lora_config or LoRAConfig()
        adapter_params = count_lora_params(model, lora_cfg)
        adapter_bytes = int(adapter_params * _bytes_per_param(training_dtype))
        total_bytes = base_bytes + adapter_bytes
        desc = (
            f"Base: {model.total_params_b:.2f}B in NF4 "
            f"+ adapters: {adapter_params / 1e6:.1f}M in {training_dtype}"
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    if model.is_moe:
        desc += (
            f" (MoE: all {model.num_experts} experts on GPU, "
            f"~{model.active_params / 1e9:.1f}B active per forward pass)"
        )

    return ComponentEstimate(name="Model weights", bytes=total_bytes, description=desc)


def optimizer_memory(
    trainable_params: int,
    optimizer: str = "adamw",
) -> ComponentEstimate:
    """Compute VRAM for optimizer states.

    AdamW: 2 fp32 states (momentum + variance) per trainable param = 8 bytes/param.
    8-bit Adam: ~2 bytes per param (quantized states).
    SGD with momentum: 1 fp32 state = 4 bytes/param.
    """
    opt = optimizer.lower()
    if opt in ("adamw", "adam"):
        # 2 states x 4 bytes (fp32) per trainable param
        total_bytes = trainable_params * 8
        desc = f"AdamW: 2 fp32 states for {trainable_params / 1e6:.1f}M trainable params"
    elif opt in ("adamw_8bit", "paged_adamw_8bit", "adam_8bit"):
        total_bytes = trainable_params * 2
        desc = f"8-bit Adam: quantized states for {trainable_params / 1e6:.1f}M params"
    elif opt == "sgd":
        total_bytes = trainable_params * 4
        desc = f"SGD+momentum: 1 fp32 state for {trainable_params / 1e6:.1f}M params"
    else:
        # Fall back to AdamW assumption
        total_bytes = trainable_params * 8
        desc = f"Unknown optimizer '{optimizer}', assuming AdamW (8 bytes/param)"

    return ComponentEstimate(name="Optimizer states", bytes=total_bytes, description=desc)


def gradient_memory(
    trainable_params: int,
    training_dtype: str = "bfloat16",
) -> ComponentEstimate:
    """Compute VRAM for gradients.  One gradient per trainable param in training dtype."""
    bpp = _bytes_per_param(training_dtype)
    total_bytes = int(trainable_params * bpp)
    desc = f"{trainable_params / 1e6:.1f}M trainable params in {training_dtype}"
    return ComponentEstimate(name="Gradients", bytes=total_bytes, description=desc)


def logits_buffer_memory(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> ComponentEstimate:
    """Compute VRAM for the logits tensor during loss computation.

    The cross-entropy loss requires logits in float32:
    batch_size * seq_len * vocab_size * 4 bytes.

    For Llama 3.x (vocab 128k): 1 * 4096 * 128000 * 4 = ~2 GB per sample.
    This is one of the most common surprise OOMs.
    """
    total_bytes = batch_size * seq_len * vocab_size * 4  # float32
    gb = total_bytes / (1024**3)
    desc = f"{batch_size} x {seq_len} x {vocab_size:,} x 4B = {gb:.2f} GB"
    if gb > 2.0:
        desc += " (consider chunked cross-entropy)"
    return ComponentEstimate(name="Logits buffer", bytes=total_bytes, description=desc)


def get_trainable_params(
    model: ModelProfile,
    method: TrainingMethod,
    lora_config: LoRAConfig | None = None,
) -> int:
    """Return the number of trainable parameters for the given method."""
    if method == TrainingMethod.FULL:
        return model.total_params
    lora_cfg = lora_config or LoRAConfig()
    return count_lora_params(model, lora_cfg)
