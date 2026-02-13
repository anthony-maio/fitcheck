"""HuggingFace Hub model resolver.

Pulls config.json from the Hub, identifies the architecture family,
and returns a normalized ModelProfile with all dimensions needed for
VRAM estimation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fitcheck.models.profiles import ModelProfile

# Maps HuggingFace architecture strings to our family identifiers
_ARCHITECTURE_FAMILIES: dict[str, str] = {
    # LLaMA family (decoder-only, SwiGLU, RoPE, optional GQA)
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",
    "PhiForCausalLM": "llama",
    "Phi3ForCausalLM": "llama",
    "InternLM2ForCausalLM": "llama",
    "Starcoder2ForCausalLM": "llama",
    # Gemma family
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
    # MoE family
    "MixtralForCausalLM": "moe",
    "Qwen2MoeForCausalLM": "moe",
}


def resolve_model(model_id: str) -> ModelProfile:
    """Resolve a HuggingFace model ID to a ModelProfile.

    Fetches config.json via huggingface_hub, extracts architecture
    dimensions, and maps to the appropriate family.

    Args:
        model_id: HuggingFace model identifier (e.g. "meta-llama/Llama-3.1-8B")

    Returns:
        Normalized ModelProfile ready for VRAM estimation.

    Raises:
        ValueError: If the model's architecture is not supported.
    """
    config = _fetch_config(model_id)
    return _config_to_profile(model_id, config)


def resolve_from_config(model_id: str, config: dict[str, Any]) -> ModelProfile:
    """Build a ModelProfile from an already-loaded config dict.

    Useful for testing or when you already have the config.json contents.
    """
    return _config_to_profile(model_id, config)


def _fetch_config(model_id: str) -> dict[str, Any]:
    """Fetch config.json from HuggingFace Hub with caching."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model resolution. "
            "Install with: pip install huggingface_hub"
        )

    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
    )
    with open(config_path) as f:
        return json.load(f)


def _config_to_profile(model_id: str, config: dict[str, Any]) -> ModelProfile:
    """Convert a HuggingFace config dict to a ModelProfile."""
    # Identify architecture
    architectures = config.get("architectures", [])
    arch_str = architectures[0] if architectures else config.get("model_type", "unknown")

    family = _ARCHITECTURE_FAMILIES.get(arch_str)
    if family is None:
        supported = sorted(_ARCHITECTURE_FAMILIES.keys())
        raise ValueError(
            f"Unsupported architecture '{arch_str}' for model '{model_id}'. "
            f"Supported: {supported}"
        )

    # Extract dimensions (handle varying key names across model configs)
    hidden_size = config.get("hidden_size", config.get("d_model", 0))
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 0))

    # GQA: num_key_value_heads < num_attention_heads
    num_kv_heads = config.get(
        "num_key_value_heads",
        config.get("num_kv_heads", num_attention_heads),  # MHA if not specified
    )

    intermediate_size = config.get(
        "intermediate_size",
        config.get("ffn_dim", config.get("n_inner", hidden_size * 4)),
    )

    vocab_size = config.get("vocab_size", 32000)
    max_pos = config.get(
        "max_position_embeddings",
        config.get("max_seq_len", config.get("n_positions", 4096)),
    )

    # MoE-specific
    num_experts = config.get("num_local_experts", config.get("num_experts"))
    experts_per_token = config.get(
        "num_experts_per_tok",
        config.get("num_experts_per_token"),
    )

    # Compute total parameter count
    total_params = _compute_param_count(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_experts=num_experts,
    )

    torch_dtype = config.get("torch_dtype", "bfloat16")

    return ModelProfile(
        model_id=model_id,
        architecture=arch_str,
        family=family,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        total_params=total_params,
        total_params_b=total_params / 1e9,
        num_experts=num_experts,
        num_experts_per_token=experts_per_token,
        max_position_embeddings=max_pos,
        torch_dtype=torch_dtype,
    )


def _compute_param_count(
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    num_experts: int | None = None,
) -> int:
    """Compute total parameter count from architecture dimensions.

    Accounts for:
    - Embedding: vocab_size * hidden_size
    - Per-layer attention: Q + K + V projections + output projection
    - Per-layer MLP: gate + up + down projections (SwiGLU)
    - Per-layer norms: 2 * hidden_size (RMSNorm, weight only)
    - LM head: vocab_size * hidden_size (may be tied with embedding)
    - For MoE: each expert has its own MLP, shared attention
    """
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_kv_heads * head_dim

    # Embedding (input + output, often tied but both allocated)
    embedding = vocab_size * hidden_size
    lm_head = vocab_size * hidden_size  # assume untied for conservative estimate

    # Per-layer attention
    q_proj = hidden_size * hidden_size  # h -> h
    k_proj = hidden_size * kv_dim  # h -> kv_dim
    v_proj = hidden_size * kv_dim  # h -> kv_dim
    o_proj = hidden_size * hidden_size  # h -> h
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # Per-layer MLP (SwiGLU: gate, up, down)
    gate_proj = hidden_size * intermediate_size
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    mlp_per_layer = gate_proj + up_proj + down_proj

    # Norms: 2 RMSNorm per layer, each has hidden_size weights
    norms_per_layer = 2 * hidden_size

    # For MoE: shared attention + num_experts * MLP + router
    if num_experts is not None and num_experts > 1:
        router_per_layer = hidden_size * num_experts
        layer_params = (
            attn_per_layer
            + num_experts * mlp_per_layer
            + router_per_layer
            + norms_per_layer
        )
    else:
        layer_params = attn_per_layer + mlp_per_layer + norms_per_layer

    # Final norm
    final_norm = hidden_size

    total = embedding + (layer_params * num_layers) + final_norm + lm_head
    return total
