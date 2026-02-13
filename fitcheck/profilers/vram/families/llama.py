"""LlamaFamily: VRAM estimation for LLaMA-architecture models.

Covers: Llama 3.x, Qwen 2.5, Mistral, Phi, and any decoder-only
transformer with SwiGLU MLP, RoPE embeddings, and optional GQA.

The activation memory formula accounts for:
- Self-attention: Q, K, V projections + attention scores + softmax output
- MLP: gate + up projections, SiLU activation, down projection
- Layer norms (RMSNorm): 2 per layer
- Residual connections

With gradient checkpointing, only sqrt(num_layers) layers store
activations; the rest are recomputed during the backward pass.
"""

from __future__ import annotations

import math

from fitcheck.models.profiles import ModelProfile, TrainingMethod
from fitcheck.models.results import ComponentEstimate


class LlamaFamily:
    """VRAM estimator for LLaMA-family architectures."""

    def activation_memory(
        self,
        config: ModelProfile,
        batch_size: int,
        seq_len: int,
        grad_checkpointing: bool,
    ) -> ComponentEstimate:
        """Compute activation memory for training.

        Assumes flash attention (standard since 2024), which processes
        attention in tiles and never materializes the full s*s score matrix.
        This dramatically reduces activation memory vs naive attention.

        Per-layer stored activations for backward pass:

        Attention block (flash attention):
          - Input to QKV projections: b * s * h (for linear backward)
          - Flash attention output: b * s * h (for O_proj backward)
          - Softmax logsumexp stats: b * n_heads * s in fp32 (small)

        MLP block (SwiGLU):
          - Input to gate/up projections: b * s * h (for linear backward)
          - Two intermediate tensors for backward through element-wise
            multiply + SiLU: 2 * b * s * intermediate

        Note: residual connections are in-place additions; RMSNorm inputs
        overlap with tensors already stored. No separate storage needed.
        """
        b = batch_size
        s = seq_len
        h = config.hidden_size
        n_heads = config.num_attention_heads
        inter = config.intermediate_size
        n_layers = config.num_layers
        bpe = 2  # bytes per element (bf16/fp16)

        # --- Attention stored activations (flash attention) ---
        # Input to QKV projections + flash attention output
        attn_stored = 2 * b * s * h * bpe
        # Softmax logsumexp statistics (fp32, one per query position per head)
        attn_stored += b * n_heads * s * 4

        # --- MLP stored activations (SwiGLU) ---
        # Input to gate/up projections
        mlp_stored = b * s * h * bpe
        # Gate output (for SiLU backward) + up output (for multiply backward)
        mlp_stored += 2 * b * s * inter * bpe

        per_layer = attn_stored + mlp_stored

        # --- Gradient checkpointing ---
        if grad_checkpointing:
            # Store activations for sqrt(n) layers, recompute the rest
            effective_layers = math.ceil(math.sqrt(n_layers))
            desc_suffix = f" (grad ckpt: {effective_layers}/{n_layers} layers stored)"
        else:
            effective_layers = n_layers
            desc_suffix = ""

        total_bytes = per_layer * effective_layers

        gb = total_bytes / (1024**3)
        attn_gb = attn_stored * effective_layers / (1024**3)
        mlp_gb = mlp_stored * effective_layers / (1024**3)
        desc = (
            f"bs={b}, seq={s}, {n_layers} layers{desc_suffix}: "
            f"attn={attn_gb:.2f}GB + mlp={mlp_gb:.2f}GB = {gb:.2f}GB"
        )
        return ComponentEstimate(
            name="Activations",
            bytes=total_bytes,
            description=desc,
        )

    def kv_cache_eval(
        self,
        config: ModelProfile,
        eval_batch_size: int,
        eval_seq_len: int,
    ) -> ComponentEstimate:
        """Compute KV-cache memory during evaluation.

        KV-cache per layer: 2 * batch * seq * kv_dim * 2 bytes
        (K and V, each batch * seq * kv_dim in bf16)
        """
        kv_dim = config.num_kv_heads * config.head_dim
        per_layer = 2 * eval_batch_size * eval_seq_len * kv_dim * 2  # bf16
        total = per_layer * config.num_layers
        gb = total / (1024**3)
        desc = (
            f"Eval KV-cache: bs={eval_batch_size}, seq={eval_seq_len}, "
            f"{config.num_layers} layers, kv_dim={kv_dim}: {gb:.2f} GB"
        )
        return ComponentEstimate(
            name="Eval KV-cache spike",
            bytes=total,
            description=desc,
        )

    def weight_memory(
        self,
        config: ModelProfile,
        method: TrainingMethod,
        training_dtype: str,
    ) -> ComponentEstimate | None:
        """LlamaFamily uses the default shared weight calculation."""
        return None
