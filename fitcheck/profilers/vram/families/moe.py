"""MoEFamily: VRAM estimation for Mixture-of-Experts architectures.

Covers: Mixtral 8x7B, Qwen-MoE, and any sparse MoE transformer with
top-k expert routing.

Key difference from dense models: MLP activations scale with
experts_per_token (top-k), not num_experts (total).  A Mixtral 8x7B
with top-2 routing has MLP activation memory similar to a ~13B dense
model, even though all 47B parameters are loaded to GPU.

v1 assumes all experts are loaded to GPU.  CPU offloading of inactive
experts is deferred to v1.1.
"""

from __future__ import annotations

import math

from fitcheck.models.profiles import ModelProfile, TrainingMethod
from fitcheck.models.results import ComponentEstimate


class MoEFamily:
    """VRAM estimator for Mixture-of-Experts architectures."""

    def activation_memory(
        self,
        config: ModelProfile,
        batch_size: int,
        seq_len: int,
        grad_checkpointing: bool,
    ) -> ComponentEstimate:
        """Compute activation memory for MoE training.

        MoE activation memory has three components:

        1. Attention (shared, identical to dense):
           - Input to QKV projections: b * s * h
           - Flash attention output: b * s * h
           - Softmax logsumexp stats: b * n_heads * s in fp32

        2. Expert MLP (proportional to experts_per_token, not num_experts):
           Per token, only top-k experts execute their MLP.
           - Input to gate/up projections: k * b * s * h
           - Two intermediates per expert: k * 2 * b * s * intermediate

        3. Router gating logits: b * s * num_experts in fp32 (small).
        """
        b = batch_size
        s = seq_len
        h = config.hidden_size
        n_heads = config.num_attention_heads
        inter = config.intermediate_size
        n_layers = config.num_layers
        k = config.num_experts_per_token or 2
        n_experts = config.num_experts or 8
        bpe = 2  # bytes per element (bf16/fp16)

        # --- Attention stored activations (shared, same as dense) ---
        attn_stored = 2 * b * s * h * bpe
        attn_stored += b * n_heads * s * 4  # logsumexp in fp32

        # --- Expert MLP stored activations (top-k experts per token) ---
        mlp_per_expert = b * s * h * bpe + 2 * b * s * inter * bpe
        mlp_stored = k * mlp_per_expert

        # --- Router gating logits ---
        router_stored = b * s * n_experts * 4  # fp32 for softmax

        per_layer = attn_stored + mlp_stored + router_stored

        # --- Gradient checkpointing ---
        if grad_checkpointing:
            effective_layers = math.ceil(math.sqrt(n_layers))
            desc_suffix = f" (grad ckpt: {effective_layers}/{n_layers} layers stored)"
        else:
            effective_layers = n_layers
            desc_suffix = ""

        total_bytes = per_layer * effective_layers

        # Build descriptive annotation
        attn_gb = attn_stored * effective_layers / (1024**3)
        mlp_gb = mlp_stored * effective_layers / (1024**3)
        router_gb = router_stored * effective_layers / (1024**3)
        total_gb = total_bytes / (1024**3)
        desc = (
            f"MoE: top-{k} of {n_experts} experts active per token. "
            f"bs={b}, seq={s}, {n_layers} layers{desc_suffix}: "
            f"attn={attn_gb:.2f}GB + mlp={mlp_gb:.2f}GB ({k} experts) "
            f"+ router={router_gb:.3f}GB = {total_gb:.2f}GB"
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

        Attention is shared (not per-expert), so KV-cache formula is
        identical to dense models: 2 * batch * seq * kv_dim * 2 per layer.
        """
        kv_dim = config.num_kv_heads * config.head_dim
        per_layer = 2 * eval_batch_size * eval_seq_len * kv_dim * 2  # bf16
        total = per_layer * config.num_layers
        gb = total / (1024**3)
        desc = (
            f"Eval KV-cache: bs={eval_batch_size}, seq={eval_seq_len}, "
            f"{config.num_layers} layers, kv_dim={kv_dim}: {gb:.2f} GB "
            f"(shared attention, not per-expert)"
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
        """MoE uses the shared weight calculation.

        The byte count is correct (total_params includes all experts).
        The description is enhanced by the shared function when it
        detects config.is_moe.
        """
        return None
