"""GemmaFamily: VRAM estimation for Gemma architectures.

Covers: Gemma 2, Gemma 3, and any Gemma variant.

Gemma uses GeGLU (memory-equivalent to SwiGLU) and RoPE, so
activation memory is identical to LlamaFamily.  The key difference
is Gemma 2's alternating sliding window attention, which reduces
KV-cache memory during evaluation: even-indexed layers use full
attention, odd-indexed layers use a sliding window.
"""

from __future__ import annotations

from fitcheck.models.profiles import ModelProfile
from fitcheck.models.results import ComponentEstimate
from fitcheck.profilers.vram.families.llama import LlamaFamily


class GemmaFamily(LlamaFamily):
    """VRAM estimator for Gemma-family architectures.

    Inherits activation_memory and weight_memory from LlamaFamily.
    Overrides kv_cache_eval to account for alternating sliding window.
    """

    def kv_cache_eval(
        self,
        config: ModelProfile,
        eval_batch_size: int,
        eval_seq_len: int,
    ) -> ComponentEstimate:
        """Compute KV-cache memory during evaluation.

        Gemma 2 uses sliding window attention on alternating layers:
          - Even layers (0, 2, 4, ...): full attention
          - Odd layers (1, 3, 5, ...): sliding window

        If no sliding_window is set, falls back to full attention
        on all layers (same as LlamaFamily).
        """
        window = config.sliding_window
        if window is None or eval_seq_len <= window:
            # No sliding window or seq fits within window -- same as dense
            return super().kv_cache_eval(config, eval_batch_size, eval_seq_len)

        kv_dim = config.num_kv_heads * config.head_dim
        n_layers = config.num_layers

        # Even layers: full attention
        full_attn_layers = (n_layers + 1) // 2  # ceil(n/2)
        # Odd layers: sliding window
        sliding_layers = n_layers // 2

        full_per_layer = 2 * eval_batch_size * eval_seq_len * kv_dim * 2  # bf16
        sliding_per_layer = 2 * eval_batch_size * window * kv_dim * 2  # bf16

        total = full_attn_layers * full_per_layer + sliding_layers * sliding_per_layer
        gb = total / (1024**3)
        full_gb = full_attn_layers * full_per_layer / (1024**3)
        sliding_gb = sliding_layers * sliding_per_layer / (1024**3)
        desc = (
            f"Eval KV-cache: bs={eval_batch_size}, seq={eval_seq_len}, "
            f"{n_layers} layers (sliding window={window}): "
            f"{full_attn_layers} full-attn={full_gb:.2f}GB + "
            f"{sliding_layers} sliding={sliding_gb:.2f}GB = {gb:.2f} GB"
        )
        return ComponentEstimate(
            name="Eval KV-cache spike",
            bytes=total,
            description=desc,
        )
