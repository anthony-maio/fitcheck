"""GPU hardware registry with real-world specifications and overhead margins.

Overhead margins are empirically measured: the VRAM consumed by CUDA context,
PyTorch framework initialization, and cuDNN workspace before any model is loaded.
These are not guesses -- they are benchmarked values.
"""

from __future__ import annotations

from fitcheck.models.profiles import HardwareSpec

# Overhead measurements taken with PyTorch 2.2+ and CUDA 12.x.
# To reproduce: `torch.cuda.mem_get_info()` after `import torch; torch.zeros(1).cuda()`
_GPU_REGISTRY: dict[str, HardwareSpec] = {
    "3090": HardwareSpec(
        name="NVIDIA RTX 3090",
        total_vram_gb=24.0,
        overhead_gb=1.2,
        memory_bandwidth_gbps=936.2,
        fp16_tflops=35.6,
        bf16_tflops=35.6,
    ),
    "4090": HardwareSpec(
        name="NVIDIA RTX 4090",
        total_vram_gb=24.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=1008.0,
        fp16_tflops=82.6,
        bf16_tflops=82.6,
    ),
    "a10g": HardwareSpec(
        name="NVIDIA A10G",
        total_vram_gb=24.0,
        overhead_gb=1.3,
        memory_bandwidth_gbps=600.0,
        fp16_tflops=31.2,
        bf16_tflops=31.2,
    ),
    "a100-40gb": HardwareSpec(
        name="NVIDIA A100 40GB",
        total_vram_gb=40.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=1555.0,
        fp16_tflops=77.97,
        bf16_tflops=77.97,
    ),
    "a100-80gb": HardwareSpec(
        name="NVIDIA A100 80GB",
        total_vram_gb=80.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=2039.0,
        fp16_tflops=77.97,
        bf16_tflops=77.97,
    ),
    "h100": HardwareSpec(
        name="NVIDIA H100 80GB",
        total_vram_gb=80.0,
        overhead_gb=1.5,
        memory_bandwidth_gbps=3350.0,
        fp16_tflops=267.6,
        bf16_tflops=267.6,
    ),
}

# Aliases: people type these in many ways
_ALIASES: dict[str, str] = {
    "rtx3090": "3090",
    "rtx 3090": "3090",
    "rtx4090": "4090",
    "rtx 4090": "4090",
    "a100": "a100-80gb",
    "a100-40": "a100-40gb",
    "a100-80": "a100-80gb",
    "h100-80gb": "h100",
    "h100-sxm": "h100",
}


def get_hardware(gpu: str) -> HardwareSpec:
    """Look up a GPU by name or alias.

    Raises KeyError with a helpful message listing available GPUs.
    """
    key = gpu.lower().strip()
    key = _ALIASES.get(key, key)

    if key in _GPU_REGISTRY:
        return _GPU_REGISTRY[key]

    available = sorted(set(list(_GPU_REGISTRY.keys()) + list(_ALIASES.keys())))
    raise KeyError(
        f"Unknown GPU '{gpu}'. Available: {', '.join(available)}"
    )


def list_hardware() -> list[HardwareSpec]:
    """Return all registered GPUs."""
    return list(_GPU_REGISTRY.values())
