"""Cloud GPU pricing for counterfactual cost display.

Shows what a local training run would cost on cloud infrastructure.
Prices are approximate hourly rates in USD from major providers.

Sources: Modal (modal.com/pricing), RunPod (runpod.io/gpu-cloud),
Lambda Labs (lambdalabs.com/service/gpu-cloud).
Updated: 2026-02-14.  Prices fluctuate; displayed as approximate.
"""

from __future__ import annotations


# GPU key -> list of (provider, usd_per_hour)
# Only cloud-available GPUs are listed.  Consumer cards (3090, 4090) are not.
_CLOUD_PRICING: dict[str, list[tuple[str, float]]] = {
    "a10g": [
        ("Modal", 0.53),
        ("RunPod", 0.38),
    ],
    "a100-40gb": [
        ("Lambda", 1.29),
        ("RunPod", 1.10),
    ],
    "a100-80gb": [
        ("Lambda", 1.99),
        ("RunPod", 1.64),
    ],
    "h100": [
        ("Lambda", 2.49),
        ("RunPod", 3.49),
    ],
}

_DISPLAY_NAMES: dict[str, str] = {
    "a10g": "A10G",
    "a100-40gb": "A100 40GB",
    "a100-80gb": "A100 80GB",
    "h100": "H100",
}


def get_cloud_prices() -> dict[str, float]:
    """Return cheapest hourly rate per cloud GPU for report display.

    Returns:
        Dict mapping display name to cheapest USD/hour rate.
        Example: {"A10G": 0.38, "A100 80GB": 1.64, "H100": 2.49}
    """
    result = {}
    for gpu_key, providers in _CLOUD_PRICING.items():
        cheapest = min(rate for _, rate in providers)
        display = _DISPLAY_NAMES.get(gpu_key, gpu_key.upper())
        result[display] = cheapest
    return result
