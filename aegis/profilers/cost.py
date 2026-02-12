from aegis.models.state import TrainingSpec, CostEstimate

# GPU pricing (USD/second) - source: modal.com/pricing
GPU_PRICING = {
    "a10g": 0.000306,  # $1.10/hr
    "t4": 0.000126,    # $0.45/hr
    "a100": 0.000793,  # $2.85/hr
}


def estimate_training_cost(spec: TrainingSpec) -> CostEstimate:
    """
    Deterministic cost estimation based on spec parameters.

    Cost formula: (num_samples * num_epochs * 1.2 overhead) / throughput * gpu_price_per_sec
    VRAM formula: base_model + optimizer_states + activation_memory
    Assumes 10k samples and 1000 samples/sec throughput (conservative for 272M models).
    """
    # Assume 10k samples (small dataset for demo)
    num_samples = 10000

    # Estimate training steps
    effective_batch_size = spec.micro_batch_size * spec.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * spec.num_epochs

    # Estimate duration (conservative: 1000 samples/sec)
    samples_per_sec = 1000
    estimated_seconds = (num_samples * spec.num_epochs) / samples_per_sec

    # Add 20% overhead for data loading, checkpointing
    estimated_seconds *= 1.2

    # Calculate cost
    gpu_price = GPU_PRICING.get(spec.target_gpu, GPU_PRICING["a10g"])
    estimated_cost = estimated_seconds * gpu_price

    # VRAM estimation (simplified)
    base_model_vram = 1.0  # ~1GB for 272M model
    optimizer_vram = base_model_vram * 4  # AdamW states
    activation_vram = (spec.micro_batch_size * spec.seq_len * 32) / 1e9  # fp32 bytes
    total_vram = base_model_vram + optimizer_vram + activation_vram

    return CostEstimate(
        estimated_cost_usd=round(estimated_cost, 4),
        estimated_vram_gb=round(total_vram, 2),
        estimated_duration_min=round(estimated_seconds / 60, 1),
        cost_breakdown={
            "gpu": spec.target_gpu,
            "gpu_price_usd_per_sec": gpu_price,
            "estimated_seconds": round(estimated_seconds, 1),
            "effective_batch_size": effective_batch_size,
            "total_steps": total_steps,
        }
    )
