import pytest
from aegis.models.state import TrainingSpec, CostEstimate
from aegis.profilers.cost import estimate_training_cost, GPU_PRICING


def test_gpu_pricing_constants():
    assert "a10g" in GPU_PRICING
    assert "t4" in GPU_PRICING
    assert GPU_PRICING["a10g"] == 0.000306  # $1.10/hr


def test_estimate_cost_basic_lora():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    assert isinstance(estimate, CostEstimate)
    assert estimate.estimated_cost_usd > 0
    assert estimate.estimated_cost_usd < 1.0  # Should be cheap
    assert estimate.estimated_vram_gb > 0
    assert estimate.estimated_duration_min > 0


def test_estimate_cost_breakdown_contains_required_fields():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    assert "gpu" in estimate.cost_breakdown
    assert "gpu_price_usd_per_sec" in estimate.cost_breakdown
    assert "estimated_seconds" in estimate.cost_breakdown
    assert "effective_batch_size" in estimate.cost_breakdown
    assert "total_steps" in estimate.cost_breakdown


def test_estimate_cost_scales_with_batch_size():
    spec_small = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=2,
        target_gpu="a10g"
    )
    spec_large = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=8,
        target_gpu="a10g"
    )

    estimate_small = estimate_training_cost(spec_small)
    estimate_large = estimate_training_cost(spec_large)

    # Smaller batch should cost similar (same samples) but slightly more overhead
    # Duration should be very similar (same number of samples processed)
    assert abs(estimate_small.estimated_duration_min - estimate_large.estimated_duration_min) < 1


def test_estimate_cost_different_gpu():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        target_gpu="a10g"
    )

    estimate_a10g = estimate_training_cost(spec)

    spec_t4 = spec.model_copy(update={"target_gpu": "t4"})
    estimate_t4 = estimate_training_cost(spec_t4)

    # T4 is cheaper per second, so total cost should be lower
    assert estimate_t4.estimated_cost_usd < estimate_a10g.estimated_cost_usd


def test_vram_estimation_includes_components():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=4,
        seq_len=512,
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    # VRAM should account for model weights, optimizer, and activations
    # For 272M model, should be several GB minimum
    assert estimate.estimated_vram_gb > 1.0
