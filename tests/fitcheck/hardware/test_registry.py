"""Tests for the GPU hardware registry."""

import pytest
from fitcheck.hardware.registry import get_hardware, list_hardware


class TestHardwareRegistry:
    """Hardware lookup returns correct specs and handles aliases."""

    def test_3090_specs(self):
        hw = get_hardware("3090")
        assert hw.name == "NVIDIA RTX 3090"
        assert hw.total_vram_gb == 24.0
        assert hw.overhead_gb == 1.2
        assert hw.usable_vram_gb == pytest.approx(22.8)

    def test_h100_specs(self):
        hw = get_hardware("h100")
        assert hw.total_vram_gb == 80.0
        assert hw.bf16_tflops > 200  # H100 is much faster than consumer cards

    def test_a100_defaults_to_80gb(self):
        """Bare 'a100' should resolve to 80GB variant."""
        hw = get_hardware("a100")
        assert hw.total_vram_gb == 80.0

    def test_a100_40gb_explicit(self):
        hw = get_hardware("a100-40gb")
        assert hw.total_vram_gb == 40.0

    def test_alias_resolution(self):
        """Various human-typed GPU names should resolve correctly."""
        assert get_hardware("rtx3090").name == "NVIDIA RTX 3090"
        assert get_hardware("RTX 3090").name == "NVIDIA RTX 3090"
        assert get_hardware("h100-sxm").name == "NVIDIA H100 80GB"

    def test_unknown_gpu_raises_with_suggestions(self):
        with pytest.raises(KeyError, match="Unknown GPU"):
            get_hardware("rtx5090")

    def test_usable_vram_always_less_than_total(self):
        """Every GPU in the registry must have overhead > 0."""
        for hw in list_hardware():
            assert hw.overhead_gb > 0, f"{hw.name} has zero overhead"
            assert hw.usable_vram_gb < hw.total_vram_gb

    def test_all_gpus_have_positive_compute(self):
        for hw in list_hardware():
            assert hw.fp16_tflops > 0, f"{hw.name} missing fp16 TFLOPS"
            assert hw.memory_bandwidth_gbps > 0, f"{hw.name} missing bandwidth"
