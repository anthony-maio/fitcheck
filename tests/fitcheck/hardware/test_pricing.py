"""Tests for cloud GPU pricing module."""

from fitcheck.hardware.pricing import get_cloud_prices


class TestCloudPrices:
    """get_cloud_prices returns valid pricing data."""

    def test_all_prices_positive(self):
        prices = get_cloud_prices()
        for name, rate in prices.items():
            assert rate > 0, f"{name} has non-positive rate: {rate}"

    def test_contains_common_cloud_gpus(self):
        prices = get_cloud_prices()
        assert "A100 80GB" in prices
        assert "H100" in prices

    def test_consumer_gpus_not_in_cloud_prices(self):
        """3090 and 4090 are consumer cards, not cloud offerings."""
        prices = get_cloud_prices()
        for name in prices:
            assert "3090" not in name
            assert "4090" not in name

    def test_returns_multiple_gpus(self):
        """Should return at least 3 cloud GPU options."""
        prices = get_cloud_prices()
        assert len(prices) >= 3
