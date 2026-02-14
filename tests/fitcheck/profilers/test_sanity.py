"""Tests for training sanity checker."""

from fitcheck.models.profiles import DatasetProfile
from fitcheck.models.results import TrainingConfig
from fitcheck.profilers.sanity import check_training_sanity


def _make_config(effective_batch: int = 16) -> TrainingConfig:
    return TrainingConfig(
        micro_batch_size=4,
        gradient_accumulation_steps=effective_batch // 4,
        effective_batch_size=effective_batch,
        seq_len=512,
    )


def _make_dataset(num_rows: int) -> DatasetProfile:
    return DatasetProfile(
        source="test.jsonl",
        num_rows=num_rows,
        detected_format="alpaca",
    )


class TestOverfitRisk:
    """Detects when dataset is too small for trainable param count."""

    def test_tiny_dataset_triggers_critical(self):
        """50 rows with 40M trainable params: strong overfit risk."""
        warnings = check_training_sanity(
            dataset=_make_dataset(50),
            config=_make_config(),
            trainable_params=40_000_000,
        )
        overfit = [w for w in warnings if w.category == "overfit"]
        assert len(overfit) == 1
        assert overfit[0].severity == "critical"

    def test_small_dataset_triggers_warning(self):
        """500 rows with 40M trainable params: moderate overfit risk."""
        warnings = check_training_sanity(
            dataset=_make_dataset(500),
            config=_make_config(),
            trainable_params=40_000_000,
        )
        overfit = [w for w in warnings if w.category == "overfit"]
        assert len(overfit) == 1
        assert overfit[0].severity == "warn"

    def test_large_dataset_no_overfit_warning(self):
        """10,000 rows with 40M trainable params: no overfit concern."""
        warnings = check_training_sanity(
            dataset=_make_dataset(10_000),
            config=_make_config(),
            trainable_params=40_000_000,
        )
        overfit = [w for w in warnings if w.category == "overfit"]
        assert len(overfit) == 0


class TestEffectiveBatchSize:
    """Flags when batch covers too much of the dataset per step."""

    def test_huge_batch_warns(self):
        """Effective batch 100 on 150 rows: >50% per step."""
        warnings = check_training_sanity(
            dataset=_make_dataset(150),
            config=_make_config(effective_batch=100),
            trainable_params=1_000_000,
        )
        batch_warnings = [w for w in warnings if w.category == "batch_size"]
        assert len(batch_warnings) == 1

    def test_reasonable_batch_no_warning(self):
        """Effective batch 16 on 10,000 rows: fine."""
        warnings = check_training_sanity(
            dataset=_make_dataset(10_000),
            config=_make_config(effective_batch=16),
            trainable_params=1_000_000,
        )
        batch_warnings = [w for w in warnings if w.category == "batch_size"]
        assert len(batch_warnings) == 0


class TestEpochOvertraining:
    """Flags excessive epochs on small datasets."""

    def test_many_epochs_small_data_warns(self):
        """10 epochs on 500 rows should warn."""
        warnings = check_training_sanity(
            dataset=_make_dataset(500),
            config=_make_config(),
            trainable_params=1_000_000,
            num_epochs=10,
        )
        epoch_warnings = [w for w in warnings if w.category == "epochs"]
        assert len(epoch_warnings) == 1

    def test_few_epochs_no_warning(self):
        """3 epochs on any dataset should not trigger epoch warning."""
        warnings = check_training_sanity(
            dataset=_make_dataset(500),
            config=_make_config(),
            trainable_params=1_000_000,
            num_epochs=3,
        )
        epoch_warnings = [w for w in warnings if w.category == "epochs"]
        assert len(epoch_warnings) == 0


class TestEdgeCases:
    """Boundary conditions."""

    def test_no_dataset_returns_empty(self):
        warnings = check_training_sanity(
            dataset=None,
            config=_make_config(),
            trainable_params=40_000_000,
        )
        assert warnings == []

    def test_zero_rows_returns_empty(self):
        warnings = check_training_sanity(
            dataset=_make_dataset(0),
            config=_make_config(),
            trainable_params=40_000_000,
        )
        assert warnings == []
