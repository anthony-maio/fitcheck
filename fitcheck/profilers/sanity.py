"""Training sanity checks for fitcheck.

Catches common training mistakes that are not VRAM-related:
overfit risk, ineffective batch sizes, and excessive epoch counts.
These checks complement the solver's VRAM analysis with training
quality guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fitcheck.models.profiles import DatasetProfile
from fitcheck.models.results import TrainingConfig


@dataclass
class SanityWarning:
    """A training sanity check result."""

    severity: Literal["info", "warn", "critical"]
    category: str
    message: str


def check_training_sanity(
    dataset: DatasetProfile | None,
    config: TrainingConfig,
    trainable_params: int,
    num_epochs: int = 3,
) -> list[SanityWarning]:
    """Run all sanity checks and return warnings.

    Args:
        dataset: Dataset profile (None if no dataset provided).
        config: The recommended training configuration.
        trainable_params: Number of trainable parameters.
        num_epochs: Planned number of training epochs.

    Returns:
        List of warnings, may be empty.
    """
    if dataset is None or dataset.num_rows == 0:
        return []

    warnings: list[SanityWarning] = []
    warnings.extend(_check_overfit_risk(dataset, trainable_params))
    warnings.extend(_check_effective_batch_vs_dataset(dataset, config))
    warnings.extend(_check_epoch_overtraining(dataset, num_epochs))
    return warnings


def _check_overfit_risk(
    dataset: DatasetProfile,
    trainable_params: int,
) -> list[SanityWarning]:
    """Flag when dataset is too small relative to trainable parameters.

    Rough heuristic: fewer than 10 examples per 1M trainable params
    is a strong overfit risk.  Fewer than 100 per 1M is worth flagging.
    """
    warnings = []
    if trainable_params == 0:
        return warnings

    trainable_m = trainable_params / 1e6
    rows_per_m = dataset.num_rows / trainable_m

    if rows_per_m < 10:
        warnings.append(SanityWarning(
            severity="critical",
            category="overfit",
            message=(
                f"Very few training examples ({dataset.num_rows:,}) relative to "
                f"trainable params ({trainable_m:.1f}M). "
                f"High risk of memorization rather than learning."
            ),
        ))
    elif rows_per_m < 100:
        warnings.append(SanityWarning(
            severity="warn",
            category="overfit",
            message=(
                f"Small dataset ({dataset.num_rows:,} rows) relative to "
                f"trainable params ({trainable_m:.1f}M). "
                f"Consider reducing epochs or using early stopping."
            ),
        ))
    return warnings


def _check_effective_batch_vs_dataset(
    dataset: DatasetProfile,
    config: TrainingConfig,
) -> list[SanityWarning]:
    """Flag when effective batch size covers a large fraction of the dataset."""
    warnings = []
    ebs = config.effective_batch_size
    if ebs > dataset.num_rows // 2:
        warnings.append(SanityWarning(
            severity="warn",
            category="batch_size",
            message=(
                f"Effective batch size ({ebs}) covers >{ebs * 100 // dataset.num_rows}% "
                f"of the dataset ({dataset.num_rows:,} rows). "
                f"Each gradient step sees most of the data -- "
                f"reduce batch size or increase dataset."
            ),
        ))
    return warnings


def _check_epoch_overtraining(
    dataset: DatasetProfile,
    num_epochs: int,
) -> list[SanityWarning]:
    """Flag when many epochs over a small dataset risks overtraining."""
    warnings = []
    if dataset.num_rows < 1000 and num_epochs > 5:
        warnings.append(SanityWarning(
            severity="warn",
            category="epochs",
            message=(
                f"{num_epochs} epochs over {dataset.num_rows:,} rows "
                f"means each example seen {num_epochs} times. "
                f"Consider fewer epochs or data augmentation."
            ),
        ))
    return warnings
