"""Enforce the MVTec AD 2 scientific data-use boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DataBoundaryError(ValueError):
    """Raised when an experiment requests a prohibited dataset boundary."""


class SplitRole(StrEnum):
    """VisionGuard-owned semantic roles for official dataset splits."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    PRELIMINARY_EVALUATION = "preliminary_evaluation"
    FINAL_EVALUATION = "final_evaluation"


@dataclass(frozen=True)
class DataBoundaryPolicy:
    """Frozen Phase 2A split policy.

    Private data is deliberately absent from runnable roles. Final private evaluation
    must later use a separately reviewed submission path that cannot expose labels.
    """

    train_split: str = "train"
    calibration_split: str = "validation"
    public_test_split: str = "test_public"
    private_splits: frozenset[str] = frozenset({"test_private", "test_private_mixed"})

    def authorize(
        self,
        split: str,
        role: SplitRole,
        *,
        configuration_frozen: bool = False,
    ) -> None:
        """Reject a split/role combination that could cause leakage or tuning."""

        if split in self.private_splits:
            raise DataBoundaryError(
                "Private MVTec AD 2 splits are unavailable to local experiment roles"
            )
        expected = {
            SplitRole.TRAIN: self.train_split,
            SplitRole.CALIBRATION: self.calibration_split,
            SplitRole.PRELIMINARY_EVALUATION: self.public_test_split,
        }
        if role is SplitRole.FINAL_EVALUATION:
            raise DataBoundaryError(
                "Final evaluation requires a separately reviewed private-server path"
            )
        if split != expected[role]:
            raise DataBoundaryError(
                f"Split {split!r} cannot be used for role {role.value!r}; "
                f"expected {expected[role]!r}"
            )
        if role is SplitRole.PRELIMINARY_EVALUATION and not configuration_frozen:
            raise DataBoundaryError(
                "test_public is allowed only after configuration and calibration freeze"
            )
