from __future__ import annotations

import pytest

from visionguard.benchmark import (
    BenchmarkRunError,
    benchmark_cells,
    execute_pending_cells,
    pending_cells,
)
from visionguard.benchmark_metrics import (
    BinaryCountAccumulator,
    Float16AuProAccumulator,
)
from visionguard.metrics import au_pro, binary_f1
from visionguard.protocol import OFFICIAL_CATEGORIES, PROTOCOL_SEEDS


def test_frozen_benchmark_matrix_has_every_category_seed_once() -> None:
    cells = benchmark_cells()

    assert len(cells) == 24
    assert len(set(cells)) == 24
    assert cells == tuple(
        (category, seed) for category in OFFICIAL_CATEGORIES for seed in PROTOCOL_SEEDS
    )


def test_resume_skips_only_completed_cells_in_frozen_order() -> None:
    completed = {benchmark_cells()[0], benchmark_cells()[5]}

    result = pending_cells(completed)

    assert result == tuple(cell for cell in benchmark_cells() if cell not in completed)


def test_resume_rejects_unknown_cell() -> None:
    with pytest.raises(BenchmarkRunError, match="unknown cells"):
        pending_cells({("not-a-category", 42)})


def test_execution_stops_at_failure_for_durable_resume() -> None:
    visited: list[tuple[str, int]] = []

    def execute(category: str, seed: int) -> None:
        visited.append((category, seed))
        if len(visited) == 3:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute_pending_cells((), execute)

    assert visited == list(benchmark_cells()[:3])


def test_streaming_binary_f1_matches_reference() -> None:
    np = pytest.importorskip("numpy")
    labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    predictions = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    accumulator = BinaryCountAccumulator()

    accumulator.update(labels, predictions)

    assert accumulator.result() == binary_f1(
        labels.reshape(-1).tolist(), predictions.reshape(-1).tolist(), level="pixel"
    )


def test_float16_histogram_au_pro_matches_reference() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    labels = [
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [0, 1]], dtype=np.uint8),
    ]
    scores = [
        np.array([[0.9, 0.2], [0.1, 0.0]], dtype=np.float16),
        np.array([[0.3, 0.2], [0.1, 0.8]], dtype=np.float16),
    ]
    accumulator = Float16AuProAccumulator()
    for label, score in zip(labels, scores, strict=True):
        accumulator.update(label, score)

    measured = accumulator.result(fpr_limit=0.05)
    reference = au_pro(
        [label.tolist() for label in labels],
        [score.astype(float).tolist() for score in scores],
        fpr_limit=0.05,
    )

    assert measured.status == reference.status == "defined"
    assert measured.value == pytest.approx(reference.value, abs=1e-12)


def test_float16_histogram_groups_exact_ties() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    accumulator = Float16AuProAccumulator()
    accumulator.update(
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
        np.full((2, 2), 0.5, dtype=np.float16),
    )

    result = accumulator.result(fpr_limit=0.05)

    assert result.status == "defined"
    assert result.value == pytest.approx(0.025)
