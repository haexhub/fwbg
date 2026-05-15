"""Tests for the consolidated compute_fold_progress helper."""
import pytest

from fwbg.utils.run_progress import compute_fold_progress


def test_fold_zero_means_not_started():
    assert compute_fold_progress(fold=0, total_folds=5) == 0.0


def test_first_fold_no_grid_progress():
    # fold=1 with no grid sweep progress: first fold is in progress, none done
    assert compute_fold_progress(fold=1, total_folds=5) == 0.0


def test_first_fold_halfway_through_grid():
    # Half-way through fold 1 of 5 → 0/5 + 0.5/5 = 0.1
    assert compute_fold_progress(
        fold=1, total_folds=5, grid_pos=5, grid_total=10,
    ) == pytest.approx(0.1)


def test_middle_fold_full_grid():
    # Fold 3 of 5, grid fully done within current fold → 2/5 + 1/5 = 0.6
    assert compute_fold_progress(
        fold=3, total_folds=5, grid_pos=10, grid_total=10,
    ) == pytest.approx(0.6)


def test_last_fold_completed():
    # Fold 5 of 5, grid fully done → 4/5 + 1/5 = 1.0
    assert compute_fold_progress(
        fold=5, total_folds=5, grid_pos=10, grid_total=10,
    ) == pytest.approx(1.0)


def test_only_grid_progress_no_folds():
    assert compute_fold_progress(
        fold=0, total_folds=0, grid_pos=3, grid_total=10,
    ) == pytest.approx(0.3)


def test_no_progress_returns_zero():
    assert compute_fold_progress(fold=0, total_folds=0) == 0.0


def test_monotonically_increasing_across_folds():
    last = -1.0
    for fold in range(1, 5):
        for gp in range(0, 11):
            value = compute_fold_progress(fold, 4, gp, 10)
            assert value >= last, f"regressed at fold={fold} gp={gp}"
            last = value
