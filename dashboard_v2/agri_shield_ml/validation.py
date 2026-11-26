"""Chronological validation split utilities for grouped forecasting data."""

import numpy as np
import pandas as pd


def grouped_walk_forward_splits(
    data,
    time_col,
    group_col,
    n_splits=4,
    test_size=6,
    gap=0,
    max_train_size=None,
):
    """Return expanding or rolling-origin row-index splits grouped by timestamp.

    Every group at a timestamp is assigned to the same fold. The returned
    integer indices refer to row positions in ``data`` and are ordered
    chronologically (then by group). ``gap`` and the window sizes count
    distinct timestamps, not rows.
    """
    if time_col not in data or group_col not in data:
        missing = {time_col, group_col}.difference(data.columns)
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if n_splits < 1 or test_size < 1 or gap < 0:
        raise ValueError("n_splits and test_size must be positive; gap cannot be negative")
    if max_train_size is not None and max_train_size < 1:
        raise ValueError("max_train_size must be positive when provided")
    if data[[time_col, group_col]].isna().any().any():
        raise ValueError("Time and group columns cannot contain missing values")

    times = pd.to_datetime(data[time_col], errors="coerce")
    if times.isna().any():
        raise ValueError(f"{time_col} must contain valid timestamps")
    unique_times = pd.Index(times.unique()).sort_values()
    required_times = n_splits * test_size
    first_test_start = len(unique_times) - required_times
    if first_test_start - gap < 1:
        raise ValueError(
            "Not enough distinct timestamps for the requested folds, gap, and training data"
        )

    group_values = data[group_col].astype(str).to_numpy()
    time_values = times.to_numpy(dtype="datetime64[ns]")
    ordered_positions = np.lexsort((group_values, time_values))
    splits = []

    for fold in range(n_splits):
        test_start = first_test_start + fold * test_size
        train_end = test_start - gap
        train_start = 0 if max_train_size is None else max(0, train_end - max_train_size)
        train_times = unique_times[train_start:train_end]
        test_times = unique_times[test_start:test_start + test_size]
        train_mask = np.isin(time_values, train_times.to_numpy(dtype="datetime64[ns]"))
        test_mask = np.isin(time_values, test_times.to_numpy(dtype="datetime64[ns]"))
        train_indices = ordered_positions[train_mask[ordered_positions]]
        test_indices = ordered_positions[test_mask[ordered_positions]]
        splits.append((train_indices, test_indices))

    return splits


def rolling_origin_splits(
    data, time_col, group_col, n_splits=4, test_size=6, gap=0, train_size=None
):
    """Return grouped rolling-origin folds with a fixed maximum train window."""
    if train_size is None:
        raise ValueError("train_size is required for rolling-origin splits")
    return grouped_walk_forward_splits(
        data,
        time_col,
        group_col,
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        max_train_size=train_size,
    )
