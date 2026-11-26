"""Metric calculations shared by validation and tests."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compare_regression_metrics(y_true, baseline_predictions, candidate_predictions):
    """Compare two regression models and report RMSE, R², and RMSE improvement."""
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_predictions)))
    candidate_rmse = float(np.sqrt(mean_squared_error(y_true, candidate_predictions)))
    return {
        "baseline_rmse": baseline_rmse,
        "baseline_r2": float(r2_score(y_true, baseline_predictions)),
        "candidate_rmse": candidate_rmse,
        "candidate_r2": float(r2_score(y_true, candidate_predictions)),
        "improvement": float((baseline_rmse - candidate_rmse) / baseline_rmse * 100),
    }
