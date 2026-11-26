"""Population drift monitoring and calibrated prediction intervals."""

import math

import numpy as np

DRIFT_PSI_THRESHOLD = 0.2
DRIFT_Z_THRESHOLD = 3.0
INTERVAL_COVERAGE = 0.9


def _population(values, bins=10):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Monitoring values must be a non-empty finite population")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([values[0] - 0.5, values[0] + 0.5])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    counts, _ = np.histogram(values, bins=edges)
    proportions = counts.astype(float) / len(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "edges": [float(value) for value in edges[1:-1]],
        "proportions": proportions.tolist(),
        "count": len(values),
    }


def build_monitoring_baseline(features, feature_names, predictions=None,
                              price_errors=None, risk_errors=None,
                              coverage=INTERVAL_COVERAGE):
    """Build persisted feature/prediction populations and conformal error widths."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Feature matrix does not match the ordered feature names")
    feature_population = {
        name: _population(features[:, index])
        for index, name in enumerate(feature_names)
    }
    prediction_population = {}
    for name, values in (predictions or {}).items():
        prediction_population[name] = _population(values)

    intervals = {"coverage": float(coverage), "method": "validation_absolute_error"}
    for name, errors in (("price", price_errors), ("risk", risk_errors)):
        if errors is None or len(errors) == 0:
            intervals[name] = None
            continue
        errors = np.abs(np.asarray(errors, dtype=float).reshape(-1))
        if not np.isfinite(errors).all():
            raise ValueError("Calibration errors must be finite")
        rank = min(len(errors), max(1, math.ceil((len(errors) + 1) * coverage)))
        intervals[name] = float(np.sort(errors)[rank - 1])
    return {
        "feature_population": feature_population,
        "prediction_population": prediction_population,
        "prediction_intervals": intervals,
    }


def monitor_population(values, baseline, psi_threshold=DRIFT_PSI_THRESHOLD,
                       z_threshold=DRIFT_Z_THRESHOLD):
    """Compare observed populations to stored means/stds and PSI reference bins."""
    observed = np.asarray(values, dtype=float).reshape(-1)
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("Observed monitoring values must be finite and non-empty")
    baseline_mean = float(baseline["mean"])
    baseline_std = float(baseline["std"])
    observed_mean = float(np.mean(observed))
    z_score = (
        abs(observed_mean - baseline_mean) / baseline_std
        if baseline_std > 0
        else (0.0 if observed_mean == baseline_mean else 1_000_000.0)
    )
    edges = np.asarray(baseline.get("edges", []), dtype=float)
    expected = np.asarray(baseline.get("proportions", []), dtype=float)
    psi = None
    if len(expected) and len(expected) == len(edges) + 1:
        observed_counts, _ = np.histogram(observed, bins=np.concatenate(([-np.inf], edges, [np.inf])))
        actual = observed_counts.astype(float) / len(observed)
        epsilon = 1e-6
        expected = np.clip(expected, epsilon, None)
        actual = np.clip(actual, epsilon, None)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    drifted = z_score >= z_threshold or (psi is not None and psi >= psi_threshold)
    return {
        "count": len(observed),
        "mean": observed_mean,
        "std": float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
        "mean_z_score": float(z_score),
        "psi": psi,
        "drifted": bool(drifted),
        "thresholds": {"mean_z_score": z_threshold, "psi": psi_threshold},
    }


def monitor_features(matrix, feature_names, baseline):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("Feature matrix does not match the ordered feature names")
    populations = baseline.get("feature_population", {})
    return {
        name: monitor_population(matrix[:, index], populations[name])
        for index, name in enumerate(feature_names)
        if name in populations
    }


def prediction_interval(prediction, half_width, coverage=INTERVAL_COVERAGE,
                        bounds=None, calibration="validation"):
    """Return a symmetric marginal interval, optionally bounded to target domain."""
    if half_width is None:
        return None
    lower = float(prediction) - float(half_width)
    upper = float(prediction) + float(half_width)
    if bounds:
        lower = max(float(bounds[0]), lower)
        upper = min(float(bounds[1]), upper)
    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(coverage),
        "calibration": calibration,
    }
