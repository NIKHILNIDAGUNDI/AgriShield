"""Versioned shared ML components for the AgriShield dashboard."""

__version__ = "1.0.0"

from .artifacts import (
    ARTIFACT_SCHEMA,
    ARTIFACT_VERSION,
    load_model_artifact,
    save_model_artifact,
)
from .features import (
    FEATURE_SOURCES,
    align_features,
    build_feature_names,
    prepare_forecasting_lags,
)
from .metrics import compare_regression_metrics
from .model import FoodSecurityDataset, FoodSecurityNet
from .monitoring import (
    build_monitoring_baseline,
    monitor_features,
    monitor_population,
    prediction_interval,
)
from .training import fit_network, predict_network
from .validation import grouped_walk_forward_splits, rolling_origin_splits

__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_VERSION",
    "FEATURE_SOURCES",
    "FoodSecurityDataset",
    "FoodSecurityNet",
    "align_features",
    "build_feature_names",
    "build_monitoring_baseline",
    "compare_regression_metrics",
    "fit_network",
    "grouped_walk_forward_splits",
    "load_model_artifact",
    "monitor_features",
    "monitor_population",
    "predict_network",
    "prediction_interval",
    "prepare_forecasting_lags",
    "rolling_origin_splits",
    "save_model_artifact",
]
