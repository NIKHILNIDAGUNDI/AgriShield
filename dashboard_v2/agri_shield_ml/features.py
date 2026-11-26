"""Shared, deterministic forecasting feature construction."""

FEATURE_SOURCES = (
    "temperature",
    "precipitation",
    "crop_yield",
    "fuel_price",
    "transport_cost",
    "inflation_rate",
    "geopolitical_risk",
    "news_sentiment",
    "food_price_index",
    "shortage_risk_score",
)


def build_feature_names(lag_months=3):
    """Return feature names in the same source-major order used by training."""
    lags = dict.fromkeys((1, 2, lag_months))
    return [f"{source}_lag_{lag}" for source in FEATURE_SOURCES for lag in lags]


def prepare_forecasting_lags(df, lag_months=3):
    """Sort by region/month, add historical features, and drop incomplete rows."""
    if lag_months < 1:
        raise ValueError("lag_months must be at least 1")
    missing = {"region", "year_month", *FEATURE_SOURCES}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for lag construction: {sorted(missing)}")

    df_sorted = df.sort_values(by=["region", "year_month"]).copy()
    lags = dict.fromkeys((1, 2, lag_months))
    for column in FEATURE_SOURCES:
        for lag in lags:
            df_sorted[f"{column}_lag_{lag}"] = df_sorted.groupby("region")[column].shift(lag)
    return df_sorted.dropna().reset_index(drop=True)


def align_features(df, feature_names):
    """Select features in manifest order and reject missing/duplicated names."""
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be unique")
    missing = set(feature_names).difference(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    return df.loc[:, feature_names]
