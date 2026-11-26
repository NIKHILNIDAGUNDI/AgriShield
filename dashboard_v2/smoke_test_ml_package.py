"""Small offline check for feature construction and artifact round-tripping."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from agri_shield_ml import (
    FoodSecurityNet,
    build_feature_names,
    load_model_artifact,
    prepare_forecasting_lags,
    save_model_artifact,
)
from sklearn.preprocessing import StandardScaler


def main():
    rows = []
    for month in range(1, 7):
        row = {
            "region": "test",
            "year_month": f"2025-{month:02d}",
            "year": 2025,
            "month": month,
        }
        for index, source in enumerate((
            "temperature", "precipitation", "crop_yield", "fuel_price",
            "transport_cost", "inflation_rate", "geopolitical_risk",
            "news_sentiment", "food_price_index", "shortage_risk_score",
        )):
            row[source] = month + index
        rows.append(row)

    lagged = prepare_forecasting_lags(pd.DataFrame(rows))
    features = build_feature_names()
    assert lagged.shape[0] == 3
    assert lagged.loc[0, "temperature_lag_1"] == 3
    assert lagged.loc[0, "temperature_lag_3"] == 1
    assert len(features) == 30

    feature_scaler = StandardScaler().fit(lagged.loc[:, features].values)
    price_scaler = StandardScaler().fit(np.array([[10.0], [20.0], [30.0]]))
    risk_scaler = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
    model = FoodSecurityNet(len(features))
    output_dir = Path(__file__).parent / ".ml_smoke_artifact"
    weights = output_dir / "model.pth"
    manifest_path = output_dir / "model.manifest.json"
    try:
        save_model_artifact(
            model, weights, manifest_path, features,
            feature_scaler, price_scaler, risk_scaler,
            {"smoke": 1.0},
        )
        restored = load_model_artifact(manifest_path)
        assert restored["manifest"]["feature_names"] == features
        assert np.allclose(
            restored["feature_scaler"].transform(lagged.loc[:, features].values),
            feature_scaler.transform(lagged.loc[:, features].values),
        )
        test_inputs = feature_scaler.transform(lagged.loc[:, features].values)
        model.eval()
        tensor = torch.as_tensor(test_inputs, dtype=torch.float32)
        with torch.no_grad():
            expected = model(tensor)
            actual = restored["model"](tensor)
        assert all(np.allclose(left.numpy(), right.numpy()) for left, right in zip(expected, actual))
        print("AgriShield ML package smoke test passed.")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
