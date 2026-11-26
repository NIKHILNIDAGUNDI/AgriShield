# PyTorch Deep Learning Training Pipeline
# Optimized for CUDA execution (e.g. RTX 5060 Graphics Card) with automatic CPU fallback
# Implements a Dual-Stage Training workflow:
# 1. Validation Stage: grouped, chronological walk-forward validation.
# 2. Production Stage: Train on all data till date to forecast the next 3 months out-of-sample.

import argparse
import json
import os
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from agri_shield_ml import (
    FoodSecurityDataset,
    FoodSecurityNet,
    align_features,
    build_feature_names,
    build_monitoring_baseline,
    compare_regression_metrics,
    fit_network,
    grouped_walk_forward_splits,
    predict_network,
    prediction_interval,
    prepare_forecasting_lags,
    save_model_artifact,
)
from agri_shield_ml.observability import get_json_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

def train_and_evaluate(csv_path=None, dashboard_dir=None, epochs=300, validation_only=False,
                       model_version=None):
    logger = get_json_logger("agrishield.training")
    # Load dataset
    if csv_path is None:
        csv_path = "food_security_data.csv"
    if not os.path.exists(csv_path):
        csv_path = "../food_security_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("food_security_data.csv not found.")
            
    df = pd.read_csv(csv_path)
    logger.info(
        "Training dataset loaded",
        extra={"event": "training_started", "rows": len(df), "epochs": int(epochs)},
    )
    
    # Process lags
    df_ml = prepare_forecasting_lags(df, lag_months=3)
    
    # Shift targets forward by 3 months (so lag features at t predict t+3)
    df_ml["future_food_price_index"] = df_ml.groupby("region")["food_price_index"].shift(-3)
    df_ml["future_shortage_risk_score"] = df_ml.groupby("region")["shortage_risk_score"].shift(-3)
    
    # Features are all lag columns
    features = build_feature_names(lag_months=3)
    align_features(df_ml, features)
    
    # Keep rows with observed future targets for cross-validation and production.
    df_with_targets = df_ml.dropna(subset=["future_food_price_index", "future_shortage_risk_score"]).reset_index(drop=True)
    # ====================================================================
    # STAGE 1: GROUPED WALK-FORWARD VALIDATION
    # ====================================================================
    print("--- STAGE 1: Grouped Walk-Forward Validation ---")
    n_splits = 4
    test_size_months = 6
    forecast_gap_months = 3
    folds = grouped_walk_forward_splits(
        df_with_targets,
        time_col="year_month",
        group_col="region",
        n_splits=n_splits,
        test_size=test_size_months,
        gap=forecast_gap_months,
    )
    validation_folds = []
    fold_predictions = []
    fold_epoch_logs = []
    metric_names = (
        "rf_price_rmse", "rf_price_r2", "torch_price_rmse", "torch_price_r2",
        "rf_risk_rmse", "rf_risk_r2", "torch_risk_rmse", "torch_risk_r2",
        "price_improvement", "risk_improvement",
    )

    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        train_df = df_with_targets.iloc[train_indices]
        test_df = df_with_targets.iloc[test_indices]
        X_train_raw = align_features(train_df, features).values
        X_test_raw = align_features(test_df, features).values
        y_price_train = train_df["future_food_price_index"].values
        y_risk_train = train_df["future_shortage_risk_score"].values
        y_price_test = test_df["future_food_price_index"].values
        y_risk_test = test_df["future_shortage_risk_score"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        scaler_price = StandardScaler()
        scaler_risk = StandardScaler()
        y_price_train_scaled = scaler_price.fit_transform(y_price_train.reshape(-1, 1))
        y_risk_train_scaled = scaler_risk.fit_transform(y_risk_train.reshape(-1, 1))
        y_price_test_scaled = scaler_price.transform(y_price_test.reshape(-1, 1))
        y_risk_test_scaled = scaler_risk.transform(y_risk_test.reshape(-1, 1))

        rf_price = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_price.fit(X_train_raw, y_price_train)
        rf_risk = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_risk.fit(X_train_raw, y_risk_train)
        rf_price_preds = rf_price.predict(X_test_raw)
        rf_risk_preds = rf_risk.predict(X_test_raw)

        train_dataset = FoodSecurityDataset(X_train, y_price_train_scaled, y_risk_train_scaled)
        test_dataset = FoodSecurityDataset(X_test, y_price_test_scaled, y_risk_test_scaled)
        batch_size = 15 if len(train_dataset) % 16 == 1 else 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        torch.manual_seed(42 + fold_number)
        np.random.seed(42 + fold_number)
        model_val = FoodSecurityNet(X_train.shape[1]).to(device)
        fold_logs = fit_network(
            model_val, train_loader, epochs, device, learning_rate=0.008,
            weight_decay=0.001, validation_loader=test_loader,
        )
        fold_epoch_logs.append(fold_logs)

        torch_price_scaled, torch_risk_scaled = predict_network(model_val, X_test, device)
        torch_price_preds = scaler_price.inverse_transform(torch_price_scaled.reshape(-1, 1)).flatten()
        torch_risk_preds = scaler_risk.inverse_transform(torch_risk_scaled.reshape(-1, 1)).flatten()
        torch_risk_preds = np.clip(torch_risk_preds, 0.0, 100.0)

        price_metrics = compare_regression_metrics(
            y_price_test, rf_price_preds, torch_price_preds
        )
        risk_metrics = compare_regression_metrics(
            y_risk_test, rf_risk_preds, torch_risk_preds
        )
        metrics = {
            "rf_price_rmse": price_metrics["baseline_rmse"],
            "rf_price_r2": price_metrics["baseline_r2"],
            "torch_price_rmse": price_metrics["candidate_rmse"],
            "torch_price_r2": price_metrics["candidate_r2"],
            "rf_risk_rmse": risk_metrics["baseline_rmse"],
            "rf_risk_r2": risk_metrics["baseline_r2"],
            "torch_risk_rmse": risk_metrics["candidate_rmse"],
            "torch_risk_r2": risk_metrics["candidate_r2"],
            "price_improvement": price_metrics["improvement"],
            "risk_improvement": risk_metrics["improvement"],
        }
        train_months = pd.to_datetime(train_df["year_month"])
        validation_months = pd.to_datetime(test_df["year_month"])
        validation_folds.append({
            "fold": fold_number,
            "train_start": train_months.min().strftime("%Y-%m"),
            "train_end": train_months.max().strftime("%Y-%m"),
            "validation_start": validation_months.min().strftime("%Y-%m"),
            "validation_end": validation_months.max().strftime("%Y-%m"),
            "train_rows": len(train_df),
            "validation_rows": len(test_df),
            "gap_months": forecast_gap_months,
            "metrics": metrics,
        })
        logger.info(
            "Walk-forward validation fold completed",
            extra={"event": "validation_fold_completed", "fold": fold_number, **metrics},
        )

        ordered_test_df = test_df.reset_index(drop=True)
        for i, row in ordered_test_df.iterrows():
            target_month = int(row["month"]) + 3
            target_year = int(row["year"])
            if target_month > 12:
                target_month -= 12
                target_year += 1
            fold_predictions.append({
                "region": row["region"],
                "target_month": f"{target_year}-{target_month:02d}",
                "feature_month": row["year_month"],
                "actual_price": float(row["future_food_price_index"]),
                "rf_predicted_price": float(rf_price_preds[i]),
                "torch_predicted_price": float(torch_price_preds[i]),
                "actual_risk": float(row["future_shortage_risk_score"]),
                "rf_predicted_risk": float(rf_risk_preds[i]),
                "torch_predicted_risk": float(torch_risk_preds[i]),
            })

    mean_metrics = {
        name: float(np.mean([fold["metrics"][name] for fold in validation_folds]))
        for name in metric_names
    }
    epoch_logs = []
    for epoch_index in range(epochs):
        epoch_values = [
            fold_logs[epoch_index] for fold_logs in fold_epoch_logs
            if len(fold_logs) > epoch_index
        ]
        if epoch_values:
            epoch_logs.append({
                "epoch": epoch_index + 1,
                "train_loss": float(np.mean([value["train_loss"] for value in epoch_values])),
                "val_loss": float(np.mean([value["val_loss"] for value in epoch_values])),
            })
    validation = {
        "strategy": "grouped_walk_forward",
        "n_splits": n_splits,
        "test_size_months": test_size_months,
        "gap_months": forecast_gap_months,
        "group_column": "region",
        "time_column": "year_month",
        "aggregation": "unweighted mean of fold metrics",
        "folds": validation_folds,
        "mean_metrics": mean_metrics,
    }
    logger.info(
        "Walk-forward validation completed",
        extra={
            "event": "validation_completed",
            "folds": len(validation_folds),
            "validation_rows": len(fold_predictions),
        },
    )
    for name in metric_names:
        print(f"Validation {name}: {mean_metrics[name]:.3f} (mean across {n_splits} folds)")
    if validation_only:
        logger.info("Validation-only run completed", extra={"event": "validation_only_completed"})
        return {"validation": validation, "metrics": mean_metrics, "epoch_logs": epoch_logs}
    if model_version is None:
        model_version = datetime.now(UTC).strftime("candidate-%Y%m%d%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9._+-]{1,80}", model_version):
        raise ValueError("model_version must contain only letters, numbers, dot, underscore, plus, or hyphen")
    predictions = fold_predictions
    
    # ====================================================================
    # STAGE 2: PRODUCTION TRAINING (Train on 100% of data to predict out-of-sample)
    # ====================================================================
    print("\n--- STAGE 2: Training Production Model (Predicting 2026-01 to 2026-03) ---")
    
    X_prod_raw = align_features(df_with_targets, features).values
    y_price_prod = df_with_targets["future_food_price_index"].values
    y_risk_prod = df_with_targets["future_shortage_risk_score"].values
    
    # Scaler for production model
    prod_scaler = StandardScaler()
    X_prod = prod_scaler.fit_transform(X_prod_raw)
    
    # RF Production training
    rf_price_prod = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_price_prod.fit(X_prod_raw, y_price_prod)
    rf_risk_prod = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_risk_prod.fit(X_prod_raw, y_risk_prod)
    
    # Target scalers for production model
    prod_scaler_price = StandardScaler()
    prod_scaler_risk = StandardScaler()
    y_price_prod_scaled = prod_scaler_price.fit_transform(y_price_prod.reshape(-1, 1))
    y_risk_prod_scaled = prod_scaler_risk.fit_transform(y_risk_prod.reshape(-1, 1))
    
    # Stage 2 PyTorch Loader
    prod_dataset = FoodSecurityDataset(X_prod, y_price_prod_scaled, y_risk_prod_scaled)
    prod_loader = DataLoader(prod_dataset, batch_size=16, shuffle=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    model_prod = FoodSecurityNet(X_prod.shape[1]).to(device)
    fit_network(model_prod, prod_loader, epochs, device, learning_rate=0.008, weight_decay=0.001)

    # Predict Out-of-sample (Oct, Nov, Dec 2025 predicting Jan, Feb, Mar 2026)
    forecast_mask = df_ml["future_food_price_index"].isna()
    df_forecast = df_ml.loc[forecast_mask].sort_values(by=["region", "year_month"])
    
    X_forecast_raw = align_features(df_forecast, features).values
    X_forecast = prod_scaler.transform(X_forecast_raw)
    
    # RF Forecasts
    rf_forecast_price = rf_price_prod.predict(X_forecast_raw)
    rf_forecast_risk = rf_risk_prod.predict(X_forecast_raw)
    
    # PyTorch Forecasts
    torch_forecast_price_scaled, torch_forecast_risk_scaled = predict_network(model_prod, X_forecast, device)
    torch_forecast_price = prod_scaler_price.inverse_transform(torch_forecast_price_scaled.reshape(-1, 1)).flatten()
    torch_forecast_risk = prod_scaler_risk.inverse_transform(torch_forecast_risk_scaled.reshape(-1, 1)).flatten()
    torch_forecast_risk = np.clip(torch_forecast_risk, 0.0, 100.0)

    monitoring_baseline = build_monitoring_baseline(
        X_prod_raw,
        features,
        predictions={
            "price": [row["torch_predicted_price"] for row in fold_predictions],
            "risk": [row["torch_predicted_risk"] for row in fold_predictions],
        },
        price_errors=[
            row["actual_price"] - row["torch_predicted_price"] for row in fold_predictions
        ],
        risk_errors=[
            row["actual_risk"] - row["torch_predicted_risk"] for row in fold_predictions
        ],
    )
    interval_calibration = monitoring_baseline["prediction_intervals"]
    for row in fold_predictions:
        row["torch_price_interval"] = prediction_interval(
            row["torch_predicted_price"], interval_calibration["price"],
            interval_calibration["coverage"],
        )
        row["torch_risk_interval"] = prediction_interval(
            row["torch_predicted_risk"], interval_calibration["risk"],
            interval_calibration["coverage"], bounds=(0, 100),
        )

    # Package output JSON
    results = {
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "epoch_logs": epoch_logs,
        "metrics": mean_metrics,
        "validation": validation,
        "predictions": predictions,
        "monitoring_baseline": monitoring_baseline,
        "out_of_sample_forecasts": []
    }
    
    # Add out-of-sample forecasts (2026-01 to 2026-03)
    for i, (_, row) in enumerate(df_forecast.iterrows()):
        feature_month = row["year_month"]
        year = int(row["year"])
        month = int(row["month"])
        
        target_month = month + 3
        target_year = year
        if target_month > 12:
            target_month -= 12
            target_year += 1
            
        target_month_str = f"{target_year}-{target_month:02d}"
        
        results["out_of_sample_forecasts"].append({
            "region": row["region"],
            "feature_month": feature_month,
            "target_month": target_month_str,
            "rf_predicted_price": float(rf_forecast_price[i]),
            "torch_predicted_price": float(torch_forecast_price[i]),
            "rf_predicted_risk": float(rf_forecast_risk[i]),
            "torch_predicted_risk": float(torch_forecast_risk[i]),
            "torch_price_interval": prediction_interval(
                torch_forecast_price[i], interval_calibration["price"],
                interval_calibration["coverage"],
            ),
            "torch_risk_interval": prediction_interval(
                torch_forecast_risk[i], interval_calibration["risk"],
                interval_calibration["coverage"], bounds=(0, 100),
            ),
        })
        
    # Write JSON
    # Resolve correct dashboard output directory
    if dashboard_dir is None:
        if os.path.exists("train_deep_learning.py"):
            dashboard_dir = "."
        elif os.path.exists("dashboard_v2"):
            dashboard_dir = "dashboard_v2"
        elif os.path.exists("dashboard"):
            dashboard_dir = "dashboard"
        else:
            dashboard_dir = "."
    os.makedirs(dashboard_dir, exist_ok=True)

    weights_path = os.path.join(dashboard_dir, f"pytorch_food_security_model.{model_version}.pth")
    manifest_path = os.path.join(dashboard_dir, f"pytorch_food_security_model.{model_version}.manifest.json")
    if os.path.exists(weights_path) or os.path.exists(manifest_path):
        raise FileExistsError(
            f"Model version {model_version!r} already has artifact files; choose a new version."
        )
        
    json_path = os.path.join(dashboard_dir, "train_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nTraining execution log and out-of-sample predictions saved to: {json_path}")
    
    # Save the PyTorch Model Weights
    manifest = save_model_artifact(
        model_prod,
        weights_path,
        manifest_path,
        features,
        prod_scaler,
        prod_scaler_price,
        prod_scaler_risk,
        results["metrics"],
        model_config={"dropout": 0.05},
        monitoring=monitoring_baseline,
    )
    logger.info(
        "Production model and monitoring baseline saved",
        extra={
            "event": "training_completed",
            "artifact_version": manifest["artifact_version"],
            "features": len(features),
            "interval_coverage": interval_calibration["coverage"],
        },
    )
    print(f"Production PyTorch weights saved to: {weights_path}")
    print(f"Versioned artifact manifest saved to: {manifest_path} (v{manifest['artifact_version']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate AgriShield food-security models.")
    parser.add_argument("--csv", help="Path to input dataset CSV.")
    parser.add_argument("--output-dir", help="Directory for dashboard JSON and model artifacts.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for each neural-network stage.")
    parser.add_argument(
        "--model-version",
        help="Version label for immutable candidate artifact filenames (defaults to a UTC timestamp).",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run walk-forward validation and print its JSON without training production models or writing artifacts.",
    )
    arguments = parser.parse_args()
    validation_result = train_and_evaluate(
        arguments.csv,
        arguments.output_dir,
        arguments.epochs,
        validation_only=arguments.validation_only,
        model_version=arguments.model_version,
    )
    if arguments.validation_only:
        print(json.dumps(validation_result, indent=2))
