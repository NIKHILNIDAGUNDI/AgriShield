"""Grid search for the shared AgriShield neural-network implementation."""

import itertools
import os

import numpy as np
import pandas as pd
import torch
from agri_shield_ml import (
    FoodSecurityDataset,
    FoodSecurityNet,
    align_features,
    build_feature_names,
    fit_network,
    predict_network,
    prepare_forecasting_lags,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

device = torch.device("cpu")


def tune():
    csv_path = "food_security_data.csv"
    if not os.path.exists(csv_path):
        csv_path = "../food_security_data.csv"
    df = pd.read_csv(csv_path)
    df_ml = prepare_forecasting_lags(df)
    df_ml["future_food_price_index"] = df_ml.groupby("region")["food_price_index"].shift(-3)
    df_ml["future_shortage_risk_score"] = df_ml.groupby("region")["shortage_risk_score"].shift(-3)
    df_with_targets = df_ml.dropna(
        subset=["future_food_price_index", "future_shortage_risk_score"]
    ).reset_index(drop=True)

    features = build_feature_names()
    train_mask = df_with_targets["year"] < 2025
    test_mask = df_with_targets["year"] == 2025
    X_train_raw = align_features(df_with_targets.loc[train_mask], features).values
    y_price_train = df_with_targets.loc[train_mask, "future_food_price_index"].values
    y_risk_train = df_with_targets.loc[train_mask, "future_shortage_risk_score"].values
    X_test_raw = align_features(df_with_targets.loc[test_mask], features).values
    y_price_test = df_with_targets.loc[test_mask, "future_food_price_index"].values
    y_risk_test = df_with_targets.loc[test_mask, "future_shortage_risk_score"].values

    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    price_scaler = StandardScaler()
    risk_scaler = StandardScaler()
    price_train_scaled = price_scaler.fit_transform(y_price_train.reshape(-1, 1))
    risk_train_scaled = risk_scaler.fit_transform(y_risk_train.reshape(-1, 1))
    train_dataset = FoodSecurityDataset(X_train, price_train_scaled, risk_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_score = float("inf")
    best_params = {}
    lrs = [0.001, 0.002, 0.003, 0.005, 0.008]
    epochs_list = [150, 200, 250, 300]
    dropouts = [0.0, 0.05, 0.1, 0.2]
    weight_decays = [1e-5, 1e-4, 1e-3]
    combinations = list(itertools.product(lrs, epochs_list, dropouts, weight_decays))
    print(f"Sweeping {len(combinations)} hyperparameter combinations...")

    for learning_rate, epochs, dropout, weight_decay in combinations:
        torch.manual_seed(42)
        model = FoodSecurityNet(X_train.shape[1], dropout)
        fit_network(
            model, train_loader, epochs, device,
            learning_rate=learning_rate, weight_decay=weight_decay,
        )
        price_scaled, risk_scaled = predict_network(model, X_test, device)
        price_predictions = price_scaler.inverse_transform(price_scaled.reshape(-1, 1)).flatten()
        risk_predictions = risk_scaler.inverse_transform(risk_scaled.reshape(-1, 1)).flatten()
        risk_predictions = np.clip(risk_predictions, 0.0, 100.0)

        price_rmse = np.sqrt(mean_squared_error(y_price_test, price_predictions))
        risk_rmse = np.sqrt(mean_squared_error(y_risk_test, risk_predictions))
        price_r2 = r2_score(y_price_test, price_predictions)
        risk_r2 = r2_score(y_risk_test, risk_predictions)
        combined_rmse = price_rmse + 2.0 * risk_rmse

        if combined_rmse < best_score:
            best_score = combined_rmse
            best_params = {
                "lr": learning_rate,
                "epochs": epochs,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "price_rmse": price_rmse,
                "price_r2": price_r2,
                "risk_rmse": risk_rmse,
                "risk_r2": risk_r2,
            }
            print(
                f"New Best! LR={learning_rate}, Epochs={epochs}, DO={dropout}, "
                f"WD={weight_decay} -> Price RMSE={price_rmse:.3f} "
                f"(R2={price_r2*100:.1f}%), Risk RMSE={risk_rmse:.3f} "
                f"(R2={risk_r2*100:.1f}%)"
            )

    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 50)
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    tune()
