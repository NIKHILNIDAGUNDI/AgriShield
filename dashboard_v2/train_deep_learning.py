# PyTorch Deep Learning Training Pipeline
# Optimized for CUDA execution (e.g. RTX 5060 Graphics Card) with automatic CPU fallback
# Implements a Dual-Stage Training workflow:
# 1. Validation Stage: Train on 2021-2024 to validate performance on 2025.
# 2. Production Stage: Train on all data till date to forecast the next 3 months out-of-sample.

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Lag engineering
def prepare_forecasting_lags(df, lag_months=3):
    df_sorted = df.sort_values(by=["region", "year_month"]).copy()
    lag_features = [
        "temperature", "precipitation", "crop_yield", 
        "fuel_price", "transport_cost", "inflation_rate", 
        "geopolitical_risk", "news_sentiment", "food_price_index",
        "shortage_risk_score"
    ]
    for col in lag_features:
        for lag in [1, 2, lag_months]:
            df_sorted[f"{col}_lag_{lag}"] = df_sorted.groupby("region")[col].shift(lag)
            
    df_clean = df_sorted.dropna().reset_index(drop=True)
    return df_clean

# PyTorch Dataset
class FoodSecurityDataset(Dataset):
    def __init__(self, X, y_price, y_risk):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_price = torch.tensor(y_price, dtype=torch.float32).view(-1, 1)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_price[idx], self.y_risk[idx]

# PyTorch Deep Learning Model
class FoodSecurityNet(nn.Module):
    def __init__(self, input_dim):
        super(FoodSecurityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.05)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.05)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.out_price = nn.Linear(32, 1)
        self.out_risk = nn.Linear(32, 1)

    def forward(self, x):
        h1 = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout2(self.relu(self.bn2(self.fc2(h1))))
        h3 = self.relu(self.bn3(self.fc3(h2)))
        
        price = self.out_price(h3)
        risk = self.out_risk(h3)
        return price, risk

def train_and_evaluate():
    # Load dataset
    csv_path = "food_security_data.csv"
    if not os.path.exists(csv_path):
        csv_path = "../food_security_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("food_security_data.csv not found.")
            
    df = pd.read_csv(csv_path)
    
    # Process lags
    df_ml = prepare_forecasting_lags(df, lag_months=3)
    
    # Shift targets forward by 3 months (so lag features at t predict t+3)
    df_ml["future_food_price_index"] = df_ml.groupby("region")["food_price_index"].shift(-3)
    df_ml["future_shortage_risk_score"] = df_ml.groupby("region")["shortage_risk_score"].shift(-3)
    
    # Features are all lag columns
    features = [col for col in df_ml.columns if "_lag_" in col]
    
    # ====================================================================
    # STAGE 1: VALIDATION SET TRAINING (Train: 2021-2024, Test: 2025)
    # ====================================================================
    print("--- STAGE 1: Training Validation Model (Evaluating on 2025) ---")
    
    # Filter rows that have future targets (i.e. up to September 2025)
    df_with_targets = df_ml.dropna(subset=["future_food_price_index", "future_shortage_risk_score"]).reset_index(drop=True)
    
    train_mask = df_with_targets["year"] < 2025
    test_mask = df_with_targets["year"] == 2025
    
    X_train_raw = df_with_targets.loc[train_mask, features].values
    y_price_train = df_with_targets.loc[train_mask, "future_food_price_index"].values
    y_risk_train = df_with_targets.loc[train_mask, "future_shortage_risk_score"].values
    
    X_test_raw = df_with_targets.loc[test_mask, features].values
    y_price_test = df_with_targets.loc[test_mask, "future_food_price_index"].values
    y_risk_test = df_with_targets.loc[test_mask, "future_shortage_risk_score"].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Scale targets
    scaler_price = StandardScaler()
    scaler_risk = StandardScaler()
    y_price_train_scaled = scaler_price.fit_transform(y_price_train.reshape(-1, 1))
    y_risk_train_scaled = scaler_risk.fit_transform(y_risk_train.reshape(-1, 1))
    y_price_test_scaled = scaler_price.transform(y_price_test.reshape(-1, 1))
    y_risk_test_scaled = scaler_risk.transform(y_risk_test.reshape(-1, 1))
    
    # Baseline RF Validation training
    rf_price_val = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_price_val.fit(X_train_raw, y_price_train)
    rf_risk_val = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_risk_val.fit(X_train_raw, y_risk_train)
    
    rf_price_preds = rf_price_val.predict(X_test_raw)
    rf_risk_preds = rf_risk_val.predict(X_test_raw)
    
    # Stage 1 PyTorch DataLoader
    train_dataset = FoodSecurityDataset(X_train, y_price_train_scaled, y_risk_train_scaled)
    test_dataset = FoodSecurityDataset(X_test, y_price_test_scaled, y_risk_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    model_val = FoodSecurityNet(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model_val.parameters(), lr=0.008, weight_decay=0.001)
    criterion = nn.MSELoss()
    
    epochs = 300
    epoch_logs = []
    
    for epoch in range(1, epochs + 1):
        model_val.train()
        train_loss = 0.0
        for batch_X, batch_y_price, batch_y_risk in train_loader:
            batch_X, batch_y_price, batch_y_risk = batch_X.to(device), batch_y_price.to(device), batch_y_risk.to(device)
            optimizer.zero_grad()
            pred_price, pred_risk = model_val(batch_X)
            loss = criterion(pred_price, batch_y_price) + criterion(pred_risk, batch_y_risk)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)

        model_val.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y_price, batch_y_risk in test_loader:
                batch_X, batch_y_price, batch_y_risk = batch_X.to(device), batch_y_price.to(device), batch_y_risk.to(device)
                pred_price, pred_risk = model_val(batch_X)
                loss = criterion(pred_price, batch_y_price) + criterion(pred_risk, batch_y_risk)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(test_loader.dataset)

        epoch_logs.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})
        
    # Validation evaluation predictions
    model_val.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        torch_price_preds_scaled, torch_risk_preds_scaled = model_val(test_inputs)
        
        # Inverse transform to original scales
        torch_price_preds = scaler_price.inverse_transform(torch_price_preds_scaled.cpu().numpy()).flatten()
        torch_risk_preds = scaler_risk.inverse_transform(torch_risk_preds_scaled.cpu().numpy()).flatten()
        
        # Clip risk predictions to [0, 100]
        torch_risk_preds = np.clip(torch_risk_preds, 0.0, 100.0)
        
    # Metrics
    rf_price_rmse = np.sqrt(mean_squared_error(y_price_test, rf_price_preds))
    rf_price_r2 = r2_score(y_price_test, rf_price_preds)
    torch_price_rmse = np.sqrt(mean_squared_error(y_price_test, torch_price_preds))
    torch_price_r2 = r2_score(y_price_test, torch_price_preds)
    
    rf_risk_rmse = np.sqrt(mean_squared_error(y_risk_test, rf_risk_preds))
    rf_risk_r2 = r2_score(y_risk_test, rf_risk_preds)
    torch_risk_rmse = np.sqrt(mean_squared_error(y_risk_test, torch_risk_preds))
    torch_risk_r2 = r2_score(y_risk_test, torch_risk_preds)
    
    price_improvement = ((rf_price_rmse - torch_price_rmse) / rf_price_rmse) * 100
    risk_improvement = ((rf_risk_rmse - torch_risk_rmse) / rf_risk_rmse) * 100
    
    print(f"Validation Metrics (Test Set 2025):")
    print(f"Price Forecast (Standard AI RF RMSE): {rf_price_rmse:.3f} | R2: {rf_price_r2*100:.1f}%")
    print(f"Price Forecast (Advanced AI NN RMSE): {torch_price_rmse:.3f} | R2: {torch_price_r2*100:.1f}%")
    print(f"Shortage Risk  (Standard AI RF RMSE): {rf_risk_rmse:.3f} | R2: {rf_risk_r2*100:.1f}%")
    print(f"Shortage Risk  (Advanced AI NN RMSE): {torch_risk_rmse:.3f} | R2: {torch_risk_r2*100:.1f}%")
    
    # ====================================================================
    # STAGE 2: PRODUCTION TRAINING (Train on 100% of data to predict out-of-sample)
    # ====================================================================
    print("\n--- STAGE 2: Training Production Model (Predicting 2026-01 to 2026-03) ---")
    
    X_prod_raw = df_with_targets[features].values
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
    optimizer_prod = optim.Adam(model_prod.parameters(), lr=0.008, weight_decay=0.001)
    
    for epoch in range(1, epochs + 1):
        model_prod.train()
        for batch_X, batch_y_price, batch_y_risk in prod_loader:
            batch_X, batch_y_price, batch_y_risk = batch_X.to(device), batch_y_price.to(device), batch_y_risk.to(device)
            optimizer_prod.zero_grad()
            pred_price, pred_risk = model_prod(batch_X)
            loss = criterion(pred_price, batch_y_price) + criterion(pred_risk, batch_y_risk)
            loss.backward()
            optimizer_prod.step()

    # Predict Out-of-sample (Oct, Nov, Dec 2025 predicting Jan, Feb, Mar 2026)
    forecast_mask = df_ml["future_food_price_index"].isna()
    df_forecast = df_ml.loc[forecast_mask].sort_values(by=["region", "year_month"])
    
    X_forecast_raw = df_forecast[features].values
    X_forecast = prod_scaler.transform(X_forecast_raw)
    
    # RF Forecasts
    rf_forecast_price = rf_price_prod.predict(X_forecast_raw)
    rf_forecast_risk = rf_risk_prod.predict(X_forecast_raw)
    
    # PyTorch Forecasts
    model_prod.eval()
    with torch.no_grad():
        forecast_inputs = torch.tensor(X_forecast, dtype=torch.float32).to(device)
        torch_forecast_price_scaled, torch_forecast_risk_scaled = model_prod(forecast_inputs)
        
        # Inverse scale forecasts
        torch_forecast_price = prod_scaler_price.inverse_transform(torch_forecast_price_scaled.cpu().numpy()).flatten()
        torch_forecast_risk = prod_scaler_risk.inverse_transform(torch_forecast_risk_scaled.cpu().numpy()).flatten()
        
        # Clip risk predictions to [0, 100]
        torch_forecast_risk = np.clip(torch_forecast_risk, 0.0, 100.0)

    # Package output JSON
    results = {
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "epoch_logs": epoch_logs, # Validation logs
        "metrics": {
            "rf_price_rmse": float(rf_price_rmse),
            "rf_price_r2": float(rf_price_r2),
            "torch_price_rmse": float(torch_price_rmse),
            "torch_price_r2": float(torch_price_r2),
            "price_improvement": float(price_improvement),
            
            "rf_risk_rmse": float(rf_risk_rmse),
            "rf_risk_r2": float(rf_risk_r2),
            "torch_risk_rmse": float(torch_risk_rmse),
            "torch_risk_r2": float(torch_risk_r2),
            "risk_improvement": float(risk_improvement)
        },
        "predictions": [],
        "out_of_sample_forecasts": []
    }
    
    # Add validation set predictions (target months mapped to 2025)
    test_df = df_with_targets.loc[test_mask].sort_values(by=["region", "year_month"])
    for i, (_, row) in enumerate(test_df.iterrows()):
        year = int(row["year"])
        month = int(row["month"])
        
        t_month = month + 3
        t_year = year
        if t_month > 12:
            t_month -= 12
            t_year += 1
            
        target_month_str = f"{t_year}-{t_month:02d}"
        
        results["predictions"].append({
            "region": row["region"],
            "target_month": target_month_str,
            "feature_month": row["year_month"],
            "actual_price": float(row["future_food_price_index"]),
            "rf_predicted_price": float(rf_price_preds[i]),
            "torch_predicted_price": float(torch_price_preds[i]),
            "actual_risk": float(row["future_shortage_risk_score"]),
            "rf_predicted_risk": float(rf_risk_preds[i]),
            "torch_predicted_risk": float(torch_risk_preds[i])
        })
        
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
            "torch_predicted_risk": float(torch_forecast_risk[i])
        })
        
    # Write JSON
    # Resolve correct dashboard output directory
    if os.path.exists("train_deep_learning.py"):
        dashboard_dir = "."
    elif os.path.exists("dashboard_v2"):
        dashboard_dir = "dashboard_v2"
    elif os.path.exists("dashboard"):
        dashboard_dir = "dashboard"
    else:
        dashboard_dir = "."
        
    json_path = os.path.join(dashboard_dir, "train_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nTraining execution log and out-of-sample predictions saved to: {json_path}")
    
    # Save the PyTorch Model Weights
    weights_path = os.path.join(dashboard_dir, "pytorch_food_security_model.pth")
    torch.save(model_prod.state_dict(), weights_path)
    print(f"Production PyTorch weights saved to: {weights_path}")

if __name__ == "__main__":
    train_and_evaluate()
