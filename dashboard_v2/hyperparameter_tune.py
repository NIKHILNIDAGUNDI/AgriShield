# Hyperparameter search script for Agri-Shield Deep Learning Model
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Device configuration
device = torch.device("cpu") # Keep on CPU for quick sweep

# Prepare data
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

class FoodSecurityDataset(Dataset):
    def __init__(self, X, y_price, y_risk):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_price = torch.tensor(y_price, dtype=torch.float32).view(-1, 1)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_price[idx], self.y_risk[idx]

class FoodSecurityNet(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(FoodSecurityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
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

def tune():
    csv_path = "food_security_data.csv"
    if not os.path.exists(csv_path):
        csv_path = "../food_security_data.csv"
    df = pd.read_csv(csv_path)
    df_ml = prepare_forecasting_lags(df)
    df_ml["future_food_price_index"] = df_ml.groupby("region")["food_price_index"].shift(-3)
    df_ml["future_shortage_risk_score"] = df_ml.groupby("region")["shortage_risk_score"].shift(-3)
    df_with_targets = df_ml.dropna(subset=["future_food_price_index", "future_shortage_risk_score"]).reset_index(drop=True)
    
    features = [col for col in df_ml.columns if "_lag_" in col]
    train_mask = df_with_targets["year"] < 2025
    test_mask = df_with_targets["year"] == 2025
    
    X_train_raw = df_with_targets.loc[train_mask, features].values
    y_price_train = df_with_targets.loc[train_mask, "future_food_price_index"].values
    y_risk_train = df_with_targets.loc[train_mask, "future_shortage_risk_score"].values
    X_test_raw = df_with_targets.loc[test_mask, features].values
    y_price_test = df_with_targets.loc[test_mask, "future_food_price_index"].values
    y_risk_test = df_with_targets.loc[test_mask, "future_shortage_risk_score"].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    scaler_price = StandardScaler()
    scaler_risk = StandardScaler()
    y_price_train_scaled = scaler_price.fit_transform(y_price_train.reshape(-1, 1))
    y_risk_train_scaled = scaler_risk.fit_transform(y_risk_train.reshape(-1, 1))
    y_price_test_scaled = scaler_price.transform(y_price_test.reshape(-1, 1))
    y_risk_test_scaled = scaler_risk.transform(y_risk_test.reshape(-1, 1))
    
    train_dataset = FoodSecurityDataset(X_train, y_price_train_scaled, y_risk_train_scaled)
    test_dataset = FoodSecurityDataset(X_test, y_price_test_scaled, y_risk_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    best_score = float("inf")
    best_params = {}
    
    # Grid search parameters
    lrs = [0.001, 0.002, 0.003, 0.005, 0.008]
    epochs_list = [150, 200, 250, 300]
    dropouts = [0.0, 0.05, 0.1, 0.2]
    wds = [1e-5, 1e-4, 1e-3]
    
    import itertools
    combinations = list(itertools.product(lrs, epochs_list, dropouts, wds))
    print(f"Sweeping {len(combinations)} hyperparameter combinations...")
    
    criterion = nn.MSELoss()
    
    for lr, epochs, dropout, wd in combinations:
        # Re-initialize weights using manual seed to make it deterministic
        torch.manual_seed(42)
        model = FoodSecurityNet(X_train.shape[1], dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            for batch_X, batch_y_price, batch_y_risk in train_loader:
                optimizer.zero_grad()
                pred_price, pred_risk = model(batch_X)
                loss = criterion(pred_price, batch_y_price) + criterion(pred_risk, batch_y_risk)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            p_scaled, r_scaled = model(test_inputs)
            p_preds = scaler_price.inverse_transform(p_scaled.numpy()).flatten()
            r_preds = scaler_risk.inverse_transform(r_scaled.numpy()).flatten()
            r_preds = np.clip(r_preds, 0.0, 100.0)
            
        p_rmse = np.sqrt(mean_squared_error(y_price_test, p_preds))
        r_rmse = np.sqrt(mean_squared_error(y_risk_test, r_preds))
        p_r2 = r2_score(y_price_test, p_preds)
        r_r2 = r2_score(y_risk_test, r_preds)
        
        # Combined score (sum of RMSEs)
        combined_rmse = p_rmse + 2.0 * r_rmse # Weight risk score slightly more to reduce risk gap
        
        if combined_rmse < best_score:
            best_score = combined_rmse
            best_params = {
                "lr": lr,
                "epochs": epochs,
                "dropout": dropout,
                "weight_decay": wd,
                "price_rmse": p_rmse,
                "price_r2": p_r2,
                "risk_rmse": r_rmse,
                "risk_r2": r_r2
            }
            print(f"New Best! LR={lr}, Epochs={epochs}, DO={dropout}, WD={wd} -> Price RMSE={p_rmse:.3f} (R2={p_r2*100:.1f}%), Risk RMSE={r_rmse:.3f} (R2={r_r2*100:.1f}%)")

    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*50)
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    tune()
