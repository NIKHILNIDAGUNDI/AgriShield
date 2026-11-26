"""Reusable training and inference helpers."""

import torch
from torch import nn, optim


def fit_network(model, train_loader, epochs, device, learning_rate=0.008,
                weight_decay=0.001, validation_loader=None):
    """Train a two-output network and optionally record per-epoch losses."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, price_target, risk_target in train_loader:
            features, price_target, risk_target = (
                features.to(device), price_target.to(device), risk_target.to(device)
            )
            optimizer.zero_grad()
            price_prediction, risk_prediction = model(features)
            loss = criterion(price_prediction, price_target) + criterion(risk_prediction, risk_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)

        if validation_loader is not None:
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for features, price_target, risk_target in validation_loader:
                    features, price_target, risk_target = (
                        features.to(device), price_target.to(device), risk_target.to(device)
                    )
                    price_prediction, risk_prediction = model(features)
                    loss = criterion(price_prediction, price_target) + criterion(risk_prediction, risk_target)
                    validation_loss += loss.item() * features.size(0)
            validation_loss /= len(validation_loader.dataset)
            epoch_logs.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(validation_loss),
            })
    return epoch_logs


def predict_network(model, features, device):
    """Run inference and return flattened price/risk outputs as NumPy arrays."""
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        price, risk = model(tensor)
    return price.detach().cpu().numpy().reshape(-1), risk.detach().cpu().numpy().reshape(-1)
