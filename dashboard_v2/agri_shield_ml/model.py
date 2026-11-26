"""Shared PyTorch model and dataset definitions."""

import torch
from torch import nn
from torch.utils.data import Dataset


class FoodSecurityDataset(Dataset):
    def __init__(self, features, price_targets, risk_targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.price_targets = torch.as_tensor(price_targets, dtype=torch.float32).view(-1, 1)
        self.risk_targets = torch.as_tensor(risk_targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.price_targets[index], self.risk_targets[index]


class FoodSecurityNet(nn.Module):
    def __init__(self, input_dim, dropout=0.05):
        super().__init__()
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

    def forward(self, features):
        hidden = self.dropout1(self.relu(self.bn1(self.fc1(features))))
        hidden = self.dropout2(self.relu(self.bn2(self.fc2(hidden))))
        hidden = self.relu(self.bn3(self.fc3(hidden)))
        return self.out_price(hidden), self.out_risk(hidden)
