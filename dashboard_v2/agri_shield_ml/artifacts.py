"""Versioned model artifact persistence with integrity verification."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from . import __version__
from .model import FoodSecurityNet

ARTIFACT_SCHEMA = "agri-shield-ml-model"
ARTIFACT_VERSION = 1


def _scaler_metadata(scaler):
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "variance": scaler.var_.astype(float).tolist(),
        "features": int(scaler.n_features_in_),
    }


def _restore_scaler(metadata):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(metadata["mean"], dtype=float)
    scaler.scale_ = np.asarray(metadata["scale"], dtype=float)
    scaler.var_ = np.asarray(metadata["variance"], dtype=float)
    scaler.n_features_in_ = metadata["features"]
    scaler.n_samples_seen_ = 1
    return scaler


def _write_json_atomic(path, payload):
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_model_artifact(model, weights_path, manifest_path, feature_names,
                        feature_scaler, price_scaler, risk_scaler, metrics,
                        model_config=None, monitoring=None):
    """Save CPU state weights and a JSON manifest binding preprocessing and metrics."""
    weights_path = Path(weights_path)
    manifest_path = Path(manifest_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = weights_path.with_name(f".{weights_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary_path)
        os.replace(temporary_path, weights_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "package_version": __version__,
        "model": {
            "architecture": "FoodSecurityNet",
            "input_dim": len(feature_names),
            "dropout": float((model_config or {}).get("dropout", 0.05)),
        },
        "feature_names": list(feature_names),
        "scalers": {
            "features": _scaler_metadata(feature_scaler),
            "price_target": _scaler_metadata(price_scaler),
            "risk_target": _scaler_metadata(risk_scaler),
        },
        "metrics": metrics,
        "monitoring": monitoring or {},
        "weights": {
            "file": weights_path.name,
            "sha256": digest,
        },
        "runtime": {"torch_version": torch.__version__},
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_model_artifact(manifest_path, map_location="cpu"):
    """Load a verified artifact manifest and its model weights."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported AgriShield model artifact schema/version")
    if manifest.get("model", {}).get("architecture") != "FoodSecurityNet":
        raise ValueError("Unsupported model architecture")
    feature_names = manifest.get("feature_names", [])
    if (
        len(feature_names) != manifest["model"].get("input_dim")
        or len(feature_names) != len(set(feature_names))
    ):
        raise ValueError("Artifact feature schema does not match model input dimensions")
    weights_name = manifest.get("weights", {}).get("file", "")
    if not weights_name or Path(weights_name).name != weights_name:
        raise ValueError("Artifact weights must be a file in the manifest directory")

    weights_path = manifest_path.parent / weights_name
    actual_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if actual_hash != manifest["weights"].get("sha256"):
        raise ValueError("Model weights hash does not match artifact manifest")

    model_config = manifest["model"]
    model = FoodSecurityNet(
        input_dim=model_config["input_dim"],
        dropout=model_config.get("dropout", 0.05),
    )
    try:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    scaler_metadata = manifest["scalers"]
    return {
        "manifest": manifest,
        "model": model,
        "feature_scaler": _restore_scaler(scaler_metadata["features"]),
        "price_scaler": _restore_scaler(scaler_metadata["price_target"]),
        "risk_scaler": _restore_scaler(scaler_metadata["risk_target"]),
    }
