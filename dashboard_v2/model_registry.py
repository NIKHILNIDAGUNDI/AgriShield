"""List registered models or atomically roll the active model back to a version."""

import argparse
import hashlib
import hmac
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agri_shield_ml.observability import get_json_logger

REGISTRY_SCHEMA = "agri-shield-model-registry"
REGISTRY_VERSION = 1
LOGGER = get_json_logger("agrishield.registry")


def _validate_target(registry, root, version, artifact_key=None):
    entry = registry.get("models", {}).get(version)
    if not isinstance(entry, dict) or str(entry.get("version")) != str(version):
        raise ValueError(f"Model version {version!r} is not registered")
    manifest_name = entry.get("manifest")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise ValueError(f"Model version {version!r} has an invalid manifest path")
    manifest_bytes = (root / manifest_name).read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if not hmac.compare_digest(digest, entry.get("manifest_sha256", "")):
        raise ValueError(f"Model version {version!r} manifest digest does not match")
    if artifact_key:
        signature = hmac.new(
            artifact_key.encode("utf-8"), manifest_bytes, hashlib.sha256
        ).hexdigest()
        if not entry.get("hmac_sha256") or not hmac.compare_digest(
            signature, entry["hmac_sha256"]
        ):
            raise ValueError(f"Model version {version!r} HMAC signature is missing or invalid")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != "agri-shield-ml-model":
        raise ValueError(f"Model version {version!r} is not an AgriShield model artifact")
    weights_name = manifest.get("weights", {}).get("file", "")
    if not weights_name or Path(weights_name).name != weights_name:
        raise ValueError(f"Model version {version!r} has an invalid weights path")
    weights_digest = hashlib.sha256((root / weights_name).read_bytes()).hexdigest()
    if not hmac.compare_digest(
        weights_digest, manifest.get("weights", {}).get("sha256", "")
    ):
        raise ValueError(f"Model version {version!r} weights digest does not match")
    return entry


def list_models(root):
    root = Path(root).resolve()
    registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported AgriShield model registry schema/version")
    return {
        "active_model": registry.get("active_model"),
        "models": registry.get("models", {}),
        "rollback_metadata": registry.get("rollback_metadata", {}),
    }


def rollback_model(root, version, reason="operator rollback", artifact_key=None):
    """Verify a registered artifact, switch active version, and retain rollback history."""
    root = Path(root).resolve()
    path = root / "model_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported AgriShield model registry schema/version")
    if not artifact_key and "AGRISHIELD_ARTIFACT_KEY" in os.environ:
        artifact_key = os.environ.get("AGRISHIELD_ARTIFACT_KEY")
        if not artifact_key:
            raise ValueError("AGRISHIELD_ARTIFACT_KEY is set but empty")
    _validate_target(registry, root, version, artifact_key=artifact_key)
    previous = str(registry.get("active_model", ""))
    version = str(version)
    if previous == version:
        raise ValueError(f"Model version {version!r} is already active")
    event = {
        "from": previous,
        "to": version,
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": str(reason),
    }
    metadata = registry.setdefault("rollback_metadata", {})
    history = metadata.setdefault("history", [])
    history.append(event)
    metadata["last_rollback"] = event
    registry["active_model"] = version

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    LOGGER.warning(
        "Active model rolled back",
        extra={"event": "model_rollback", **event},
    )
    return event


def register_model(root, version, manifest_name, activate=False,
                   reason="operator registration", artifact_key=None):
    """Register an immutable manifest/weights pair, optionally making it active."""
    root = Path(root).resolve()
    path = root / "model_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported AgriShield model registry schema/version")
    version = str(version)
    if not version or version in registry.get("models", {}):
        raise ValueError(f"Model version {version!r} is empty or already registered")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise ValueError("Manifest must be a file in the dashboard directory")
    if not artifact_key and "AGRISHIELD_ARTIFACT_KEY" in os.environ:
        artifact_key = os.environ.get("AGRISHIELD_ARTIFACT_KEY")
        if not artifact_key:
            raise ValueError("AGRISHIELD_ARTIFACT_KEY is set but empty")
    manifest_path = root / manifest_name
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != "agri-shield-ml-model":
        raise ValueError("Manifest is not an AgriShield model artifact")
    weights_name = manifest.get("weights", {}).get("file", "")
    if not weights_name or Path(weights_name).name != weights_name:
        raise ValueError("Manifest weights must be a file in the dashboard directory")
    weights_digest = hashlib.sha256((root / weights_name).read_bytes()).hexdigest()
    if not hmac.compare_digest(weights_digest, manifest["weights"].get("sha256", "")):
        raise ValueError("Model weights hash does not match artifact manifest")

    entry = {
        "version": version,
        "manifest": manifest_name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    if artifact_key:
        entry["hmac_sha256"] = hmac.new(
            artifact_key.encode("utf-8"), manifest_bytes, hashlib.sha256
        ).hexdigest()
    registry.setdefault("models", {})[version] = entry
    event = None
    if activate:
        previous = str(registry.get("active_model", ""))
        event = {
            "from": previous,
            "to": version,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": str(reason),
        }
        registry["active_model"] = version
        metadata = registry.setdefault("rollback_metadata", {})
        metadata.setdefault("activation_history", []).append(event)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    LOGGER.info(
        "Model version registered",
        extra={
            "event": "model_registered",
            "version": version,
            "activated": bool(activate),
            "previous_active": event["from"] if event else registry.get("active_model"),
        },
    )
    return {"version": version, "activated": bool(activate), "activation": event}


def activate_model(root, version, reason="operator activation", artifact_key=None):
    """Activate a registered model after verifying its manifest and optional HMAC."""
    root = Path(root).resolve()
    path = root / "model_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError("Unsupported AgriShield model registry schema/version")
    if not artifact_key and "AGRISHIELD_ARTIFACT_KEY" in os.environ:
        artifact_key = os.environ.get("AGRISHIELD_ARTIFACT_KEY")
        if not artifact_key:
            raise ValueError("AGRISHIELD_ARTIFACT_KEY is set but empty")
    _validate_target(registry, root, str(version), artifact_key=artifact_key)
    previous = str(registry.get("active_model", ""))
    version = str(version)
    if previous == version:
        raise ValueError(f"Model version {version!r} is already active")
    event = {
        "from": previous,
        "to": version,
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": str(reason),
    }
    metadata = registry.setdefault("rollback_metadata", {})
    metadata.setdefault("activation_history", []).append(event)
    registry["active_model"] = version
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    LOGGER.info(
        "Registered model activated",
        extra={"event": "model_activated", **event},
    )
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List registered model versions.")
    action.add_argument("--rollback", metavar="VERSION", help="Activate a verified model version.")
    action.add_argument("--register", metavar="VERSION", help="Register immutable candidate artifacts.")
    action.add_argument("--activate", metavar="VERSION", help="Activate an already registered version.")
    parser.add_argument("--manifest", help="Candidate manifest filename (required with --register).")
    parser.add_argument("--reason", default="operator rollback", help="Reason recorded in registry history.")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list_models(args.root), indent=2, ensure_ascii=False))
    elif args.register:
        if not args.manifest:
            parser.error("--manifest is required with --register")
        print(json.dumps(
            register_model(args.root, args.register, args.manifest, False, args.reason),
            indent=2,
        ))
    elif args.activate:
        if args.manifest:
            parser.error("--manifest is only valid with --register")
        print(json.dumps(activate_model(args.root, args.activate, args.reason), indent=2))
    else:
        if args.manifest:
            parser.error("--manifest is only valid with --register")
        print(json.dumps(rollback_model(args.root, args.rollback, args.reason), indent=2))


if __name__ == "__main__":
    main()
