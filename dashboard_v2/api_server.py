"""Versioned standard-library HTTP API and static server for AgriShield."""

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sys
import threading
import uuid
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import torch
from agri_shield_ml import FEATURE_SOURCES, load_model_artifact, predict_network
from agri_shield_ml.monitoring import (
    monitor_features,
    monitor_population,
    prediction_interval,
)
from agri_shield_ml.observability import get_json_logger

API_VERSION = "1.0.0"
REGISTRY_SCHEMA = "agri-shield-model-registry"
REGISTRY_VERSION = 1
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ROWS = 5000
ROOT = Path(__file__).resolve().parent
LOGGER = get_json_logger("agrishield.api")
LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class APIError(ValueError):
    """An invalid API request that should be returned to the caller."""


def canonical_json(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _next_month(year_month, offset=3):
    year, month = (int(value) for value in year_month.split("-"))
    month += offset
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return f"{year:04d}-{month:02d}"


def _validate_numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise APIError(f"{label} must be a finite number")
    return number


class AgriShieldAPI:
    """Loads a verified model once and serves prediction, registry, and audit data."""

    def __init__(self, root=ROOT, artifact_key=None, audit_path=None, registry_path=None):
        self.root = Path(root).resolve()
        self.registry_path = _resolve_path(registry_path, self.root, "model_registry.json")
        self.audit_path = _resolve_path(audit_path, self.root, "audit.jsonl")
        self.monitoring_path = self.audit_path.with_name("monitoring.jsonl")
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8"):
                pass
            with self.monitoring_path.open("a", encoding="utf-8"):
                pass
        except OSError as error:
            raise RuntimeError(f"Audit storage is not writable: {error}") from error
        self.require_hmac = (
            "AGRISHIELD_ARTIFACT_KEY" in os.environ
            if artifact_key is None
            else bool(artifact_key)
        )
        self.artifact_key = artifact_key if artifact_key is not None else os.environ.get(
            "AGRISHIELD_ARTIFACT_KEY"
        )
        if self.require_hmac and not self.artifact_key:
            raise RuntimeError("AGRISHIELD_ARTIFACT_KEY is set but empty")
        self.audit_lock = threading.Lock()
        self.registry, self.model_version, self.model_bundle = self._load_registered_model()
        self.feature_names = tuple(self.model_bundle["manifest"]["feature_names"])
        self.monitor_lock = threading.Lock()

    def _read_verified_manifest(self, registry, registry_path=None):
        registry_path = Path(registry_path or self.registry_path)
        if registry.get("schema") != REGISTRY_SCHEMA or registry.get("registry_version") != REGISTRY_VERSION:
            raise RuntimeError("Unsupported AgriShield model registry schema/version")
        active_version = registry.get("active_model")
        entry = registry.get("models", {}).get(active_version)
        if not active_version or not isinstance(entry, dict):
            raise RuntimeError("Model registry active_model is missing or invalid")
        manifest_name = entry.get("manifest")
        if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
            raise RuntimeError("Model registry manifest must be a file in the dashboard directory")

        manifest_path = registry_path.parent / manifest_name
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        expected_digest = entry.get("manifest_sha256", "")
        if not hmac.compare_digest(manifest_digest, expected_digest):
            raise RuntimeError("Model manifest SHA-256 digest does not match the registry")

        if self.require_hmac:
            expected_signature = entry.get("hmac_sha256", "")
            actual_signature = hmac.new(
                self.artifact_key.encode("utf-8"), manifest_bytes, hashlib.sha256
            ).hexdigest()
            if not expected_signature or not hmac.compare_digest(
                actual_signature, expected_signature
            ):
                raise RuntimeError("Model manifest HMAC-SHA256 signature is missing or invalid")
            verification = "hmac-sha256"
        else:
            verification = "sha256-manifest-digest"

        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("schema") != "agri-shield-ml-model":
            raise RuntimeError("Registered artifact is not an AgriShield model manifest")
        if not entry.get("version") or str(entry["version"]) != str(active_version):
            raise RuntimeError("Model registry version does not match its active model entry")
        return manifest_path, manifest, verification

    def _load_registered_model(self):
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        manifest_path, _manifest, verification = self._read_verified_manifest(
            registry, registry_path=self.registry_path
        )
        bundle = load_model_artifact(manifest_path, map_location="cpu")
        bundle["verification"] = verification
        version = str(registry["active_model"])
        return registry, version, bundle

    def health(self):
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "active_model_version": self.model_version,
        }

    def models(self):
        entries = []
        for version, model in self.registry.get("models", {}).items():
            entries.append({
                "version": version,
                "active": version == self.model_version,
                "artifact_schema": "agri-shield-ml-model",
                "verification": self.model_bundle["verification"],
                "manifest_sha256": model.get("manifest_sha256"),
                "feature_names": self.feature_names,
            })
        return {
            "api_version": API_VERSION,
            "registry_version": REGISTRY_VERSION,
            "active_model_version": self.model_version,
            "rollback_metadata": self.registry.get("rollback_metadata", {}),
            "models": entries,
        }

    def _feature_vector(self, values):
        if not isinstance(values, dict):
            raise APIError("features must be an object keyed by manifest feature name")
        missing = [name for name in self.feature_names if name not in values]
        extra = sorted(set(values).difference(self.feature_names))
        if missing or extra:
            raise APIError(
                f"Feature names do not match model schema (missing={missing}, extra={extra})"
            )
        return [_validate_numeric(values[name], name) for name in self.feature_names]

    def _row_predictions(self, rows, forecast_only):
        if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
            raise APIError(f"rows must be a non-empty array of at most {MAX_ROWS} records")

        grouped = {}
        seen = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise APIError(f"rows[{index}] must be an object")
            region = row.get("region")
            year_month = row.get("year_month")
            if not isinstance(region, str) or not region.strip() or len(region) > 100:
                raise APIError(f"rows[{index}].region must be a non-empty string")
            if not isinstance(year_month, str) or not MONTH_PATTERN.fullmatch(year_month):
                raise APIError(f"rows[{index}].year_month must use YYYY-MM format")
            key = (region, year_month)
            if key in seen:
                raise APIError(f"Duplicate region/month record: {region} {year_month}")
            seen.add(key)
            values = {}
            for source in FEATURE_SOURCES:
                if source not in row:
                    raise APIError(f"rows[{index}] is missing {source}")
                values[source] = _validate_numeric(row[source], f"rows[{index}].{source}")
            grouped.setdefault(region.strip(), []).append((year_month, values))

        vectors = []
        metadata = []
        for region, records in grouped.items():
            records.sort(key=lambda item: item[0])
            for index in range(1, len(records)):
                previous, current = records[index - 1], records[index]
                if _next_month(previous[0], 1) != current[0]:
                    raise APIError(f"Monthly records for {region} must be consecutive")
            region_features = []
            for index in range(3, len(records)):
                month, _ = records[index]
                feature_values = {}
                for source in FEATURE_SOURCES:
                    for lag, previous_index in ((1, index - 1), (2, index - 2), (3, index - 3)):
                        feature_values[f"{source}_lag_{lag}"] = records[previous_index][1][source]
                vector = self._feature_vector(feature_values)
                region_features.append((month, vector))
            if forecast_only:
                region_features = region_features[-3:]
            for feature_month, vector in region_features:
                vectors.append(vector)
                metadata.append({
                    "region": region,
                    "feature_month": feature_month,
                    "target_month": _next_month(feature_month),
                })

        if not vectors:
            raise APIError("At least four monthly records per region are required")
        return vectors, metadata

    def _predict_vectors(self, vectors):
        feature_array = np.asarray(vectors, dtype=np.float64)
        scaler = self.model_bundle["feature_scaler"]
        scaled_features = scaler.transform(feature_array)
        price_scaled, risk_scaled = predict_network(
            self.model_bundle["model"], scaled_features, torch.device("cpu")
        )
        price = self.model_bundle["price_scaler"].inverse_transform(
            price_scaled.reshape(-1, 1)
        ).reshape(-1)
        risk = self.model_bundle["risk_scaler"].inverse_transform(
            risk_scaled.reshape(-1, 1)
        ).reshape(-1)
        monitoring = self.model_bundle["manifest"].get("monitoring") or {}
        interval_config = monitoring.get("prediction_intervals") or {}
        coverage = float(interval_config.get("coverage", 0.9))
        price_width = interval_config.get("price")
        risk_width = interval_config.get("risk")
        if price_width is None:
            price_width = 1.645 * float(
                self.model_bundle["manifest"].get("metrics", {}).get("torch_price_rmse", 0.0)
            )
            price_calibration = "normal_approximation_from_rmse"
        else:
            price_calibration = interval_config.get("method", "validation_absolute_error")
        if risk_width is None:
            risk_width = 1.645 * float(
                self.model_bundle["manifest"].get("metrics", {}).get("torch_risk_rmse", 0.0)
            )
            risk_calibration = "normal_approximation_from_rmse"
        else:
            risk_calibration = interval_config.get("method", "validation_absolute_error")
        return [
            {
                "predicted_price": float(p),
                "predicted_shortage_risk": float(np.clip(r, 0, 100)),
                "price_interval": prediction_interval(
                    p, price_width, coverage, calibration=price_calibration
                ),
                "shortage_risk_interval": prediction_interval(
                    np.clip(r, 0, 100), risk_width, coverage, bounds=(0, 100),
                    calibration=risk_calibration,
                ),
            }
            for p, r in zip(price, risk)
        ]

    def _monitor_predictions(self, vectors, predictions):
        baseline = self.model_bundle["manifest"].get("monitoring") or {}
        feature_baselines = baseline.get("feature_population") or {}
        if not feature_baselines:
            scaler = self.model_bundle["feature_scaler"]
            feature_baselines = {
                name: {"mean": float(mean), "std": float(scale)}
                for name, mean, scale in zip(
                    self.feature_names, scaler.mean_, scaler.scale_
                )
            }
        feature_report = monitor_features(vectors, self.feature_names, {
            "feature_population": feature_baselines,
        })
        prediction_baselines = baseline.get("prediction_population") or {}
        prediction_report = {}
        for field, key in (
            ("predicted_price", "price"),
            ("predicted_shortage_risk", "risk"),
        ):
            if key in prediction_baselines:
                prediction_report[key] = monitor_population(
                    [row[field] for row in predictions], prediction_baselines[key]
                )
        return {
            "baseline_available": bool(feature_baselines),
            "feature_drift": feature_report,
            "prediction_drift": prediction_report,
        }

    def predict(self, request_data):
        if not isinstance(request_data, dict):
            raise APIError("Request body must be a JSON object")
        forecast_only = request_data.get("forecast_only", False)
        if not isinstance(forecast_only, bool):
            raise APIError("forecast_only must be a boolean")

        if "rows" in request_data:
            vectors, metadata = self._row_predictions(request_data["rows"], forecast_only)
            results = self._predict_vectors(vectors)
            predictions = [
                {**meta, **prediction} for meta, prediction in zip(metadata, results)
            ]
            feature_names = list(self.feature_names)
        elif "features" in request_data:
            vector = self._feature_vector(request_data["features"])
            prediction = self._predict_vectors([vector])[0]
            predictions = [{
                "region": request_data.get("region"),
                "feature_month": request_data.get("feature_month"),
                "target_month": request_data.get("target_month"),
                **prediction,
            }]
            feature_names = list(self.feature_names)
        else:
            raise APIError("Request must provide either rows or features")

        vectors_for_monitoring = vectors if "rows" in request_data else [vector]
        monitor_report = self._monitor_predictions(vectors_for_monitoring, predictions)
        timestamp = datetime.now(UTC).isoformat()
        request_hash = hashlib.sha256(canonical_json(request_data)).hexdigest()
        response_hash = hashlib.sha256(canonical_json({
            "api_version": API_VERSION,
            "model_version": self.model_version,
            "predictions": predictions,
            "monitoring": monitor_report,
        })).hexdigest()
        audit_record = self._append_audit({
            "timestamp": timestamp,
            "model_version": self.model_version,
            "feature_names": feature_names,
            "request_hash": request_hash,
            "response_hash": response_hash,
        })
        monitor_record = {
            "timestamp": timestamp,
            "request_id": audit_record["request_id"],
            "model_version": self.model_version,
            **monitor_report,
        }
        self._append_monitor(monitor_record)
        LOGGER.info(
            "Prediction served",
            extra={
                "event": "prediction_served",
                "request_id": audit_record["request_id"],
                "model_version": self.model_version,
                "prediction_count": len(predictions),
                "drifted_features": sum(
                    int(report["drifted"]) for report in monitor_report["feature_drift"].values()
                ),
                "drifted_predictions": sum(
                    int(report["drifted"])
                    for report in monitor_report["prediction_drift"].values()
                ),
                "request_hash": request_hash,
                "response_hash": response_hash,
            },
        )
        return {
            "api_version": API_VERSION,
            "model_version": self.model_version,
            "predictions": predictions,
            "monitoring": monitor_report,
            "audit": {
                "request_id": audit_record["request_id"],
                "timestamp": timestamp,
                "model_version": self.model_version,
                "feature_names": feature_names,
                "request_hash": request_hash,
                "response_hash": response_hash,
                "audit_hash": audit_record["audit_hash"],
            },
        }

    def _append_monitor(self, record):
        self.monitoring_path.parent.mkdir(parents=True, exist_ok=True)
        with self.monitor_lock, self.monitoring_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def monitoring(self, limit=100):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise APIError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise APIError("limit must be between 1 and 1000")
        records = []
        if self.monitoring_path.exists():
            with self.monitor_lock, self.monitoring_path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
        return {"api_version": API_VERSION, "records": records[-limit:]}

    def _append_audit(self, record):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_lock:
            previous_hash = ""
            if self.audit_path.exists():
                with self.audit_path.open("rb") as stream:
                    for line in stream:
                        if line.strip():
                            previous_hash = json.loads(line).get("audit_hash", "")
            entry = {
                "request_id": str(uuid.uuid4()),
                **record,
                "previous_hash": previous_hash,
            }
            entry["audit_hash"] = hashlib.sha256(canonical_json(entry)).hexdigest()
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return entry

    def audit(self, limit=100):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise APIError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise APIError("limit must be between 1 and 1000")
        records = []
        if self.audit_path.exists():
            with self.audit_lock, self.audit_path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
        return {"api_version": API_VERSION, "records": records[-limit:]}


def _resolve_path(value, root, default):
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def sign_registry(root=ROOT, artifact_key=None, registry_path=None):
    """Explicitly sign the current manifest in the persisted registry."""
    root = Path(root).resolve()
    key = artifact_key if artifact_key is not None else os.environ.get("AGRISHIELD_ARTIFACT_KEY")
    if not key:
        raise RuntimeError("Set AGRISHIELD_ARTIFACT_KEY before signing the registry")
    path = _resolve_path(registry_path, root, "model_registry.json")
    registry = json.loads(path.read_text(encoding="utf-8"))
    active_version = registry["active_model"]
    entry = registry["models"][active_version]
    manifest_path = path.parent / entry["manifest"]
    manifest_bytes = manifest_path.read_bytes()
    entry["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    entry["hmac_sha256"] = hmac.new(
        key.encode("utf-8"), manifest_bytes, hashlib.sha256
    ).hexdigest()
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class AgriShieldRequestHandler(SimpleHTTPRequestHandler):
    api = None
    dashboard_root = ROOT

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(directory or self.dashboard_root), **kwargs)

    def log_message(self, format_string, *args):
        LOGGER.info(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "status": int(args[1]) if len(args) > 1 and str(args[1]).isdigit() else None,
                "path": urlsplit(self.path).path,
            },
        )

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if path == "/v1/health":
            return self._send_json(200, self.api.health())
        if path == "/v1/models":
            return self._send_json(200, self.api.models())
        if path == "/v1/audit":
            limit = parse_qs(parsed.query).get("limit", ["100"])[0]
            try:
                return self._send_json(200, self.api.audit(limit))
            except APIError as error:
                return self._send_json(400, {"error": str(error)})
        if path == "/v1/monitoring":
            limit = parse_qs(parsed.query).get("limit", ["100"])[0]
            try:
                return self._send_json(200, self.api.monitoring(limit))
            except APIError as error:
                return self._send_json(400, {"error": str(error)})
        if path.startswith("/v1/"):
            return self._send_json(404, {"error": "Unknown API route"})
        self.path = path
        return super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != "/v1/predict":
            return self._send_json(404, {"error": "Unknown API route"})
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json(400, {"error": "Invalid Content-Length"})
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            return self._send_json(413, {"error": "Request body is empty or too large"})
        try:
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
            result = self.api.predict(request_data)
            return self._send_json(200, result)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            LOGGER.warning(
                "Prediction request rejected",
                extra={"event": "prediction_rejected", "error_type": type(error).__name__},
            )
            return self._send_json(400, {"error": str(error)})
        except Exception as error:
            LOGGER.exception(
                "Prediction request failed",
                extra={"event": "prediction_failed", "error_type": type(error).__name__},
            )
            return self._send_json(500, {"error": "Prediction could not be completed"})


def make_server(
    host="127.0.0.1",
    port=8000,
    artifact_key=None,
    audit_path=None,
    registry_path=None,
    dashboard_root=ROOT,
):
    dashboard_root = Path(dashboard_root).resolve()
    if not dashboard_root.is_dir():
        raise RuntimeError(f"Dashboard root does not exist or is not a directory: {dashboard_root}")
    api = AgriShieldAPI(
        root=dashboard_root,
        artifact_key=artifact_key,
        audit_path=audit_path,
        registry_path=registry_path,
    )
    handler = type(
        "ConfiguredAgriShieldRequestHandler",
        (AgriShieldRequestHandler,),
        {"api": api, "dashboard_root": dashboard_root},
    )
    return ThreadingHTTPServer((host, port), handler)


def _argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--artifact-key")
    parser.add_argument("--audit-path")
    parser.add_argument("--registry-path")
    parser.add_argument("--dashboard-root")
    parser.add_argument("--log-level", type=str.upper, choices=LOG_LEVELS)
    parser.add_argument("--sign-registry", action="store_true")
    return parser


def resolve_config(argv=None, environ=None):
    """Resolve CLI-over-environment-over-default server settings."""
    environ = os.environ if environ is None else environ
    parser = _argument_parser()
    args = parser.parse_args(argv)
    values = {
        "host": (
            args.host
            if args.host is not None
            else environ.get("AGRISHIELD_HOST", "127.0.0.1")
        ),
        "port": args.port if args.port is not None else environ.get("AGRISHIELD_PORT", 8000),
        "artifact_key": (
            args.artifact_key
            if args.artifact_key is not None
            else environ.get("AGRISHIELD_ARTIFACT_KEY")
        ),
        "audit_path": args.audit_path or environ.get("AGRISHIELD_AUDIT_PATH"),
        "registry_path": args.registry_path or environ.get("AGRISHIELD_REGISTRY_PATH"),
        "dashboard_root": (
            args.dashboard_root
            or environ.get("AGRISHIELD_DASHBOARD_ROOT")
            or ROOT
        ),
        "log_level": args.log_level or environ.get("AGRISHIELD_LOG_LEVEL") or "INFO",
        "sign_registry": args.sign_registry,
    }
    try:
        values["port"] = int(values["port"])
    except (TypeError, ValueError) as error:
        parser.error(f"AGRISHIELD_PORT must be an integer: {error}")
    if not 0 <= values["port"] <= 65535:
        parser.error("port must be between 0 and 65535")
    if not isinstance(values["host"], str) or not values["host"].strip():
        parser.error("host must be a non-empty address")
    values["log_level"] = str(values["log_level"]).upper()
    if values["log_level"] not in LOG_LEVELS:
        parser.error(f"log level must be one of: {', '.join(LOG_LEVELS)}")
    values["dashboard_root"] = Path(values["dashboard_root"]).resolve()
    values["audit_path"] = _resolve_path(
        values["audit_path"], values["dashboard_root"], "audit.jsonl"
    )
    values["registry_path"] = _resolve_path(
        values["registry_path"], values["dashboard_root"], "model_registry.json"
    )
    return values


def main(argv=None, environ=None):
    parser = _argument_parser()
    # Resolve once using the shared parser so CLI options override environment settings.
    config = resolve_config(argv, environ=environ)
    if config["artifact_key"] == "":
        parser.error("AGRISHIELD_ARTIFACT_KEY is set but empty")
    logging_level = getattr(logging, config["log_level"])
    LOGGER.setLevel(logging_level)
    get_json_logger("agrishield.registry").setLevel(logging_level)
    if config["sign_registry"]:
        try:
            sign_registry(
                root=config["dashboard_root"],
                registry_path=config["registry_path"],
                artifact_key=config["artifact_key"],
            )
        except Exception as error:
            print(f"AgriShield startup failed while signing registry: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        print(f"Signed active model manifest in {config['registry_path']}")
        return
    try:
        server = make_server(
            config["host"],
            config["port"],
            artifact_key=config["artifact_key"],
            audit_path=config["audit_path"],
            registry_path=config["registry_path"],
            dashboard_root=config["dashboard_root"],
        )
    except Exception as error:
        print(f"AgriShield startup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(
        f"AgriShield API v{API_VERSION} serving on "
        f"http://{config['host']}:{server.server_address[1]}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
