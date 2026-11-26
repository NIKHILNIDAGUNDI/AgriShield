import hashlib
import hmac
import json
import math
import sys
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_server import AgriShieldAPI, APIError, make_server, resolve_config

ROOT = Path(__file__).resolve().parents[1]


def sample_rows():
    rows = []
    for month in range(1, 7):
        rows.append({
            "region": "Test Region",
            "year_month": f"2025-{month:02d}",
            "temperature": 15 + month,
            "precipitation": 50 + month,
            "crop_yield": 90 + month,
            "fuel_price": 60 + month,
            "transport_cost": 1.2,
            "inflation_rate": 3.1,
            "geopolitical_risk": 0.1,
            "news_sentiment": 0.2,
            "food_price_index": 120 + month,
            "shortage_risk_score": 10 + month,
        })
    return rows


class APIServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_audit_path = ROOT / "tests" / ".class-audit.jsonl"
        cls.class_monitoring_path = ROOT / "tests" / "monitoring.jsonl"
        cls.class_audit_path.unlink(missing_ok=True)
        cls.class_monitoring_path.unlink(missing_ok=True)
        cls.api = AgriShieldAPI(
            root=ROOT, artifact_key="", audit_path=cls.class_audit_path
        )

    @classmethod
    def tearDownClass(cls):
        cls.class_audit_path.unlink(missing_ok=True)
        cls.class_monitoring_path.unlink(missing_ok=True)

    def setUp(self):
        self.audit_path = ROOT / "tests" / ".audit-test.jsonl"
        self.monitoring_path = ROOT / "tests" / ".monitoring-test.jsonl"
        self.api.audit_path = self.audit_path
        self.api.monitoring_path = self.monitoring_path
        self.audit_path.unlink(missing_ok=True)
        self.monitoring_path.unlink(missing_ok=True)

    def tearDown(self):
        self.audit_path.unlink(missing_ok=True)
        self.monitoring_path.unlink(missing_ok=True)

    def test_startup_config_uses_cli_over_environment_over_defaults(self):
        config = resolve_config(
            [
                "--host", "0.0.0.0",
                "--port", "8123",
                "--audit-path", "custom-audit.jsonl",
                "--log-level", "DEBUG",
            ],
            environ={
                "AGRISHIELD_HOST": "127.0.0.2",
                "AGRISHIELD_PORT": "9000",
                "AGRISHIELD_ARTIFACT_KEY": "test-key",
                "AGRISHIELD_AUDIT_PATH": "env-audit.jsonl",
                "AGRISHIELD_REGISTRY_PATH": "registry/registry.json",
                "AGRISHIELD_DASHBOARD_ROOT": str(ROOT),
                "AGRISHIELD_LOG_LEVEL": "warning",
            },
        )
        self.assertEqual(config["host"], "0.0.0.0")
        self.assertEqual(config["port"], 8123)
        self.assertEqual(config["artifact_key"], "test-key")
        self.assertEqual(config["audit_path"], (ROOT / "custom-audit.jsonl").resolve())
        self.assertEqual(
            config["registry_path"], (ROOT / "registry" / "registry.json").resolve()
        )
        self.assertEqual(config["dashboard_root"], ROOT.resolve())
        self.assertEqual(config["log_level"], "DEBUG")

        defaults = resolve_config([], environ={})
        self.assertEqual(defaults["host"], "127.0.0.1")
        self.assertEqual(defaults["port"], 8000)
        self.assertEqual(defaults["log_level"], "INFO")

    def test_startup_config_rejects_invalid_port(self):
        with self.assertRaises(SystemExit):
            resolve_config([], environ={"AGRISHIELD_PORT": "not-a-port"})
        with self.assertRaises(SystemExit):
            resolve_config(["--port", "65536"], environ={})

    def test_health_endpoint_reports_loaded_model(self):
        server = make_server(
            host="127.0.0.1",
            port=0,
            artifact_key="",
            audit_path=self.audit_path,
            dashboard_root=ROOT,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            with urlopen(
                f"{base_url}/v1/health", timeout=5
            ) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["api_version"], "1.0.0")
            self.assertEqual(health["active_model_version"], "1.0.0")

            api = server.RequestHandlerClass.api
            original_predict = api.predict

            def fail_prediction(_request):
                raise RuntimeError("internal detail")

            api.predict = fail_prediction
            request = Request(
                f"{base_url}/v1/predict",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self.assertRaises(HTTPError) as failure:
                    urlopen(request, timeout=5)
                self.assertEqual(failure.exception.code, 500)
                body = json.loads(failure.exception.read().decode("utf-8"))
                self.assertEqual(body["error"], "Prediction could not be completed")
                failure.exception.close()
            finally:
                api.predict = original_predict
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_registry_digest_and_keyed_signature_verification(self):
        registry = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))
        manifest_path = ROOT / registry["models"][registry["active_model"]]["manifest"]
        manifest = manifest_path.read_bytes()
        entry = registry["models"][registry["active_model"]]
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), entry["manifest_sha256"])
        mismatched = json.loads(json.dumps(registry))
        mismatched["models"][mismatched["active_model"]]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.api._read_verified_manifest(mismatched)

        key = "test-signing-key"
        entry["hmac_sha256"] = hmac.new(key.encode(), manifest, hashlib.sha256).hexdigest()
        old_key = self.api.artifact_key
        old_require_hmac = self.api.require_hmac
        try:
            self.api.artifact_key = key
            self.api.require_hmac = True
            _, _, verification = self.api._read_verified_manifest(registry)
            self.assertEqual(verification, "hmac-sha256")
            self.api.artifact_key = "incorrect-key"
            with self.assertRaisesRegex(RuntimeError, "HMAC-SHA256"):
                self.api._read_verified_manifest(registry)
        finally:
            self.api.artifact_key = old_key
            self.api.require_hmac = old_require_hmac

    def test_predict_returns_versioned_audit_metadata_and_persists_record(self):
        result = self.api.predict({"rows": sample_rows(), "forecast_only": True})
        self.assertEqual(result["api_version"], "1.0.0")
        self.assertEqual(result["model_version"], "1.0.0")
        self.assertEqual([row["target_month"] for row in result["predictions"]], [
            "2025-07", "2025-08", "2025-09"
        ])
        self.assertTrue(math.isfinite(result["predictions"][0]["predicted_price"]))
        interval = result["predictions"][0]["price_interval"]
        self.assertLessEqual(interval["lower"], result["predictions"][0]["predicted_price"])
        self.assertGreaterEqual(interval["upper"], result["predictions"][0]["predicted_price"])
        self.assertEqual(interval["coverage"], 0.9)
        self.assertTrue(result["monitoring"]["feature_drift"])
        self.assertTrue(result["monitoring"]["baseline_available"])
        self.assertEqual(result["audit"]["model_version"], "1.0.0")
        self.assertIn("temperature_lag_1", result["audit"]["feature_names"])
        self.assertEqual(len(result["audit"]["request_hash"]), 64)
        self.assertEqual(len(result["audit"]["response_hash"]), 64)
        self.assertTrue(result["audit"]["timestamp"].endswith("+00:00"))
        self.api.predict({"rows": sample_rows(), "forecast_only": True})
        audit = self.api.audit(10)["records"]
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["audit_hash"], result["audit"]["audit_hash"])
        self.assertEqual(audit[0]["previous_hash"], "")
        self.assertEqual(audit[1]["previous_hash"], audit[0]["audit_hash"])
        monitoring = self.api.monitoring(10)["records"]
        self.assertEqual(len(monitoring), 2)
        self.assertEqual(monitoring[0]["request_id"], audit[0]["request_id"])

    def test_explicit_feature_prediction_requires_exact_manifest_schema(self):
        features = {name: 1.0 for name in self.api.feature_names}
        result = self.api.predict({"features": features, "region": "Test"})
        self.assertEqual(len(result["predictions"]), 1)
        self.assertTrue(math.isfinite(result["predictions"][0]["predicted_price"]))
        with self.assertRaisesRegex(APIError, "Feature names"):
            self.api.predict({"features": {"temperature_lag_1": 1.0}})

    def test_prediction_monitoring_compares_outputs_to_registry_baseline(self):
        manifest = self.api.model_bundle["manifest"]
        original = deepcopy(manifest.get("monitoring"))
        manifest["monitoring"] = {
            "feature_population": {},
            "prediction_population": {
                "price": {"mean": -1000.0, "std": 1.0},
                "risk": {"mean": -1000.0, "std": 1.0},
            },
        }
        try:
            result = self.api.predict({"rows": sample_rows(), "forecast_only": True})
        finally:
            if original is None:
                manifest.pop("monitoring", None)
            else:
                manifest["monitoring"] = original

        self.assertTrue(result["monitoring"]["prediction_drift"]["price"]["drifted"])
        self.assertTrue(result["monitoring"]["prediction_drift"]["risk"]["drifted"])

    def test_invalid_and_non_finite_feature_values_are_rejected(self):
        invalid = sample_rows()
        invalid[0]["temperature"] = float("nan")
        with self.assertRaisesRegex(APIError, "finite number"):
            self.api.predict({"rows": invalid})

        with self.assertRaisesRegex(APIError, "(?i)at least four"):
            self.api.predict({"rows": sample_rows()[:3]})


if __name__ == "__main__":
    unittest.main()
