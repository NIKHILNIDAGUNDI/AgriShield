import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_registry import activate_model, list_models, register_model, rollback_model

ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryRollbackTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ".rollback-fixture"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()
        for version in ("1.0.0", "0.9.0"):
            weights_name = f"{version}.pth"
            weights_bytes = f"weights-{version}".encode()
            (self.root / weights_name).write_bytes(weights_bytes)
            (self.root / f"{version}.manifest.json").write_text(
                json.dumps({
                    "schema": "agri-shield-ml-model",
                    "weights": {
                        "file": weights_name,
                        "sha256": hashlib.sha256(weights_bytes).hexdigest(),
                    },
                }), encoding="utf-8"
            )
        models = {}
        for version in ("1.0.0", "0.9.0"):
            manifest_name = f"{version}.manifest.json"
            models[version] = {
                "version": version,
                "manifest": manifest_name,
                "manifest_sha256": hashlib.sha256(
                    (self.root / manifest_name).read_bytes()
                ).hexdigest(),
            }
        registry = {
            "schema": "agri-shield-model-registry",
            "registry_version": 1,
            "active_model": "1.0.0",
            "models": models,
        }
        (self.root / "model_registry.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_rollback_verifies_target_and_persists_history(self):
        event = rollback_model(self.root, "0.9.0", reason="bad validation")
        registry = list_models(self.root)

        self.assertEqual(event["from"], "1.0.0")
        self.assertEqual(event["to"], "0.9.0")
        self.assertEqual(registry["active_model"], "0.9.0")
        self.assertEqual(registry["rollback_metadata"]["last_rollback"]["reason"], "bad validation")

    def test_rollback_rejects_bad_digest_and_active_model(self):
        with self.assertRaisesRegex(ValueError, "already active"):
            rollback_model(self.root, "1.0.0")
        with self.assertRaisesRegex(ValueError, "digest"):
            registry_path = self.root / "model_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["models"]["0.9.0"]["manifest_sha256"] = "0" * 64
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            rollback_model(self.root, "0.9.0")

    def test_registration_validates_artifact_and_can_activate_candidate(self):
        weights = b"candidate weights"
        (self.root / "candidate.pth").write_bytes(weights)
        manifest_path = self.root / "candidate.manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": "agri-shield-ml-model",
            "weights": {
                "file": "candidate.pth",
                "sha256": hashlib.sha256(weights).hexdigest(),
            },
        }), encoding="utf-8")

        registered = register_model(self.root, "candidate-2", manifest_path.name)
        activated = activate_model(self.root, "candidate-2", reason="candidate approved")

        self.assertFalse(registered["activated"])
        self.assertEqual(activated["from"], "1.0.0")
        self.assertEqual(list_models(self.root)["active_model"], "candidate-2")


if __name__ == "__main__":
    unittest.main()
