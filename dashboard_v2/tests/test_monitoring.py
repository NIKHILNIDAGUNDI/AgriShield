import json
import logging
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_shield_ml.monitoring import (
    build_monitoring_baseline,
    monitor_population,
    prediction_interval,
)
from agri_shield_ml.observability import JsonFormatter


class MonitoringTests(unittest.TestCase):
    def test_population_monitoring_detects_location_and_distribution_drift(self):
        baseline = build_monitoring_baseline(
            np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            ["feature"],
        )["feature_population"]["feature"]

        stable = monitor_population([0.0, 1.0, 2.0, 3.0, 4.0], baseline)
        shifted = monitor_population([30.0, 31.0, 32.0], baseline)

        self.assertFalse(stable["drifted"])
        self.assertTrue(shifted["drifted"])
        self.assertGreater(shifted["mean_z_score"], 3.0)
        self.assertIsNotNone(shifted["psi"])

    def test_validation_errors_produce_conformal_interval_width_and_bounds(self):
        baseline = build_monitoring_baseline(
            [[1.0], [2.0]], ["feature"],
            price_errors=[1.0, -2.0, 3.0, -4.0, 5.0],
            risk_errors=[2.0, -3.0, 4.0, -5.0, 6.0],
        )
        self.assertEqual(baseline["prediction_intervals"]["price"], 5.0)
        price_interval = prediction_interval(100, 5, coverage=0.9)
        risk_interval = prediction_interval(2, 5, bounds=(0, 100))

        self.assertEqual((price_interval["lower"], price_interval["upper"]), (95, 105))
        self.assertEqual((risk_interval["lower"], risk_interval["upper"]), (0, 7))

    def test_structured_logging_is_valid_json_with_event_fields(self):
        record = logging.LogRecord(
            "agrishield.test", logging.INFO, __file__, 1, "prediction served", (), None
        )
        record.event = "prediction_served"
        record.model_version = "1.0.0"

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["event"], "prediction_served")
        self.assertEqual(payload["model_version"], "1.0.0")
        self.assertEqual(payload["message"], "prediction served")


if __name__ == "__main__":
    unittest.main()
