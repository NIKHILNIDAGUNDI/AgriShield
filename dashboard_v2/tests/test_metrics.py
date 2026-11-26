import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_shield_ml import compare_regression_metrics


class RegressionMetricsTests(unittest.TestCase):
    def test_reports_rmse_r2_and_relative_rmse_improvement(self):
        metrics = compare_regression_metrics(
            [1, 2, 3],
            [2, 2, 2],
            [1, 2, 2],
        )

        self.assertAlmostEqual(metrics["baseline_rmse"], (2 / 3) ** 0.5)
        self.assertEqual(metrics["baseline_r2"], 0.0)
        self.assertAlmostEqual(metrics["candidate_rmse"], (1 / 3) ** 0.5)
        self.assertAlmostEqual(metrics["candidate_r2"], 0.5)
        self.assertAlmostEqual(metrics["improvement"], (1 - 1 / (2 ** 0.5)) * 100)


if __name__ == "__main__":
    unittest.main()
