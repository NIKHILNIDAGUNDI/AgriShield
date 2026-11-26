import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_shield_ml import (
    FEATURE_SOURCES,
    align_features,
    build_feature_names,
    prepare_forecasting_lags,
)


class ForecastingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            {
                "region": region,
                "year_month": month,
                **{
                    source: offset + (int(month[-2:]) - 1) * 10
                    for offset, source in enumerate(FEATURE_SOURCES)
                },
            }
            for month in ("2024-04", "2024-01", "2024-03", "2024-02", "2024-05")
            for region in ("South", "North")
        ])

    def test_lags_sort_by_month_and_never_cross_region_boundaries(self):
        lagged = prepare_forecasting_lags(self.data, lag_months=3)

        self.assertEqual(len(lagged), 2 * (5 - 3))
        for region, rows in lagged.groupby("region"):
            self.assertEqual(rows["year_month"].tolist(), ["2024-04", "2024-05"])
            for _, row in rows.iterrows():
                month_index = int(row["year_month"][-2:]) - 1
                for source_index, source in enumerate(FEATURE_SOURCES):
                    base = source_index + 10 * month_index
                    self.assertEqual(row[f"{source}_lag_1"], base - 10)
                    self.assertEqual(row[f"{source}_lag_2"], base - 20)
                    self.assertEqual(row[f"{source}_lag_3"], base - 30)
                self.assertEqual(row["region"], region)

    def test_feature_alignment_preserves_manifest_order_end_to_end(self):
        lagged = prepare_forecasting_lags(self.data, lag_months=3)
        expected_names = build_feature_names(lag_months=3)
        reversed_names = list(reversed(expected_names))
        aligned = align_features(lagged, reversed_names)

        self.assertEqual(aligned.columns.tolist(), reversed_names)
        self.assertEqual(aligned.iloc[0].tolist(), lagged.loc[0, reversed_names].tolist())

    def test_rejects_missing_or_duplicate_features(self):
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            align_features(pd.DataFrame({"known": [1]}), ["missing"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            align_features(pd.DataFrame({"known": [1]}), ["known", "known"])

    def test_rejects_invalid_lag_or_missing_source_columns(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            prepare_forecasting_lags(self.data, lag_months=0)
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            prepare_forecasting_lags(self.data.drop(columns=["temperature"]))


if __name__ == "__main__":
    unittest.main()
