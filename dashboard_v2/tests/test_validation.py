import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_shield_ml import grouped_walk_forward_splits, rolling_origin_splits


class GroupedWalkForwardSplitsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            {"year_month": month, "region": region}
            for month in ("2024-04", "2024-01", "2024-03", "2024-02")
            for region in ("South", "North")
        ])

    def test_splits_keep_all_regions_together_and_preserve_time_order(self):
        folds = grouped_walk_forward_splits(
            self.data,
            "year_month",
            "region",
            n_splits=2,
            test_size=1,
            gap=1,
        )

        self.assertEqual(len(folds), 2)
        self.assertLess(len(folds[0][0]), len(folds[1][0]))
        for train_indices, test_indices in folds:
            train = self.data.iloc[train_indices]
            test = self.data.iloc[test_indices]
            self.assertEqual(test.groupby("year_month")["region"].nunique().tolist(), [2])
            region_counts = train.groupby("year_month")["region"].nunique().tolist()
            self.assertEqual(region_counts, [2] * train["year_month"].nunique())
            self.assertEqual(
                train.assign(_month=pd.to_datetime(train["year_month"]))
                .sort_values(["_month", "region"])["year_month"].tolist(),
                train["year_month"].tolist(),
            )
            self.assertLess(train["year_month"].max(), test["year_month"].min())
            self.assertEqual(
                (pd.Period(test["year_month"].min(), freq="M")
                 - pd.Period(train["year_month"].max(), freq="M")).n,
                2,
            )

    def test_rolling_origin_uses_fixed_train_window(self):
        folds = rolling_origin_splits(
            self.data,
            "year_month",
            "region",
            n_splits=2,
            test_size=1,
            train_size=1,
        )

        self.assertTrue(all(len(train_indices) == 2 for train_indices, _ in folds))
        self.assertEqual(
            self.data.iloc[folds[0][0]]["year_month"].unique().tolist(),
            ["2024-02"],
        )

    def test_rejects_insufficient_history_and_missing_train_size(self):
        with self.assertRaisesRegex(ValueError, "Not enough distinct timestamps"):
            grouped_walk_forward_splits(
                self.data.iloc[:2], "year_month", "region", n_splits=2, test_size=1
            )
        with self.assertRaisesRegex(ValueError, "train_size is required"):
            rolling_origin_splits(self.data, "year_month", "region")


if __name__ == "__main__":
    unittest.main()
