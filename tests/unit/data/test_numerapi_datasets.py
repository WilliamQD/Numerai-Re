from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerai_re.data.numerapi_datasets import (
    pick_benchmark_models_parquet,
    resolve_split_parquet,
    resolve_split_parquet_with_report,
)


class NumerapiDatasetResolverTests(unittest.TestCase):
    def test_resolve_split_prefers_int8_when_enabled(self) -> None:
        datasets = ["v5.2/live.parquet", "v5.2/live_int8.parquet"]
        selected = resolve_split_parquet(datasets, "v5.2", ("live",), use_int8=True)
        self.assertEqual(selected, "v5.2/live_int8.parquet")

    def test_resolve_split_falls_back_to_default_name(self) -> None:
        datasets: list[str] = []
        selected = resolve_split_parquet(datasets, "v5.2", ("validation",), use_int8=False)
        self.assertEqual(selected, "v5.2/validation.parquet")

    def test_resolve_split_report_exposes_int8_absence(self) -> None:
        datasets = ["v5.2/train.parquet"]
        report = resolve_split_parquet_with_report(datasets, "v5.2", ("train",), use_int8=True)
        self.assertEqual(report.selected, "v5.2/train.parquet")
        self.assertFalse(report.selected_is_int8)
        self.assertEqual(report.int8_candidates, ())
        self.assertEqual(report.non_int8_candidates, ("v5.2/train.parquet",))

    def test_pick_benchmark_models_prefers_exact_name(self) -> None:
        datasets = [
            "v5.2/train_benchmark_models_alt.parquet",
            "v5.2/train_benchmark_models.parquet",
        ]
        selected = pick_benchmark_models_parquet(datasets, "v5.2", "train")
        self.assertEqual(selected, "v5.2/train_benchmark_models.parquet")

    def test_pick_benchmark_models_raises_on_ambiguity(self) -> None:
        datasets = [
            "v5.2/train_benchmark_models_a.parquet",
            "v5.2/train_benchmark_models_b.parquet",
        ]
        with self.assertRaises(RuntimeError):
            pick_benchmark_models_parquet(datasets, "v5.2", "train")


if __name__ == "__main__":
    unittest.main()
