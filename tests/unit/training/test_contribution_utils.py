from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerai_re.data.bench_matrix_builder import align_bench_to_ids
from numerai_re.metrics.numerai_metrics import bmc_mean_per_era
from numerai_re.training.tune_blend import blended_predictions


class ContributionUtilsTests(unittest.TestCase):
    def test_align_bench_to_ids_orders_rows(self) -> None:
        ids = np.array(["a", "b", "c"])
        bench_df = pl.DataFrame(
            {
                "id": ["c", "a", "b"],
                "bench_1": [0.3, 0.1, 0.2],
                "bench_2": [1.3, 1.1, 1.2],
            }
        )
        bench, cols = align_bench_to_ids(ids, bench_df, "id")
        self.assertEqual(cols, ["bench_1", "bench_2"])
        np.testing.assert_allclose(bench[:, 0], np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_align_bench_to_ids_raises_on_missing_ids(self) -> None:
        with self.assertRaises(RuntimeError):
            align_bench_to_ids(
                np.array(["a", "b"]),
                pl.DataFrame({"id": ["a"], "bench_1": [0.1]}),
                "id",
            )

    def test_bmc_and_blend_predictions_are_finite(self) -> None:
        pred = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        target = np.array([0.4, 0.2, 0.3, 0.1], dtype=np.float32)
        era = np.array(["era1", "era1", "era2", "era2"])
        bench = np.array(
            [
                [0.2, 0.3],
                [0.1, 0.5],
                [0.4, 0.6],
                [0.3, 0.2],
            ],
            dtype=np.float32,
        )
        blended = blended_predictions(pred, era, bench, alpha=0.5, neutralize_prop=0.5)
        bmc = bmc_mean_per_era(blended, target, era, bench, neutralize_prop=0.5)
        self.assertEqual(blended.shape, pred.shape)
        self.assertTrue(np.isfinite(blended).all())
        self.assertTrue(np.isfinite(bmc))


if __name__ == "__main__":
    unittest.main()
