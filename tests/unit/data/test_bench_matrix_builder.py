from __future__ import annotations

import sys
import unittest
from pathlib import Path
import types

import numpy as np
try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None
    fake_polars = types.ModuleType("polars")
    fake_polars.DataFrame = object
    fake_polars.Utf8 = object
    sys.modules["polars"] = fake_polars

HAS_REAL_POLARS = pl is not None and hasattr(pl, "col") and hasattr(pl, "DataFrame")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerai_re.data.bench_matrix_builder import BenchmarkAlignmentError, align_bench_to_ids


class BenchMatrixBuilderTests(unittest.TestCase):
    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_casts_id_types_before_join(self) -> None:
        ids = np.array([1, 2], dtype=np.int64)
        bench = pl.DataFrame({"id": ["1", "2"], "benchmark_1": [0.1, 0.2]})
        matrix, cols = align_bench_to_ids(ids, bench, "id")
        self.assertEqual(cols, ["benchmark_1"])
        self.assertEqual(matrix.shape, (2, 1))

    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_reports_missing_ids(self) -> None:
        ids = np.array(["a", "b"], dtype=object)
        bench = pl.DataFrame({"id": ["a"], "benchmark_1": [0.1]})
        with self.assertRaises(BenchmarkAlignmentError) as ctx:
            align_bench_to_ids(ids, bench, "id")
        message = str(ctx.exception)
        self.assertIn("missing_count=1", message)
        self.assertIn("b", message)

    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_rejects_duplicate_ids(self) -> None:
        ids = np.array(["a"], dtype=object)
        bench = pl.DataFrame({"id": ["a", "a"], "benchmark_1": [0.1, 0.2]})
        with self.assertRaises(BenchmarkAlignmentError) as ctx:
            align_bench_to_ids(ids, bench, "id")
        self.assertIn("duplicate_count=1", str(ctx.exception))

    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_drops_sparse_columns_by_default(self) -> None:
        ids = np.array(["a", "b", "c"], dtype=object)
        bench = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "benchmark_dense": [0.1, 0.2, 0.3],
                "benchmark_sparse": [0.4, None, 0.6],
            }
        )
        matrix, cols = align_bench_to_ids(ids, bench, "id")
        self.assertEqual(cols, ["benchmark_dense"])
        self.assertEqual(matrix.shape, (3, 1))

    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_raises_when_sparse_columns_not_dropped(self) -> None:
        ids = np.array(["a", "b", "c"], dtype=object)
        bench = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "benchmark_dense": [0.1, 0.2, 0.3],
                "benchmark_sparse": [0.4, None, 0.6],
            }
        )
        with self.assertRaises(BenchmarkAlignmentError) as ctx:
            align_bench_to_ids(ids, bench, "id", drop_sparse_columns=False)
        self.assertIn("contain nulls after sparse-column filtering", str(ctx.exception))

    @unittest.skipIf(not HAS_REAL_POLARS, "real polars is not installed")
    def test_align_enforces_min_columns_after_filtering(self) -> None:
        ids = np.array(["a", "b"], dtype=object)
        bench = pl.DataFrame({"id": ["a", "b"], "benchmark_sparse": [None, 0.2]})
        with self.assertRaises(BenchmarkAlignmentError) as ctx:
            align_bench_to_ids(ids, bench, "id", min_benchmark_columns=1)
        self.assertIn("removed too many columns", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
