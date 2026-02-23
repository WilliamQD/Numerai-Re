from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from numerai_re.data.data_loading import load_split_numpy, _cache_manifest_path, _cache_valid


def _write_test_parquet(path: Path, n_rows: int = 20, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    feature_cols = [f"feature_{i}" for i in range(6)]
    data = {col: rng.integers(0, 5, n_rows).tolist() for col in feature_cols}
    data["era"] = [f"era{i // 4 + 1}" for i in range(n_rows)]
    data["id"] = [f"row_{i:04d}" for i in range(n_rows)]
    data["target"] = rng.random(n_rows).tolist()
    # Sprinkle a few null targets to exercise filtering
    for idx in [2, 7]:
        data["target"][idx] = None
    pl.DataFrame(data).write_parquet(str(path))
    return feature_cols


class LoadSplitNumpyTests(unittest.TestCase):
    def test_float32_hint_preserves_parquet_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.parquet"
            features = _write_test_parquet(path)
            x, y, era, row_id = load_split_numpy(
                path, features, "id", "era", "target", feature_dtype=np.float32
            )
            self.assertTrue(np.issubdtype(x.dtype, np.integer))
            self.assertEqual(y.dtype, np.float32)
            # 2 null targets were inserted → 20 - 2 = 18 rows
            self.assertEqual(x.shape, (18, len(features)))

    def test_int8_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.parquet"
            features = _write_test_parquet(path)
            x, y, era, row_id = load_split_numpy(
                path, features, "id", "era", "target", feature_dtype=np.int8
            )
            self.assertEqual(x.dtype, np.int8)
            self.assertEqual(x.shape[1], len(features))

    def test_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.parquet"
            features = _write_test_parquet(path)

            # First call: populates cache
            x1, y1, era1, id1 = load_split_numpy(
                path, features, "id", "era", "target", feature_dtype=np.float32, use_cache=True
            )
            manifest_path = _cache_manifest_path(path, features, np.float32)
            self.assertTrue(_cache_valid(path, manifest_path))

            # Second call: loads from cache
            x2, y2, era2, id2 = load_split_numpy(
                path, features, "id", "era", "target", feature_dtype=np.float32, use_cache=True
            )
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(era1, era2)
            np.testing.assert_array_equal(id1, id2)

    def test_cache_invalidated_on_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.parquet"
            features = _write_test_parquet(path, seed=0)
            load_split_numpy(
                path, features, "id", "era", "target", feature_dtype=np.float32, use_cache=True
            )
            # Overwrite with different data and force mtime well beyond tolerance
            _write_test_parquet(path, seed=99)
            mtime = os.path.getmtime(path)
            os.utime(path, (mtime + 5, mtime + 5))
            manifest_path = _cache_manifest_path(path, features, np.float32)
            self.assertFalse(_cache_valid(path, manifest_path))

    def test_no_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.parquet"
            features = _write_test_parquet(path)
            load_split_numpy(path, features, "id", "era", "target")
            manifest_path = _cache_manifest_path(path, features, np.float32)
            self.assertFalse(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
