from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _ensure_train_colab_import_deps() -> None:
    if "lightgbm" not in sys.modules:
        fake_lgb = types.ModuleType("lightgbm")
        fake_lgb.Booster = object
        fake_lgb.Dataset = object
        sys.modules["lightgbm"] = fake_lgb
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")
    if "numerapi" not in sys.modules:
        fake_numerapi = types.ModuleType("numerapi")
        fake_numerapi.NumerAPI = object
        sys.modules["numerapi"] = fake_numerapi
    if "polars" not in sys.modules:
        fake_polars = types.ModuleType("polars")
        fake_polars.concat = lambda *_args, **_kwargs: object()
        sys.modules["polars"] = fake_polars


_ensure_train_colab_import_deps()
from training_runtime import load_train_valid_frames  # noqa: E402


class TrainColabLoadingTests(unittest.TestCase):
    def test_preserves_split_order_without_resorting(self) -> None:
        cfg = SimpleNamespace(
            dataset_version="v5.2",
            numerai_data_dir=Path("/tmp"),
            use_int8_parquet=True,
            load_mode="in_memory",
            id_col="id",
            era_col="era",
            target_col="target",
        )
        feature_cols = ["feature_0", "feature_1"]
        x_train = np.array([[1, 2], [3, 4]], dtype=np.int8)
        y_train = np.array([0.1, 0.2], dtype=np.float32)
        era_train = np.array(["era2", "era1"], dtype=object)
        id_train = np.array(["t1", "t2"], dtype=object)
        x_valid = np.array([[5, 6], [7, 8]], dtype=np.int8)
        y_valid = np.array([0.3, 0.4], dtype=np.float32)
        era_valid = np.array(["era4", "era3"], dtype=object)
        id_valid = np.array(["v1", "v2"], dtype=object)

        with (
            patch(
                "training_runtime.load_split_numpy",
                side_effect=[
                    (x_train, y_train, era_train, id_train),
                    (x_valid, y_valid, era_valid, id_valid),
                ],
            ),
            patch(
                "training_runtime.load_and_align_benchmarks",
                return_value=SimpleNamespace(
                    train=np.zeros((len(id_train), 1), dtype=np.float32),
                    valid=np.zeros((len(id_valid), 1), dtype=np.float32),
                    all=np.zeros((len(id_train) + len(id_valid), 1), dtype=np.float32),
                    cols=["bm"],
                ),
            ),
        ):
            data = load_train_valid_frames(
                cfg,
                Path("/tmp/train.parquet"),
                Path("/tmp/validation.parquet"),
                {"train": Path("/tmp/bt.parquet"), "validation": Path("/tmp/bv.parquet")},
                feature_cols,
            )

        np.testing.assert_array_equal(data.x_train, x_train)
        np.testing.assert_array_equal(data.x_valid, x_valid)
        np.testing.assert_array_equal(data.era_train, era_train)
        np.testing.assert_array_equal(data.era_valid, era_valid)
        np.testing.assert_array_equal(data.x_all, np.concatenate([x_train, x_valid], axis=0))


if __name__ == "__main__":
    unittest.main()
