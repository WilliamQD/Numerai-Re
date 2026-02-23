from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from numerai_re.runtime.config import InferenceRuntimeConfig, TrainRuntimeConfig


class TrainRuntimeConfigTests(unittest.TestCase):
    def test_feature_set_defaults_to_all(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.feature_set_name, "all")

    def test_feature_set_env_override_is_respected(self) -> None:
        with patch.dict(
            os.environ,
            {"WANDB_API_KEY": "dummy-key", "NUMERAI_FEATURE_SET": "medium"},
            clear=True,
        ):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.feature_set_name, "medium")

    def test_feature_sampling_defaults_are_set(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.max_features_per_model, 1200)
        self.assertEqual(cfg.feature_sampling_strategy, "sharded_shuffle")
        self.assertEqual(cfg.feature_sampling_master_seed, 0)

    def test_use_int8_parquet_defaults_to_true(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertTrue(cfg.use_int8_parquet)

    def test_use_int8_parquet_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key", "USE_INT8_PARQUET": "false"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertFalse(cfg.use_int8_parquet)

    def test_status_update_seconds_defaults_to_60(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.status_update_seconds, 60)

    def test_status_update_seconds_must_be_positive(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key", "STATUS_UPDATE_SECONDS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                TrainRuntimeConfig.from_env()

    def test_walkforward_num_boost_round_defaults_to_lgbm_num_boost_round(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key", "LGBM_NUM_BOOST_ROUND": "1234"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.num_boost_round, 1234)
        self.assertEqual(cfg.walkforward_num_boost_round, 1234)

    def test_walkforward_num_boost_round_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "dummy-key",
                "LGBM_NUM_BOOST_ROUND": "1500",
                "WALKFORWARD_NUM_BOOST_ROUND": "600",
            },
            clear=True,
        ):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.num_boost_round, 1500)
        self.assertEqual(cfg.walkforward_num_boost_round, 600)

    def test_load_mode_accepts_cached(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key", "LOAD_MODE": "cached"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertEqual(cfg.load_mode, "cached")

    def test_load_mode_rejects_unknown_value(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key", "LOAD_MODE": "cache"}, clear=True):
            with self.assertRaises(ValueError):
                TrainRuntimeConfig.from_env()

    def test_benchmark_sparse_filter_defaults(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "dummy-key"}, clear=True):
            cfg = TrainRuntimeConfig.from_env()
        self.assertTrue(cfg.bench_drop_sparse_columns)
        self.assertEqual(cfg.bench_max_null_ratio_per_column, 0.0)
        self.assertEqual(cfg.bench_min_columns, 1)
        self.assertEqual(cfg.bench_min_covered_rows_per_window, 512)
        self.assertEqual(cfg.bench_min_covered_eras_per_window, 8)

    def test_benchmark_sparse_filter_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "dummy-key",
                "BENCH_DROP_SPARSE_COLUMNS": "false",
                "BENCH_MAX_NULL_RATIO_PER_COLUMN": "0.2",
                "BENCH_MIN_COLUMNS": "3",
                "BENCH_MIN_COVERED_ROWS_PER_WINDOW": "1024",
                "BENCH_MIN_COVERED_ERAS_PER_WINDOW": "12",
            },
            clear=True,
        ):
            cfg = TrainRuntimeConfig.from_env()
        self.assertFalse(cfg.bench_drop_sparse_columns)
        self.assertEqual(cfg.bench_max_null_ratio_per_column, 0.2)
        self.assertEqual(cfg.bench_min_columns, 3)
        self.assertEqual(cfg.bench_min_covered_rows_per_window, 1024)
        self.assertEqual(cfg.bench_min_covered_eras_per_window, 12)


class InferenceRuntimeConfigTests(unittest.TestCase):
    def test_inference_config_loads_required_fields(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NUMERAI_PUBLIC_ID": "pid",
                "NUMERAI_SECRET_KEY": "sid",
                "NUMERAI_MODEL_NAME": "model",
                "WANDB_ENTITY": "entity",
                "WANDB_PROJECT": "project",
            },
            clear=True,
        ):
            cfg = InferenceRuntimeConfig.from_env()
        self.assertEqual(cfg.numerai_public_id, "pid")
        self.assertEqual(cfg.numerai_secret_key, "sid")
        self.assertEqual(cfg.numerai_model_name, "model")
        self.assertEqual(cfg.wandb_entity, "entity")
        self.assertEqual(cfg.wandb_project, "project")


if __name__ == "__main__":
    unittest.main()
