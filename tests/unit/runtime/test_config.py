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
