from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import InferenceRuntimeConfig, TrainRuntimeConfig


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


class InferenceRuntimeConfigTests(unittest.TestCase):
    def test_allow_features_by_model_missing_default_false(self) -> None:
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
        self.assertFalse(cfg.allow_features_by_model_missing)


if __name__ == "__main__":
    unittest.main()
