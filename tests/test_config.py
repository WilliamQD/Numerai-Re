from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import TrainRuntimeConfig


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


if __name__ == "__main__":
    unittest.main()
