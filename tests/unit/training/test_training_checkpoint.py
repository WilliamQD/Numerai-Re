from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys_path_added = False
try:
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        sys_path_added = True
except Exception:
    pass

from numerai_re.training.training_checkpoint import load_training_checkpoint, write_training_checkpoint


class TrainingCheckpointTests(unittest.TestCase):
    def test_roundtrip_checkpoint_payload(self) -> None:
        cfg = SimpleNamespace(
            dataset_version="v5.2",
            feature_set_name="medium",
            lgbm_seeds=(1,),
            max_features_per_model=100,
            feature_sampling_strategy="random",
            feature_sampling_master_seed=7,
        )
        members = [
            {
                "seed": 1,
                "model_file": "m1.txt",
                "best_iteration": 10,
                "best_valid_rmse": 0.1,
                "best_valid_corr": 0.02,
                "corr_scan_period": 50,
                "features_key": "m1.txt",
                "n_features_used": 20,
                "features_hash": "abc",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "training_checkpoint.json"
            write_training_checkpoint(
                checkpoint_path,
                cfg,
                {"learning_rate": 0.01},
                members,
                walkforward={"enabled": False},
                postprocess={"schema_version": 1},
            )
            loaded = load_training_checkpoint(
                checkpoint_path,
                cfg,
                {"learning_rate": 0.01},
                expected_walkforward={"enabled": False},
                expected_postprocess={"schema_version": 1},
            )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["seed"], 1)
        self.assertEqual(loaded[0]["model_file"], "m1.txt")

    def test_checkpoint_mismatch_raises(self) -> None:
        cfg = SimpleNamespace(
            dataset_version="v5.2",
            feature_set_name="medium",
            lgbm_seeds=(1,),
            max_features_per_model=100,
            feature_sampling_strategy="random",
            feature_sampling_master_seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "training_checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "dataset_version": "v5.1",
                        "feature_set": "medium",
                        "seeds": [1],
                        "lgb_params": {"learning_rate": 0.01},
                        "max_features_per_model": 100,
                        "feature_sampling_strategy": "random",
                        "feature_sampling_master_seed": 7,
                        "members": [],
                    }
                )
            )
            with self.assertRaises(RuntimeError):
                load_training_checkpoint(checkpoint_path, cfg, {"learning_rate": 0.01})


if __name__ == "__main__":
    unittest.main()
