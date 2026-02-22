from __future__ import annotations

import unittest

from src.feature_sampling import sample_features_for_seed


class FeatureSamplingTests(unittest.TestCase):
    def test_sharded_shuffle_is_deterministic(self) -> None:
        pool = [f"feature_{i:04d}" for i in range(50)]
        first = sample_features_for_seed(pool, seed=42, model_index=0, n_models=3, max_features_per_model=12, master_seed=7)
        second = sample_features_for_seed(pool, seed=42, model_index=0, n_models=3, max_features_per_model=12, master_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_disable_sampling_when_max_non_positive(self) -> None:
        pool = [f"f{i}" for i in range(8)]
        sampled = sample_features_for_seed(pool, seed=1, model_index=0, n_models=2, max_features_per_model=0, master_seed=9)
        self.assertEqual(sampled, pool)


if __name__ == "__main__":
    unittest.main()
