from __future__ import annotations

import hashlib
import random


def sample_features_for_seed(
    feature_pool: list[str],
    seed: int,
    model_index: int,
    n_models: int,
    max_features_per_model: int,
    master_seed: int = 0,
    strategy: str = "sharded_shuffle",
) -> list[str]:
    if max_features_per_model <= 0 or max_features_per_model >= len(feature_pool):
        return list(feature_pool)
    if strategy != "sharded_shuffle":
        raise ValueError(f"Unsupported FEATURE_SAMPLING_STRATEGY: {strategy}")
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    if model_index < 0 or model_index >= n_models:
        raise ValueError("model_index must be in [0, n_models)")

    shuffled = list(feature_pool)
    random.Random(master_seed).shuffle(shuffled)
    shard_size = max(1, len(shuffled) // n_models)
    start = model_index * shard_size
    end = len(shuffled) if model_index == n_models - 1 else min(len(shuffled), start + shard_size)
    selected = list(shuffled[start:end])
    if len(selected) < max_features_per_model:
        remaining = [col for col in shuffled if col not in set(selected)]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: max_features_per_model - len(selected)])
    return sorted(selected[:max_features_per_model])


def features_hash(feature_cols: list[str]) -> str:
    payload = "\n".join(feature_cols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
