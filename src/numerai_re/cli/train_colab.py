"""Colab training entrypoint using NumerAI official NumerAPI dataset flow."""

from __future__ import annotations

import logging
import os
import time

from typing import Final

from numerai_re.runtime.config import _optional_bool_env


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

BALANCED_PROFILE_DEFAULTS: Final[dict[str, str]] = {
    "NUMERAI_FEATURE_SET": "small",
    "LOAD_MODE": "cached",
    "WALKFORWARD_MAX_WINDOWS": "2",
    "BLEND_USE_WINDOWS": "2",
    "LGBM_SEEDS": "42,1337",
    "LGBM_NUM_BOOST_ROUND": "2500",
    "WALKFORWARD_NUM_BOOST_ROUND": "1800",
    "LGBM_EARLY_STOPPING_ROUNDS": "200",
    "WALKFORWARD_EARLY_STOPPING_ROUNDS": "200",
    "BLEND_ALPHA_GRID": "0.0,0.1,0.2,0.3,0.4,0.5",
    "BENCH_NEUTRALIZE_PROP_GRID": "0.0,0.2,0.4",
}

EFFECTIVE_KNOBS: Final[tuple[str, ...]] = (
    "TRAIN_PROFILE",
    "NUMERAI_FEATURE_SET",
    "LOAD_MODE",
    "WALKFORWARD_MAX_WINDOWS",
    "BLEND_USE_WINDOWS",
    "LGBM_SEEDS",
    "LGBM_NUM_BOOST_ROUND",
    "WALKFORWARD_NUM_BOOST_ROUND",
    "LGBM_EARLY_STOPPING_ROUNDS",
    "WALKFORWARD_EARLY_STOPPING_ROUNDS",
    "BLEND_ALPHA_GRID",
    "BENCH_NEUTRALIZE_PROP_GRID",
    "LGBM_DEVICE",
)


def _apply_train_profile() -> None:
    profile = os.getenv("TRAIN_PROFILE", "full").strip().lower() or "full"
    os.environ["TRAIN_PROFILE"] = profile
    if profile == "balanced":
        for key, value in BALANCED_PROFILE_DEFAULTS.items():
            os.environ.setdefault(key, value)
        logging.getLogger(__name__).info("phase=colab_profile_applied profile=balanced")
    elif profile == "full":
        logging.getLogger(__name__).info("phase=colab_profile_applied profile=full")
    else:
        logging.getLogger(__name__).warning("phase=colab_profile_unknown profile=%s", profile)


def _probe_lgbm_gpu_if_requested() -> None:
    logger = logging.getLogger(__name__)
    if not _optional_bool_env("RUN_LGBM_GPU_PROBE", default=False):
        return
    try:
        import numpy as np
        import lightgbm as lgb
    except Exception as exc:
        logger.warning("phase=lgbm_probe_import_failed reason=%s", exc)
        return

    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(12000, 120)).astype(np.float32)
    y_train = rng.normal(size=(12000,)).astype(np.float32)
    x_valid = rng.normal(size=(3000, 120)).astype(np.float32)
    y_valid = rng.normal(size=(3000,)).astype(np.float32)
    train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data, free_raw_data=False)
    base_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    def _run_probe(device: str) -> float:
        params = dict(base_params)
        params["device"] = device
        start = time.perf_counter()
        lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        return time.perf_counter() - start

    cpu_elapsed = _run_probe("cpu")
    logger.info("phase=lgbm_probe_cpu_ok elapsed_seconds=%.2f", cpu_elapsed)
    try:
        gpu_elapsed = _run_probe("gpu")
    except Exception as exc:
        os.environ["LGBM_DEVICE"] = "cpu"
        logger.warning("phase=lgbm_probe_gpu_failed reason=%s resolved_device=cpu", exc)
        return
    os.environ["LGBM_DEVICE"] = "gpu"
    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float("inf")
    logger.info(
        "phase=lgbm_probe_gpu_ok elapsed_seconds=%.2f speedup_vs_cpu=%.2f resolved_device=gpu",
        gpu_elapsed,
        speedup,
    )


def _log_effective_knobs() -> None:
    logger = logging.getLogger(__name__)
    fields = " ".join(f"{key}={os.getenv(key, '')}" for key in EFFECTIVE_KNOBS)
    logger.info("phase=colab_effective_knobs %s", fields)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if _optional_bool_env("TRAIN_DRY_RUN", default=False):
        from numerai_re.training.training_dry_run import train_dry_run

        train_dry_run()
    else:
        _apply_train_profile()
        _probe_lgbm_gpu_if_requested()
        _log_effective_knobs()
        from numerai_re.training.training_pipeline import train

        train()
