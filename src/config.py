"""Runtime configuration parsing for training and inference entrypoints."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainRuntimeConfig:
    dataset_version: str = "v5.2"
    feature_set_name: str = "all"
    target_col: str = "target"
    era_col: str = "era"
    id_col: str = "id"
    model_name: str = "lgbm_numerai_v52"
    num_boost_round: int = 5000
    early_stopping_rounds: int = 300
    wandb_project: str = "numerai-mlops"
    wandb_entity: str | None = None
    wandb_api_key: str = ""
    numerai_public_id: str | None = None
    numerai_secret_key: str | None = None
    numerai_data_dir: Path = Path("/content/numerai_data")
    lgbm_device: str = "cpu"
    lgbm_learning_rate: float = 0.02
    lgbm_num_leaves: int = 128
    lgbm_feature_fraction: float = 0.7
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 1
    lgbm_min_data_in_leaf: int = 1000
    lgbm_seeds: tuple[int, ...] = (42, 1337, 2026)
    corr_scan_period: int = 100
    corr_scan_max_iters: int | None = None
    select_best_by: str = "corr"
    walkforward_enabled: bool = True
    walkforward_chunk_size: int = 156
    walkforward_purge_eras: int = 8
    walkforward_max_windows: int = 4
    walkforward_tune_seed: int | None = None
    walkforward_log_models: bool = False
    payout_weight_corr: float = 0.75
    payout_weight_bmc: float = 2.25
    blend_alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    bench_neutralize_prop_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    blend_tune_seed: int | None = None
    blend_use_windows: int | None = None
    max_features_per_model: int = 1200
    feature_sampling_strategy: str = "sharded_shuffle"
    feature_sampling_master_seed: int = 0
    use_int8_parquet: bool = True
    load_backend: str = "polars"
    load_mode: str = "in_memory"

    @classmethod
    def from_env(cls) -> "TrainRuntimeConfig":
        wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
        if not wandb_api_key:
            raise RuntimeError(
                "WANDB_API_KEY environment variable is not set. "
                "Please set it before running training to enable Weights & Biases logging."
            )

        lgbm_device = os.getenv("LGBM_DEVICE", "cpu").strip().lower() or "cpu"
        if lgbm_device not in {"gpu", "cpu"}:
            raise ValueError("Invalid LGBM_DEVICE. Expected one of: gpu, cpu.")

        numerai_public_id = os.getenv("NUMERAI_PUBLIC_ID", "").strip() or None
        numerai_secret_key = os.getenv("NUMERAI_SECRET_KEY", "").strip() or None
        if bool(numerai_public_id) != bool(numerai_secret_key):
            raise RuntimeError(
                "NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY must be set together for authenticated training downloads."
            )

        dataset_version = os.getenv("NUMERAI_DATASET_VERSION", "v5.2").strip() or "v5.2"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_version):
            raise ValueError("Invalid NUMERAI_DATASET_VERSION. Use only letters, numbers, dot, underscore, or hyphen.")
        if ".." in dataset_version or dataset_version.startswith((".", "-")):
            raise ValueError("Invalid NUMERAI_DATASET_VERSION.")

        seeds_str = os.getenv("LGBM_SEEDS", "42,1337,2026").strip()
        try:
            lgbm_seeds = tuple(int(seed.strip()) for seed in seeds_str.split(",") if seed.strip())
        except ValueError as exc:
            raise ValueError("Invalid LGBM_SEEDS. Expected comma-separated integers.") from exc
        if not lgbm_seeds:
            raise ValueError("Invalid LGBM_SEEDS. Provide at least one integer seed.")

        corr_scan_period = int(os.getenv("CORR_SCAN_PERIOD", "100"))
        if corr_scan_period <= 0:
            raise ValueError("Invalid CORR_SCAN_PERIOD. Expected positive integer.")

        corr_scan_max_iters_raw = os.getenv("CORR_SCAN_MAX_ITERS")
        corr_scan_max_iters: int | None = None
        if corr_scan_max_iters_raw is not None and corr_scan_max_iters_raw.strip():
            corr_scan_max_iters = int(corr_scan_max_iters_raw.strip())
            if corr_scan_max_iters <= 0:
                raise ValueError("Invalid CORR_SCAN_MAX_ITERS. Expected positive integer when set.")

        select_best_by = (os.getenv("SELECT_BEST_BY", "corr").strip().lower() or "corr")
        if select_best_by not in {"rmse", "corr"}:
            raise ValueError("Invalid SELECT_BEST_BY. Expected one of: rmse, corr.")

        walkforward_chunk_size = int(os.getenv("WALKFORWARD_CHUNK_SIZE", "156"))
        if walkforward_chunk_size <= 0:
            raise ValueError("Invalid WALKFORWARD_CHUNK_SIZE. Expected positive integer.")

        walkforward_purge_eras = int(os.getenv("WALKFORWARD_PURGE_ERAS", "8"))
        if walkforward_purge_eras < 0:
            raise ValueError("Invalid WALKFORWARD_PURGE_ERAS. Expected non-negative integer.")

        walkforward_max_windows = int(os.getenv("WALKFORWARD_MAX_WINDOWS", "4"))
        if walkforward_max_windows <= 0:
            raise ValueError("Invalid WALKFORWARD_MAX_WINDOWS. Expected positive integer.")

        walkforward_tune_seed_raw = os.getenv("WALKFORWARD_TUNE_SEED")
        walkforward_tune_seed = int(walkforward_tune_seed_raw.strip()) if walkforward_tune_seed_raw else lgbm_seeds[0]
        if walkforward_tune_seed not in lgbm_seeds:
            raise ValueError("Invalid WALKFORWARD_TUNE_SEED. Expected a value from LGBM_SEEDS.")

        blend_tune_seed_raw = os.getenv("BLEND_TUNE_SEED")
        blend_tune_seed = int(blend_tune_seed_raw.strip()) if blend_tune_seed_raw else walkforward_tune_seed
        if blend_tune_seed not in lgbm_seeds:
            raise ValueError("Invalid BLEND_TUNE_SEED. Expected a value from LGBM_SEEDS.")
        blend_use_windows = int(os.getenv("BLEND_USE_WINDOWS", str(walkforward_max_windows)))
        if blend_use_windows <= 0:
            raise ValueError("Invalid BLEND_USE_WINDOWS. Expected positive integer.")

        return cls(
            dataset_version=dataset_version,
            feature_set_name=os.getenv("NUMERAI_FEATURE_SET", "all"),
            id_col=os.getenv("NUMERAI_ID_COL", "id").strip() or "id",
            model_name=os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v52"),
            wandb_project=os.getenv("WANDB_PROJECT", "numerai-mlops"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_api_key=wandb_api_key,
            numerai_public_id=numerai_public_id,
            numerai_secret_key=numerai_secret_key,
            numerai_data_dir=Path(os.getenv("NUMERAI_DATA_DIR", "/content/numerai_data")),
            lgbm_device=lgbm_device,
            num_boost_round=int(os.getenv("LGBM_NUM_BOOST_ROUND", "5000")),
            early_stopping_rounds=int(os.getenv("LGBM_EARLY_STOPPING_ROUNDS", "300")),
            lgbm_learning_rate=float(os.getenv("LGBM_LEARNING_RATE", "0.02")),
            lgbm_num_leaves=int(os.getenv("LGBM_NUM_LEAVES", "128")),
            lgbm_feature_fraction=float(os.getenv("LGBM_FEATURE_FRACTION", "0.7")),
            lgbm_bagging_fraction=float(os.getenv("LGBM_BAGGING_FRACTION", "0.8")),
            lgbm_bagging_freq=int(os.getenv("LGBM_BAGGING_FREQ", "1")),
            lgbm_min_data_in_leaf=int(os.getenv("LGBM_MIN_DATA_IN_LEAF", "1000")),
            lgbm_seeds=lgbm_seeds,
            corr_scan_period=corr_scan_period,
            corr_scan_max_iters=corr_scan_max_iters,
            select_best_by=select_best_by,
            walkforward_enabled=_optional_bool_env("WALKFORWARD_ENABLED", default=True),
            walkforward_chunk_size=walkforward_chunk_size,
            walkforward_purge_eras=walkforward_purge_eras,
            walkforward_max_windows=walkforward_max_windows,
            walkforward_tune_seed=walkforward_tune_seed,
            walkforward_log_models=_optional_bool_env("WALKFORWARD_LOG_MODELS", default=False),
            payout_weight_corr=float(os.getenv("PAYOUT_WEIGHT_CORR", "0.75")),
            payout_weight_bmc=float(os.getenv("PAYOUT_WEIGHT_BMC", "2.25")),
            blend_alpha_grid=_optional_float_list_env(
                "BLEND_ALPHA_GRID",
                default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            ),
            bench_neutralize_prop_grid=_optional_float_list_env(
                "BENCH_NEUTRALIZE_PROP_GRID",
                default=(0.0, 0.25, 0.5, 0.75, 1.0),
            ),
            blend_tune_seed=blend_tune_seed,
            blend_use_windows=blend_use_windows,
            max_features_per_model=int(os.getenv("MAX_FEATURES_PER_MODEL", "1200")),
            feature_sampling_strategy=os.getenv("FEATURE_SAMPLING_STRATEGY", "sharded_shuffle").strip()
            or "sharded_shuffle",
            feature_sampling_master_seed=int(os.getenv("FEATURE_SAMPLING_MASTER_SEED", "0")),
            use_int8_parquet=_optional_bool_env("USE_INT8_PARQUET", default=True),
            load_backend=os.getenv("LOAD_BACKEND", "polars").strip().lower() or "polars",
            load_mode=os.getenv("LOAD_MODE", "in_memory").strip().lower() or "in_memory",
        )


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    numerai_public_id: str
    numerai_secret_key: str
    numerai_model_name: str
    wandb_entity: str
    wandb_project: str
    dataset_version: str = "v5.2"
    wandb_model_name: str = "lgbm_numerai_v52"
    min_pred_std: float = 1e-6
    max_abs_exposure: float = 0.30
    exposure_sample_rows: int = 200_000
    exposure_sample_seed: int = 0
    allow_dataset_version_mismatch: bool = False
    allow_features_by_model_missing: bool = False

    @classmethod
    def from_env(cls) -> "InferenceRuntimeConfig":
        min_pred_std_str = os.getenv("MIN_PRED_STD", "1e-6")
        try:
            min_pred_std = float(min_pred_std_str)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid MIN_PRED_STD value {min_pred_std_str!r}: must be a valid float"
            ) from exc

        max_abs_exposure_str = os.getenv("MAX_ABS_EXPOSURE", "0.30")
        try:
            max_abs_exposure = float(max_abs_exposure_str)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid MAX_ABS_EXPOSURE value {max_abs_exposure_str!r}: must be a valid float"
            ) from exc
        exposure_sample_rows = int(os.getenv("EXPOSURE_SAMPLE_ROWS", "200000"))
        if exposure_sample_rows <= 0:
            raise RuntimeError("Invalid EXPOSURE_SAMPLE_ROWS: must be a positive integer")

        return cls(
            numerai_public_id=_required_env("NUMERAI_PUBLIC_ID"),
            numerai_secret_key=_required_env("NUMERAI_SECRET_KEY"),
            numerai_model_name=_required_env("NUMERAI_MODEL_NAME"),
            wandb_entity=_required_env("WANDB_ENTITY"),
            wandb_project=_required_env("WANDB_PROJECT"),
            dataset_version=os.getenv("NUMERAI_DATASET_VERSION", "v5.2"),
            wandb_model_name=os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v52"),
            min_pred_std=min_pred_std,
            max_abs_exposure=max_abs_exposure,
            exposure_sample_rows=exposure_sample_rows,
            exposure_sample_seed=int(os.getenv("EXPOSURE_SAMPLE_SEED", "0")),
            allow_dataset_version_mismatch=_optional_bool_env("ALLOW_DATASET_VERSION_MISMATCH", default=False),
            allow_features_by_model_missing=_optional_bool_env("ALLOW_FEATURES_BY_MODEL_MISSING", default=False),
        )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid {name} value {value!r}: expected one of true/false, yes/no, on/off, 1/0.")


def _optional_float_list_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value {value!r}: expected comma-separated floats.") from exc
    if not parsed:
        raise RuntimeError(f"Invalid {name}: provide at least one float value.")
    return parsed
