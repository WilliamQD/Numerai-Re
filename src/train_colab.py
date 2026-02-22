"""Colab training entrypoint using NumerAI official NumerAPI dataset flow.

This script is designed for high-memory Colab Pro environments and trains a
LightGBM model on NumerAI v5.2 with W&B artifact tracking.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import wandb
from numerapi import NumerAPI

from config import TrainRuntimeConfig


FEATURES_FILENAME = "features.json"
MANIFEST_FILENAME = "train_manifest.json"
ARTIFACT_SCHEMA_VERSION = 2
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
    feature_cols: list[str]
    x_train: Any
    y_train: np.ndarray
    x_valid: Any
    y_valid: np.ndarray


@dataclass(frozen=True)
class FitResult:
    seed: int
    model: lgb.Booster
    best_iteration: int
    best_valid_rmse: float


@dataclass(frozen=True)
class SavedArtifact:
    model_paths: list[Path]
    features_path: Path
    manifest_path: Path


BASE_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 128,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 1000,
    "max_depth": -1,
    "verbosity": -1,
    "device": "cpu",
}


def _resolve_lgb_params(cfg: TrainRuntimeConfig) -> dict[str, object]:
    params = dict(BASE_LGB_PARAMS)
    params.update(
        {
            "learning_rate": cfg.lgbm_learning_rate,
            "num_leaves": cfg.lgbm_num_leaves,
            "feature_fraction": cfg.lgbm_feature_fraction,
            "bagging_fraction": cfg.lgbm_bagging_fraction,
            "bagging_freq": cfg.lgbm_bagging_freq,
            "min_data_in_leaf": cfg.lgbm_min_data_in_leaf,
            "device": cfg.lgbm_device,
        }
    )
    if cfg.lgbm_device == "gpu":
        params["gpu_use_dp"] = False
    else:
        params.pop("gpu_use_dp", None)
    return params


def _download_with_numerapi(cfg: TrainRuntimeConfig, data_dir: Path) -> tuple[Path, Path, Path]:
    version_data_dir = data_dir / cfg.dataset_version
    version_data_dir.mkdir(parents=True, exist_ok=True)
    numerapi_kwargs: dict[str, str] = {}
    if cfg.numerai_public_id and cfg.numerai_secret_key:
        numerapi_kwargs = {
            "public_id": cfg.numerai_public_id,
            "secret_key": cfg.numerai_secret_key,
        }
    napi = NumerAPI(**numerapi_kwargs)

    train_path = version_data_dir / "train.parquet"
    validation_path = version_data_dir / "validation.parquet"
    features_path = version_data_dir / FEATURES_FILENAME

    required_files = (
        (f"{cfg.dataset_version}/train.parquet", train_path),
        (f"{cfg.dataset_version}/validation.parquet", validation_path),
        (f"{cfg.dataset_version}/{FEATURES_FILENAME}", features_path),
    )
    for dataset_path, local_path in required_files:
        if local_path.exists():
            logger.info("phase=dataset_reused path=%s", local_path)
            continue
        logger.info("phase=dataset_downloading dataset=%s path=%s", dataset_path, local_path)
        napi.download_dataset(dataset_path, str(local_path))

    return train_path, validation_path, features_path


def _load_feature_list(features_path: Path, feature_set_name: str) -> list[str]:
    payload = json.loads(features_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON structure in features file {features_path}: expected object.")

    feature_sets = payload.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise ValueError(f'Missing or invalid "feature_sets" key in features file {features_path}.')

    if feature_set_name not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(f"Unknown feature set: {feature_set_name}. Available: {available}")

    feature_list = feature_sets[feature_set_name]
    if not isinstance(feature_list, list) or not all(isinstance(f, str) for f in feature_list):
        raise ValueError(f'Feature set "{feature_set_name}" in file {features_path} is invalid.')
    return feature_list


def _downcast_floats(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    exprs = []
    for col_name, dtype in schema.items():
        if dtype == pl.Float64:
            exprs.append(pl.col(col_name).cast(pl.Float32))
        else:
            exprs.append(pl.col(col_name))
    return lf.select(exprs)


def _load_frame(path: Path, selected_cols: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(str(path)).select(selected_cols)
    return _downcast_floats(lf).collect(streaming=True)


def init_wandb_run(cfg: TrainRuntimeConfig, lgb_params: dict[str, object]) -> Any:
    wandb.login()
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        job_type="train",
        tags=["numerai", cfg.dataset_version, "colab", "lightgbm", "ensemble"],
        config={
            "dataset_version": cfg.dataset_version,
            "feature_set": cfg.feature_set_name,
            "target_col": cfg.target_col,
            "era_col": cfg.era_col,
            "num_boost_round": cfg.num_boost_round,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "lgbm_seeds": list(cfg.lgbm_seeds),
            **lgb_params,
        },
    )


def load_train_valid_frames(cfg: TrainRuntimeConfig) -> LoadedData:
    train_path, validation_path, features_path = _download_with_numerapi(cfg, cfg.numerai_data_dir)
    feature_cols = _load_feature_list(features_path, cfg.feature_set_name)
    logger.info(
        "phase=datasets_downloaded dataset_version=%s data_dir=%s n_features=%d",
        cfg.dataset_version,
        cfg.numerai_data_dir,
        len(feature_cols),
    )

    selected_cols = feature_cols + [cfg.target_col, cfg.era_col]
    train_df = _load_frame(train_path, selected_cols)
    valid_df = _load_frame(validation_path, selected_cols)

    logger.info("phase=frame_loaded split=train rows=%d cols=%d", train_df.height, train_df.width)
    logger.info("phase=frame_loaded split=validation rows=%d cols=%d", valid_df.height, valid_df.width)

    x_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.get_column(cfg.target_col).to_numpy().astype(np.float32)
    x_valid = valid_df.select(feature_cols).to_pandas()
    y_valid = valid_df.get_column(cfg.target_col).to_numpy().astype(np.float32)

    del train_df, valid_df
    gc.collect()

    return LoadedData(feature_cols=feature_cols, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)


def fit_lgbm(cfg: TrainRuntimeConfig, lgb_params: dict[str, object], data: LoadedData, seed: int) -> FitResult:
    dtrain = lgb.Dataset(data.x_train, label=data.y_train, feature_name=data.feature_cols)
    dvalid = lgb.Dataset(data.x_valid, label=data.y_valid, reference=dtrain, feature_name=data.feature_cols)

    fit_params = dict(lgb_params)
    fit_params["seed"] = seed
    callbacks = [
        lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(100),
    ]
    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks.append(lgb.record_evaluation(evals_result))

    try:
        model = lgb.train(
            params=fit_params,
            train_set=dtrain,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
    except lgb.basic.LightGBMError as exc:
        if fit_params.get("device") != "gpu" or "No OpenCL device found" not in str(exc):
            raise
        logger.warning("phase=lgbm_gpu_unavailable requested_device=gpu fallback_device=cpu reason=%s", exc)
        fit_params["device"] = "cpu"
        fit_params.pop("gpu_use_dp", None)
        evals_result = {}
        callbacks = [
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result),
        ]
        model = lgb.train(
            params=fit_params,
            train_set=dtrain,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        lgb_params["device"] = "cpu"
        lgb_params.pop("gpu_use_dp", None)

    best_iter = int(model.best_iteration)
    best_rmse = float(evals_result["valid"]["rmse"][max(0, best_iter - 1)])
    logger.info("phase=model_trained seed=%d best_iteration=%d best_valid_rmse=%.6f", seed, best_iter, best_rmse)
    return FitResult(seed=seed, model=model, best_iteration=best_iter, best_valid_rmse=best_rmse)


def save_and_log_artifact(
    cfg: TrainRuntimeConfig,
    run: Any,
    lgb_params: dict[str, object],
    data: LoadedData,
    fit_results: list[FitResult],
) -> SavedArtifact:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    model_paths: list[Path] = []
    for fit_result in fit_results:
        model_path = out_dir / f"{cfg.model_name}_seed{fit_result.seed}.txt"
        fit_result.model.save_model(str(model_path))
        model_paths.append(model_path)

    features_out = out_dir / FEATURES_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME
    features_out.write_text(json.dumps(data.feature_cols, indent=2))

    manifest_out.write_text(
        json.dumps(
            {
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_version": cfg.dataset_version,
                "feature_set": cfg.feature_set_name,
                "target_col": cfg.target_col,
                "era_col": cfg.era_col,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "ensemble_strategy": "mean",
                "seeds": list(cfg.lgbm_seeds),
                "members": [
                    {
                        "seed": result.seed,
                        "model_file": f"{cfg.model_name}_seed{result.seed}.txt",
                        "best_iteration": result.best_iteration,
                        "best_valid_rmse": result.best_valid_rmse,
                    }
                    for result in fit_results
                ],
                "best_iteration_mean": float(np.mean([result.best_iteration for result in fit_results])),
                "best_valid_rmse_mean": float(np.mean([result.best_valid_rmse for result in fit_results])),
                "n_features": len(data.feature_cols),
                "model_name": cfg.model_name,
                "model_file": f"{cfg.model_name}_seed{fit_results[0].seed}.txt",
                "model_files": [path.name for path in model_paths],
                "lgb_params": lgb_params,
            },
            indent=2,
        )
    )

    artifact = wandb.Artifact(name=cfg.model_name, type="model")
    for model_path in model_paths:
        artifact.add_file(str(model_path), name=model_path.name)
    artifact.add_file(str(features_out), name=FEATURES_FILENAME)
    artifact.add_file(str(manifest_out), name=MANIFEST_FILENAME)
    run.log_artifact(artifact, aliases=["latest", "prod"])

    logger.info("phase=artifact_uploaded artifact_name=%s aliases=latest,prod model_count=%d", cfg.model_name, len(model_paths))
    return SavedArtifact(model_paths=model_paths, features_path=features_out, manifest_path=manifest_out)


def train() -> None:
    cfg = TrainRuntimeConfig.from_env()
    logger.info(
        "phase=config_loaded dataset_version=%s feature_set=%s model_name=%s lgbm_device=%s seeds=%s",
        cfg.dataset_version,
        cfg.feature_set_name,
        cfg.model_name,
        cfg.lgbm_device,
        list(cfg.lgbm_seeds),
    )

    lgb_params = _resolve_lgb_params(cfg)
    run = init_wandb_run(cfg, lgb_params)
    data = load_train_valid_frames(cfg)

    fit_results = [fit_lgbm(cfg, lgb_params, data, seed) for seed in cfg.lgbm_seeds]
    wandb.log(
        {
            "best_iteration_mean": float(np.mean([result.best_iteration for result in fit_results])),
            "best_valid_rmse_mean": float(np.mean([result.best_valid_rmse for result in fit_results])),
            "n_models": len(fit_results),
        }
    )

    save_and_log_artifact(cfg, run, lgb_params, data, fit_results)
    run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train()
