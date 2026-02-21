"""Colab training entrypoint using NumerAI official NumerAPI dataset flow.

This script is designed for high-memory Colab Pro environments and trains a
LightGBM model on NumerAI v4.3 with W&B artifact tracking.
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
ARTIFACT_SCHEMA_VERSION = 1
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    dataset_version: str = "v5.2"
    feature_set_name: str = "medium"
    target_col: str = "target"
    era_col: str = "era"
    model_name: str = "lgbm_numerai_v52"
    num_boost_round: int = 5000
    early_stopping_rounds: int = 250


@dataclass(frozen=True)
class LoadedData:
    feature_cols: list[str]
    x_train: Any
    y_train: np.ndarray
    x_valid: Any
    y_valid: np.ndarray


@dataclass(frozen=True)
class FitResult:
    model: lgb.Booster
    best_iteration: int
    best_valid_rmse: float


@dataclass(frozen=True)
class SavedArtifact:
    model_path: Path
    features_path: Path
    manifest_path: Path


BASE_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 192,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "min_data_in_leaf": 600,
    "max_depth": -1,
    "seed": 42,
    "verbosity": -1,
    "device": "gpu",
    "gpu_use_dp": False,
}


def _resolve_lgb_params(cfg: TrainRuntimeConfig) -> dict[str, object]:
    params = dict(BASE_LGB_PARAMS)
    params["device"] = cfg.lgbm_device
    if cfg.lgbm_device != "gpu":
        params.pop("gpu_use_dp", None)
    return params


def _download_with_numerapi(cfg: TrainRuntimeConfig, data_dir: Path) -> tuple[Path, Path, Path]:
    """Download train/validation/features files, matching official example flow."""
    data_dir.mkdir(parents=True, exist_ok=True)
    napi = NumerAPI()

    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    features_path = data_dir / FEATURES_FILENAME

    napi.download_dataset(f"{cfg.dataset_version}/train.parquet", str(train_path))
    napi.download_dataset(f"{cfg.dataset_version}/validation.parquet", str(validation_path))
    napi.download_dataset(f"{cfg.dataset_version}/{FEATURES_FILENAME}", str(features_path))

    return train_path, validation_path, features_path


def _load_feature_list(features_path: Path, feature_set_name: str) -> list[str]:
    try:
        payload = json.loads(features_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in features file {features_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Unexpected JSON structure in features file {features_path}: expected a JSON object at the top level."
        )

    feature_sets = payload.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise ValueError(
            f'Missing or invalid "feature_sets" key in features file {features_path}: '
            "expected an object mapping names to lists of features."
        )

    if feature_set_name not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(f"Unknown feature set: {feature_set_name}. Available: {available}")

    feature_list = feature_sets[feature_set_name]
    if not isinstance(feature_list, list) or not all(isinstance(f, str) for f in feature_list):
        raise ValueError(
            f'Feature set "{feature_set_name}" in file {features_path} is not a list of feature name strings.'
        )
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
    lf = _downcast_floats(lf)
    return lf.collect(streaming=True)


def init_wandb_run(cfg: TrainRuntimeConfig, lgb_params: dict[str, object]) -> Any:
    try:
        wandb.login()
    except Exception as exc:
        raise RuntimeError(
            "Failed to authenticate with Weights & Biases. "
            "Check that WANDB_API_KEY is set correctly and has not expired."
        ) from exc

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        job_type="train",
        tags=["numerai", cfg.dataset_version, "colab", "lightgbm"],
        config={
            "dataset_version": cfg.dataset_version,
            "feature_set": cfg.feature_set_name,
            "target_col": cfg.target_col,
            "era_col": cfg.era_col,
            "num_boost_round": cfg.num_boost_round,
            "early_stopping_rounds": cfg.early_stopping_rounds,
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
    logger.info(
        "phase=frame_loaded split=train rows=%d cols=%d",
        train_df.height,
        train_df.width,
    )
    logger.info(
        "phase=frame_loaded split=validation rows=%d cols=%d",
        valid_df.height,
        valid_df.width,
    )

    x_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.get_column(cfg.target_col).to_numpy().astype(np.float32)
    x_valid = valid_df.select(feature_cols).to_pandas()
    y_valid = valid_df.get_column(cfg.target_col).to_numpy().astype(np.float32)

    del train_df, valid_df
    gc.collect()

    return LoadedData(
        feature_cols=feature_cols,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )


def fit_lgbm(cfg: TrainRuntimeConfig, lgb_params: dict[str, object], data: LoadedData) -> FitResult:
    dtrain = lgb.Dataset(data.x_train, label=data.y_train, feature_name=data.feature_cols)
    dvalid = lgb.Dataset(data.x_valid, label=data.y_valid, reference=dtrain, feature_name=data.feature_cols)

    evals_result: dict[str, dict[str, list[float]]] = {}
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result),
        ],
    )

    best_iter = int(model.best_iteration)
    best_rmse_index = max(0, best_iter - 1)
    best_rmse = float(evals_result["valid"]["rmse"][best_rmse_index])
    logger.info(
        "phase=model_trained best_iteration=%d best_valid_rmse=%.6f n_features=%d train_rows=%d valid_rows=%d",
        best_iter,
        best_rmse,
        len(data.feature_cols),
        len(data.y_train),
        len(data.y_valid),
    )

    return FitResult(
        model=model,
        best_iteration=best_iter,
        best_valid_rmse=best_rmse,
    )


def save_and_log_artifact(
    cfg: TrainRuntimeConfig,
    run: Any,
    lgb_params: dict[str, object],
    data: LoadedData,
    fit_result: FitResult,
) -> SavedArtifact:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / f"{cfg.model_name}.txt"
    features_out = out_dir / FEATURES_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME

    fit_result.model.save_model(str(model_path))
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
                "best_iteration": fit_result.best_iteration,
                "best_valid_rmse": fit_result.best_valid_rmse,
                "n_features": len(data.feature_cols),
                "model_name": cfg.model_name,
                "model_file": f"{cfg.model_name}.txt",
                "lgb_params": lgb_params,
            },
            indent=2,
        )
    )

    artifact = wandb.Artifact(
        name=cfg.model_name,
        type="model",
        metadata={
            "dataset_version": cfg.dataset_version,
            "feature_set": cfg.feature_set_name,
            "best_iteration": fit_result.best_iteration,
            "best_valid_rmse": fit_result.best_valid_rmse,
        },
    )
    artifact.add_file(str(model_path), name=f"{cfg.model_name}.txt")
    artifact.add_file(str(features_out), name=FEATURES_FILENAME)
    artifact.add_file(str(manifest_out), name=MANIFEST_FILENAME)
    run.log_artifact(artifact, aliases=["latest", "prod"])
    logger.info(
        "phase=artifact_uploaded artifact_name=%s aliases=%s model_file=%s",
        cfg.model_name,
        "latest,prod",
        model_path.name,
    )

    return SavedArtifact(model_path=model_path, features_path=features_out, manifest_path=manifest_out)


def train() -> None:
    cfg = TrainRuntimeConfig.from_env()
    logger.info(
        "phase=config_loaded dataset_version=%s feature_set=%s model_name=%s lgbm_device=%s",
        cfg.dataset_version,
        cfg.feature_set_name,
        cfg.model_name,
        cfg.lgbm_device,
    )

    lgb_params = _resolve_lgb_params(cfg)
    run = init_wandb_run(cfg, lgb_params)
    data = load_train_valid_frames(cfg)
    fit_result = fit_lgbm(cfg, lgb_params, data)
    wandb.log(
        {
            "best_iteration": fit_result.best_iteration,
            "best_valid_rmse": fit_result.best_valid_rmse,
        }
    )
    save_and_log_artifact(cfg, run, lgb_params, data, fit_result)
    run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train()
