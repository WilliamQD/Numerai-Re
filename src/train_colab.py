"""Colab training entrypoint using NumerAI official NumerAPI dataset flow.

This script is designed for high-memory Colab Pro environments and trains a
LightGBM model on NumerAI v4.3 with W&B artifact tracking.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import wandb
from numerapi import NumerAPI


FEATURES_FILENAME = "features.json"
MANIFEST_FILENAME = "train_manifest.json"
ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    dataset_version: str = "v4.3"
    feature_set_name: str = "medium"
    target_col: str = "target"
    era_col: str = "era"
    model_name: str = "lgbm_numerai_v43"
    num_boost_round: int = 5000
    early_stopping_rounds: int = 250


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


def _resolve_lgb_params() -> dict[str, object]:
    params = dict(BASE_LGB_PARAMS)
    device = os.getenv("LGBM_DEVICE", "gpu").strip().lower() or "gpu"
    if device not in {"gpu", "cpu"}:
        raise ValueError("Invalid LGBM_DEVICE. Expected one of: gpu, cpu.")
    params["device"] = device
    if device != "gpu":
        params.pop("gpu_use_dp", None)
    return params


def _download_with_numerapi(cfg: TrainConfig, data_dir: Path) -> tuple[Path, Path, Path]:
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


def train() -> None:
    cfg = TrainConfig(
        feature_set_name=os.getenv("NUMERAI_FEATURE_SET", "medium"),
        model_name=os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v43"),
    )

    lgb_params = _resolve_lgb_params()

    wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
    if not wandb_api_key:
        raise RuntimeError(
            "WANDB_API_KEY environment variable is not set. "
            "Please set it before running training to enable Weights & Biases logging."
        )

    try:
        wandb.login()
    except Exception as exc:
        raise RuntimeError(
            "Failed to authenticate with Weights & Biases. "
            "Check that WANDB_API_KEY is set correctly and has not expired."
        ) from exc
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "numerai-mlops"),
        entity=os.getenv("WANDB_ENTITY"),
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

    data_dir = Path(os.getenv("NUMERAI_DATA_DIR", "/content/numerai_data"))
    train_path, validation_path, features_path = _download_with_numerapi(cfg, data_dir)
    feature_cols = _load_feature_list(features_path, cfg.feature_set_name)

    selected_cols = feature_cols + [cfg.target_col, cfg.era_col]
    train_df = _load_frame(train_path, selected_cols)
    valid_df = _load_frame(validation_path, selected_cols)

    x_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.get_column(cfg.target_col).to_numpy().astype(np.float32)

    x_valid = valid_df.select(feature_cols).to_pandas()
    y_valid = valid_df.get_column(cfg.target_col).to_numpy().astype(np.float32)

    del train_df, valid_df
    gc.collect()

    dtrain = lgb.Dataset(x_train, label=y_train, feature_name=feature_cols)
    dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=feature_cols)

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
    wandb.log({"best_iteration": best_iter, "best_valid_rmse": best_rmse})

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / f"{cfg.model_name}.txt"
    features_out = out_dir / FEATURES_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME

    model.save_model(str(model_path))
    features_out.write_text(json.dumps(feature_cols, indent=2))
    manifest_out.write_text(
        json.dumps(
            {
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_version": cfg.dataset_version,
                "feature_set": cfg.feature_set_name,
                "target_col": cfg.target_col,
                "era_col": cfg.era_col,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "best_iteration": best_iter,
                "best_valid_rmse": best_rmse,
                "n_features": len(feature_cols),
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
            "best_iteration": best_iter,
            "best_valid_rmse": best_rmse,
        },
    )
    artifact.add_file(str(model_path), name=f"{cfg.model_name}.txt")
    artifact.add_file(str(features_out), name=FEATURES_FILENAME)
    artifact.add_file(str(manifest_out), name=MANIFEST_FILENAME)
    run.log_artifact(artifact, aliases=["latest", "prod"])
    run.finish()


if __name__ == "__main__":
    train()
