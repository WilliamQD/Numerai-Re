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


@dataclass(frozen=True)
class TrainConfig:
    dataset_version: str = "v4.3"
    feature_set_name: str = "medium"
    target_col: str = "target"
    era_col: str = "era"
    model_name: str = "lgbm_numerai_v43"
    num_boost_round: int = 5000
    early_stopping_rounds: int = 250


LGB_PARAMS = {
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


def _download_with_numerapi(cfg: TrainConfig, data_dir: Path) -> tuple[Path, Path, Path]:
    """Download train/validation/features files, matching official example flow."""
    data_dir.mkdir(parents=True, exist_ok=True)
    napi = NumerAPI()

    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    features_path = data_dir / "features.json"

    napi.download_dataset(f"{cfg.dataset_version}/train.parquet", str(train_path))
    napi.download_dataset(f"{cfg.dataset_version}/validation.parquet", str(validation_path))
    napi.download_dataset(f"{cfg.dataset_version}/features.json", str(features_path))

    return train_path, validation_path, features_path


def _load_feature_list(features_path: Path, feature_set_name: str) -> list[str]:
    payload = json.loads(features_path.read_text())
    feature_sets = payload["feature_sets"]
    if feature_set_name not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(f"Unknown feature set: {feature_set_name}. Available: {available}")
    return feature_sets[feature_set_name]


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

    wandb.login()
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
            **LGB_PARAMS,
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
        params=LGB_PARAMS,
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
    best_rmse = float(evals_result["valid"]["rmse"][best_iter - 1])
    wandb.log({"best_iteration": best_iter, "best_valid_rmse": best_rmse})

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / f"{cfg.model_name}.txt"
    features_out = out_dir / "features.json"
    manifest_out = out_dir / "train_manifest.json"

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
                "best_iteration": best_iter,
                "best_valid_rmse": best_rmse,
                "n_features": len(feature_cols),
                "model_name": cfg.model_name,
                "lgb_params": LGB_PARAMS,
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
    artifact.add_file(str(features_out), name="features.json")
    artifact.add_file(str(manifest_out), name="train_manifest.json")
    run.log_artifact(artifact, aliases=["latest", "prod"])
    run.finish()


if __name__ == "__main__":
    train()
