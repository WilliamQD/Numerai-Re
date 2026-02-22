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
from era_utils import era_to_int
from numerai_metrics import mean_per_era_numerai_corr
from walkforward import build_windows


FEATURES_FILENAME = "features.json"
MANIFEST_FILENAME = "train_manifest.json"
CHECKPOINT_FILENAME = "training_checkpoint.json"
ARTIFACT_SCHEMA_VERSION = 2
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
    feature_cols: list[str]
    x_train: Any
    y_train: np.ndarray
    era_train: np.ndarray
    x_valid: Any
    y_valid: np.ndarray
    era_valid: np.ndarray
    x_all: Any
    y_all: np.ndarray
    era_all: np.ndarray
    era_all_int: np.ndarray


@dataclass(frozen=True)
class FitResult:
    seed: int
    model: lgb.Booster
    best_iteration: int
    best_valid_rmse: float
    best_valid_corr: float
    train_rmse_curve: list[float]
    valid_rmse_curve: list[float]
    valid_corr_curve: list[float]
    corr_scan_iters: list[int]


@dataclass(frozen=True)
class SavedArtifact:
    model_paths: list[Path]
    features_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class WalkforwardReport:
    recommended_num_iteration: int
    mean_corr: float
    std_corr: float
    sharpe: float
    hit_rate: float
    windows: list[dict[str, float | int]]


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


def _checkpoint_dir(cfg: TrainRuntimeConfig) -> Path:
    return cfg.numerai_data_dir / cfg.dataset_version / "checkpoints" / cfg.model_name


def _write_training_checkpoint(
    checkpoint_path: Path,
    cfg: TrainRuntimeConfig,
    lgb_params: dict[str, object],
    members: list[dict[str, object]],
    walkforward: dict[str, object] | None = None,
) -> None:
    payload = {
        "dataset_version": cfg.dataset_version,
        "feature_set": cfg.feature_set_name,
        "seeds": list(cfg.lgbm_seeds),
        "lgb_params": lgb_params,
        "completed_seeds": [int(member["seed"]) for member in members],
        "members": members,
    }
    if walkforward is not None:
        payload["walkforward"] = walkforward
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(checkpoint_path)


def _load_training_checkpoint(
    checkpoint_path: Path,
    cfg: TrainRuntimeConfig,
    lgb_params: dict[str, object],
    expected_walkforward: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if not checkpoint_path.exists():
        return []

    payload = json.loads(checkpoint_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload at {checkpoint_path}: expected JSON object.")

    expected_meta = {
        "dataset_version": cfg.dataset_version,
        "feature_set": cfg.feature_set_name,
        "seeds": list(cfg.lgbm_seeds),
        "lgb_params": lgb_params,
    }
    for key, expected_value in expected_meta.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(
                f"Checkpoint mismatch for '{key}': got {payload.get(key)!r}, expected {expected_value!r}. "
                f"Delete {checkpoint_path} to retrain from scratch."
            )
    if expected_walkforward is not None and payload.get("walkforward") != expected_walkforward:
        raise RuntimeError(
            "Checkpoint mismatch for 'walkforward'. "
            f"Got {payload.get('walkforward')!r}, expected {expected_walkforward!r}. "
            f"Delete {checkpoint_path} to retrain from scratch."
        )

    members = payload.get("members")
    if not isinstance(members, list):
        raise RuntimeError(f"Invalid checkpoint payload at {checkpoint_path}: expected 'members' list.")

    normalized_members: list[dict[str, object]] = []
    for member in members:
        if not isinstance(member, dict):
            raise RuntimeError(f"Invalid checkpoint member in {checkpoint_path}: expected object, got {type(member)!r}.")
        try:
            seed = int(member["seed"])
            model_file = str(member["model_file"])
            best_iteration = int(member["best_iteration"])
            best_valid_rmse = float(member["best_valid_rmse"])
            best_valid_corr = float(member.get("best_valid_corr", np.nan))
            corr_scan_period = int(member["corr_scan_period"]) if member.get("corr_scan_period") is not None else None
            train_mode = str(member["train_mode"]) if member.get("train_mode") is not None else None
            recommended_num_iteration = (
                int(member["recommended_num_iteration"]) if member.get("recommended_num_iteration") is not None else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid checkpoint member schema in {checkpoint_path}: {member!r}") from exc
        if not model_file:
            raise RuntimeError(f"Invalid checkpoint member model_file in {checkpoint_path}: {member!r}")
        normalized_members.append(
            {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": best_iteration,
                "best_valid_rmse": best_valid_rmse,
                "best_valid_corr": best_valid_corr,
                "corr_scan_period": corr_scan_period,
                "train_mode": train_mode,
                "recommended_num_iteration": recommended_num_iteration,
            }
        )
    return normalized_members


def _log_seed_observability(result: FitResult) -> None:
    seed_label = str(result.seed)
    wandb.log(
        {
            f"seed/{seed_label}/best_iteration": result.best_iteration,
            f"seed/{seed_label}/best_valid_rmse": result.best_valid_rmse,
            f"seed/{seed_label}/best_valid_corr": result.best_valid_corr,
            f"seed/{seed_label}/learning_curve": wandb.plot.line_series(
                xs=list(range(1, len(result.train_rmse_curve) + 1)),
                ys=[result.train_rmse_curve, result.valid_rmse_curve],
                keys=["train_rmse", "valid_rmse"],
                title=f"Seed {seed_label} RMSE",
                xname="iteration",
            ),
            f"seed/{seed_label}/corr_curve": wandb.plot.line_series(
                xs=result.corr_scan_iters,
                ys=[result.valid_corr_curve],
                keys=["valid_corr_mean_per_era"],
                title=f"Seed {seed_label} Numerai CORR (scan)",
                xname="iteration",
            ),
        }
    )


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
            "corr_scan_period": cfg.corr_scan_period,
            "corr_scan_max_iters": cfg.corr_scan_max_iters,
            "select_best_by": cfg.select_best_by,
            "walkforward_enabled": cfg.walkforward_enabled,
            "walkforward_chunk_size": cfg.walkforward_chunk_size,
            "walkforward_purge_eras": cfg.walkforward_purge_eras,
            "walkforward_max_windows": cfg.walkforward_max_windows,
            "walkforward_tune_seed": cfg.walkforward_tune_seed,
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
    era_train = train_df.get_column(cfg.era_col).to_numpy()
    x_valid = valid_df.select(feature_cols).to_pandas()
    y_valid = valid_df.get_column(cfg.target_col).to_numpy().astype(np.float32)
    era_valid = valid_df.get_column(cfg.era_col).to_numpy()
    all_df = pl.concat([train_df, valid_df], how="vertical")
    x_all = all_df.select(feature_cols).to_pandas()
    y_all = all_df.get_column(cfg.target_col).to_numpy().astype(np.float32)
    era_all = all_df.get_column(cfg.era_col).to_numpy()
    era_all_int = era_to_int(era_all)

    order = np.argsort(era_all_int, kind="stable")
    x_all = x_all.iloc[order].reset_index(drop=True)
    y_all = y_all[order]
    era_all = era_all[order]
    era_all_int = era_all_int[order]

    del train_df, valid_df, all_df
    gc.collect()

    return LoadedData(
        feature_cols=feature_cols,
        x_train=x_train,
        y_train=y_train,
        era_train=era_train,
        x_valid=x_valid,
        y_valid=y_valid,
        era_valid=era_valid,
        x_all=x_all,
        y_all=y_all,
        era_all=era_all,
        era_all_int=era_all_int,
    )


def _corr_scan_iterations(cfg: TrainRuntimeConfig, max_iter: int) -> list[int]:
    if cfg.corr_scan_max_iters is not None:
        max_iter = min(max_iter, int(cfg.corr_scan_max_iters))
    max_iter = max(1, int(max_iter))
    scan_period = int(cfg.corr_scan_period)
    corr_scan_iters = list(range(scan_period, max_iter + 1, scan_period))
    if not corr_scan_iters or corr_scan_iters[-1] != max_iter:
        corr_scan_iters.append(max_iter)
    return corr_scan_iters


def _best_corr_iteration(
    model: lgb.Booster,
    x_valid: Any,
    y_valid: np.ndarray,
    era_valid: np.ndarray,
    corr_scan_iters: list[int],
) -> tuple[float, int, list[float]]:
    best_corr = float("-inf")
    best_corr_iter = corr_scan_iters[0]
    corr_curve: list[float] = []
    for i in corr_scan_iters:
        preds = model.predict(x_valid, num_iteration=i)
        corr_i = mean_per_era_numerai_corr(preds, y_valid, era_valid)
        corr_curve.append(float(corr_i))
        if corr_i > best_corr:
            best_corr = float(corr_i)
            best_corr_iter = int(i)
    return best_corr, best_corr_iter, corr_curve


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
    train_curve = [float(v) for v in evals_result["train"]["rmse"]]
    valid_curve = [float(v) for v in evals_result["valid"]["rmse"]]
    max_iter = int(model.current_iteration())
    corr_scan_iters = _corr_scan_iterations(cfg, max_iter)
    best_corr, best_corr_iter, corr_curve = _best_corr_iteration(
        model=model,
        x_valid=data.x_valid,
        y_valid=data.y_valid,
        era_valid=data.era_valid,
        corr_scan_iters=corr_scan_iters,
    )

    selected_best_iter = best_corr_iter if cfg.select_best_by == "corr" else best_iter
    logger.info(
        "phase=model_trained seed=%d best_iteration_rmse=%d best_iteration_corr=%d selected_best_iteration=%d "
        "best_valid_rmse=%.6f best_valid_corr=%.6f select_best_by=%s",
        seed,
        best_iter,
        best_corr_iter,
        selected_best_iter,
        best_rmse,
        best_corr,
        cfg.select_best_by,
    )
    return FitResult(
        seed=seed,
        model=model,
        best_iteration=selected_best_iter,
        best_valid_rmse=best_rmse,
        best_valid_corr=best_corr,
        train_rmse_curve=train_curve,
        valid_rmse_curve=valid_curve,
        valid_corr_curve=corr_curve,
        corr_scan_iters=corr_scan_iters,
    )


def fit_lgbm_final(
    lgb_params: dict[str, object],
    x: Any,
    y: np.ndarray,
    feature_cols: list[str],
    seed: int,
    num_boost_round: int,
) -> lgb.Booster:
    dtrain = lgb.Dataset(x, label=y, feature_name=feature_cols)
    params = dict(lgb_params)
    params["seed"] = seed
    return lgb.train(params=params, train_set=dtrain, num_boost_round=num_boost_round)


def evaluate_walkforward(cfg: TrainRuntimeConfig, lgb_params: dict[str, object], data: LoadedData) -> WalkforwardReport:
    windows = build_windows(
        era_numbers=data.era_all_int,
        chunk_size=cfg.walkforward_chunk_size,
        purge_eras=cfg.walkforward_purge_eras,
    )
    if cfg.walkforward_max_windows:
        windows = windows[-cfg.walkforward_max_windows :]
    if not windows:
        raise RuntimeError("No valid walk-forward windows found for current data and configuration.")

    tune_seed = int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0])
    rows: list[dict[str, float | int]] = []
    best_iters: list[int] = []
    window_corrs: list[float] = []

    for window in windows:
        train_idx = np.flatnonzero(data.era_all_int <= window.train_end)
        valid_idx = np.flatnonzero((data.era_all_int >= window.val_start) & (data.era_all_int <= window.val_end))
        if train_idx.size == 0 or valid_idx.size == 0:
            continue

        x_train = data.x_all.iloc[train_idx].reset_index(drop=True)
        y_train = data.y_all[train_idx]
        x_valid = data.x_all.iloc[valid_idx].reset_index(drop=True)
        y_valid = data.y_all[valid_idx]
        era_valid = data.era_all[valid_idx]

        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=data.feature_cols)
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=data.feature_cols)

        fit_params = dict(lgb_params)
        fit_params["seed"] = tune_seed
        evals_result: dict[str, dict[str, list[float]]] = {}
        try:
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=cfg.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(evals_result),
                ],
            )
        except lgb.basic.LightGBMError as exc:
            if fit_params.get("device") != "gpu" or "No OpenCL device found" not in str(exc):
                raise
            logger.warning("phase=wf_lgbm_gpu_unavailable requested_device=gpu fallback_device=cpu reason=%s", exc)
            fit_params["device"] = "cpu"
            fit_params.pop("gpu_use_dp", None)
            evals_result = {}
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=cfg.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(evals_result),
                ],
            )

        corr_scan_iters = _corr_scan_iterations(cfg, int(model.current_iteration()))
        best_corr, best_iter, _ = _best_corr_iteration(
            model=model,
            x_valid=x_valid,
            y_valid=y_valid,
            era_valid=era_valid,
            corr_scan_iters=corr_scan_iters,
        )
        preds = model.predict(x_valid, num_iteration=best_iter)
        corr_mean_per_era = float(mean_per_era_numerai_corr(preds, y_valid, era_valid))

        row = {
            "window_id": int(window.window_id),
            "train_end": int(window.train_end),
            "val_start": int(window.val_start),
            "val_end": int(window.val_end),
            "purge_start": int(window.purge_start),
            "purge_end": int(window.purge_end),
            "best_iter": int(best_iter),
            "best_corr": float(best_corr),
            "corr_mean_per_era": corr_mean_per_era,
        }
        rows.append(row)
        best_iters.append(int(best_iter))
        window_corrs.append(corr_mean_per_era)

    if not rows:
        raise RuntimeError("No walk-forward windows produced usable train/validation splits.")

    corr_arr = np.asarray(window_corrs, dtype=np.float64)
    mean_corr = float(np.mean(corr_arr))
    std_corr = float(np.std(corr_arr))
    sharpe = float(mean_corr / std_corr) if std_corr > 0 else 0.0
    hit_rate = float(np.mean(corr_arr > 0.0))
    recommended_num_iteration = int(np.median(np.asarray(best_iters, dtype=np.int32)))

    table = wandb.Table(
        columns=[
            "window_id",
            "train_end",
            "val_start",
            "val_end",
            "purge_start",
            "purge_end",
            "best_iter",
            "best_corr",
            "corr_mean_per_era",
        ]
    )
    for row in rows:
        table.add_data(
            int(row["window_id"]),
            int(row["train_end"]),
            int(row["val_start"]),
            int(row["val_end"]),
            int(row["purge_start"]),
            int(row["purge_end"]),
            int(row["best_iter"]),
            float(row["best_corr"]),
            float(row["corr_mean_per_era"]),
        )
    wandb.log(
        {
            "walkforward/windows": table,
            "walkforward/mean_corr": mean_corr,
            "walkforward/std_corr": std_corr,
            "walkforward/sharpe": sharpe,
            "walkforward/hit_rate": hit_rate,
            "walkforward/recommended_num_iteration": recommended_num_iteration,
        }
    )
    return WalkforwardReport(
        recommended_num_iteration=recommended_num_iteration,
        mean_corr=mean_corr,
        std_corr=std_corr,
        sharpe=sharpe,
        hit_rate=hit_rate,
        windows=rows,
    )


def save_and_log_artifact(
    cfg: TrainRuntimeConfig,
    run: Any,
    lgb_params: dict[str, object],
    data: LoadedData,
    members: list[dict[str, object]],
    checkpoint_dir: Path,
    wf_report: WalkforwardReport | None = None,
) -> SavedArtifact:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    model_paths: list[Path] = []
    for member in members:
        model_filename = str(member["model_file"])
        model_path = checkpoint_dir / model_filename
        if not model_path.exists():
            raise RuntimeError(f"Missing checkpointed model file: {model_path}")
        model_paths.append(model_path)
    valid_corr_values = [float(member.get("best_valid_corr", np.nan)) for member in members]

    features_out = out_dir / FEATURES_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME
    wf_windows_out = out_dir / "walkforward_windows.json"
    features_out.write_text(json.dumps(data.feature_cols, indent=2))
    wf_windows_payload = wf_report.windows if wf_report is not None else []
    wf_windows_out.write_text(json.dumps(wf_windows_payload, indent=2))

    walkforward_payload = None
    if wf_report is not None:
        walkforward_payload = {
            "enabled": bool(cfg.walkforward_enabled),
            "chunk_size": int(cfg.walkforward_chunk_size),
            "purge_eras": int(cfg.walkforward_purge_eras),
            "max_windows": int(cfg.walkforward_max_windows),
            "tune_seed": int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0]),
            "recommended_num_iteration": int(wf_report.recommended_num_iteration),
            "mean_corr": float(wf_report.mean_corr),
            "std_corr": float(wf_report.std_corr),
            "sharpe": float(wf_report.sharpe),
            "hit_rate": float(wf_report.hit_rate),
        }

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
                "members": members,
                "best_iteration_mean": float(np.mean([float(member["best_iteration"]) for member in members])),
                "best_valid_rmse_mean": float(np.mean([float(member["best_valid_rmse"]) for member in members])),
                "best_valid_corr_mean": float(np.nanmean(valid_corr_values)),
                "n_features": len(data.feature_cols),
                "model_name": cfg.model_name,
                "model_file": str(members[0]["model_file"]),
                "model_files": [path.name for path in model_paths],
                "lgb_params": lgb_params,
                "walkforward": walkforward_payload,
            },
            indent=2,
        )
    )

    artifact = wandb.Artifact(name=cfg.model_name, type="model")
    for model_path in model_paths:
        artifact.add_file(str(model_path), name=model_path.name)
    artifact.add_file(str(features_out), name=FEATURES_FILENAME)
    artifact.add_file(str(manifest_out), name=MANIFEST_FILENAME)
    artifact.add_file(str(wf_windows_out), name=wf_windows_out.name)
    run.log_artifact(artifact, aliases=["latest", "candidate"])

    logger.info(
        "phase=artifact_uploaded artifact_name=%s aliases=latest,candidate model_count=%d",
        cfg.model_name,
        len(model_paths),
    )
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
    wf_report: WalkforwardReport | None = None
    checkpoint_walkforward: dict[str, object] | None = None
    recommended_iter: int | None = None
    if cfg.walkforward_enabled:
        wf_report = evaluate_walkforward(cfg, lgb_params, data)
        recommended_iter = int(wf_report.recommended_num_iteration)
        checkpoint_walkforward = {
            "enabled": True,
            "chunk_size": int(cfg.walkforward_chunk_size),
            "purge_eras": int(cfg.walkforward_purge_eras),
            "max_windows": int(cfg.walkforward_max_windows),
            "tune_seed": int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0]),
            "recommended_num_iteration": recommended_iter,
        }
        logger.info(
            "phase=walkforward_completed recommended_num_iteration=%d mean_corr=%.6f std_corr=%.6f sharpe=%.6f hit_rate=%.6f",
            recommended_iter,
            wf_report.mean_corr,
            wf_report.std_corr,
            wf_report.sharpe,
            wf_report.hit_rate,
        )
    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME

    members = _load_training_checkpoint(
        checkpoint_path,
        cfg,
        lgb_params,
        expected_walkforward=checkpoint_walkforward,
    )
    completed_seeds = {int(member["seed"]) for member in members}
    if completed_seeds:
        logger.info(
            "phase=checkpoint_loaded checkpoint_path=%s completed_seeds=%s",
            checkpoint_path,
            sorted(completed_seeds),
        )

    for member in members:
        model_path = checkpoint_dir / str(member["model_file"])
        if not model_path.exists():
            raise RuntimeError(f"Checkpoint references missing model file: {model_path}")

    for seed in cfg.lgbm_seeds:
        if seed in completed_seeds:
            logger.info("phase=seed_skipped_already_completed seed=%d", seed)
            continue
        if cfg.walkforward_enabled:
            if recommended_iter is None:
                raise RuntimeError("Walk-forward is enabled but recommended_num_iteration is not available.")
            model_file = f"{cfg.model_name}_seed{seed}.txt"
            final_model = fit_lgbm_final(
                lgb_params=lgb_params,
                x=data.x_all,
                y=data.y_all,
                feature_cols=data.feature_cols,
                seed=seed,
                num_boost_round=recommended_iter,
            )
            final_model.save_model(str(checkpoint_dir / model_file), num_iteration=recommended_iter)
            member = {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": recommended_iter,
                "best_valid_rmse": float("nan"),
                "best_valid_corr": float(wf_report.mean_corr if wf_report is not None else np.nan),
                "corr_scan_period": cfg.corr_scan_period,
                "train_mode": "walkforward_final",
                "recommended_num_iteration": recommended_iter,
            }
            wandb.log(
                {
                    f"seed/{seed}/best_iteration": recommended_iter,
                    f"seed/{seed}/train_mode": "walkforward_final",
                }
            )
        else:
            fit_result = fit_lgbm(cfg, lgb_params, data, seed)
            model_file = f"{cfg.model_name}_seed{seed}.txt"
            fit_result.model.save_model(str(checkpoint_dir / model_file), num_iteration=fit_result.best_iteration)
            member = {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": fit_result.best_iteration,
                "best_valid_rmse": fit_result.best_valid_rmse,
                "best_valid_corr": fit_result.best_valid_corr,
                "corr_scan_period": cfg.corr_scan_period,
            }
            _log_seed_observability(fit_result)
        members.append(member)
        completed_seeds.add(seed)
        _write_training_checkpoint(
            checkpoint_path,
            cfg,
            lgb_params,
            members,
            walkforward=checkpoint_walkforward,
        )
        logger.info(
            "phase=seed_checkpoint_saved checkpoint_path=%s seed=%d completed=%d total=%d",
            checkpoint_path,
            seed,
            len(completed_seeds),
            len(cfg.lgbm_seeds),
        )

    if len(members) != len(cfg.lgbm_seeds):
        raise RuntimeError(
            f"Incomplete checkpoint state: expected {len(cfg.lgbm_seeds)} seeds, got {len(members)} members."
        )

    summary_table = wandb.Table(columns=["seed", "best_iteration", "best_valid_rmse", "best_valid_corr", "model_file"])
    for member in members:
        summary_table.add_data(
            int(member["seed"]),
            int(member["best_iteration"]),
            float(member["best_valid_rmse"]),
            float(member.get("best_valid_corr", np.nan)),
            str(member["model_file"]),
        )
    valid_corr_values = [float(member.get("best_valid_corr", np.nan)) for member in members]
    best_corr_member = max(
        members,
        key=lambda member: float(member.get("best_valid_corr", float("-inf"))),
    )
    wandb.log(
        {
            "best_iteration_mean": float(np.mean([float(member["best_iteration"]) for member in members])),
            "best_valid_rmse_mean": float(np.mean([float(member["best_valid_rmse"]) for member in members])),
            "best_valid_corr_mean": float(np.nanmean(valid_corr_values)),
            "best_valid_corr_max": float(np.nanmax(valid_corr_values)),
            "best_valid_corr_best_seed": int(best_corr_member["seed"]),
            "n_models": len(members),
            "ensemble_members": summary_table,
        }
    )

    save_and_log_artifact(cfg, run, lgb_params, data, members, checkpoint_dir, wf_report=wf_report)
    run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train()
