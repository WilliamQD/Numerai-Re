"""Colab training entrypoint using NumerAI official NumerAPI dataset flow.

This script is designed for high-memory Colab Pro environments and trains a
LightGBM model on NumerAI v5.2 with W&B artifact tracking.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import wandb
from numerapi import NumerAPI

from bench_matrix_builder import align_bench_to_ids
from benchmarks import download_benchmark_parquets, load_benchmark_frame
from config import TrainRuntimeConfig, _optional_bool_env
from data_loading import load_split_numpy
from era_utils import era_to_int
from feature_sampling import features_hash, sample_features_for_seed
from numerai_metrics import mean_per_era_numerai_corr
from postprocess import PostprocessConfig, apply_postprocess
from tune_blend import BlendTuneReport, tune_blend_on_windows
from walkforward import build_windows


FEATURES_FILENAME = "features.json"
FEATURES_UNION_FILENAME = "features_union.json"
FEATURES_BY_MODEL_FILENAME = "features_by_model.json"
MANIFEST_FILENAME = "train_manifest.json"
CHECKPOINT_FILENAME = "training_checkpoint.json"
ARTIFACT_SCHEMA_VERSION = 4
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
    feature_cols: list[str]
    x_train: Any
    y_train: np.ndarray
    era_train: np.ndarray
    id_train: np.ndarray
    bench_train: np.ndarray
    x_valid: Any
    y_valid: np.ndarray
    era_valid: np.ndarray
    id_valid: np.ndarray
    bench_valid: np.ndarray
    x_all: Any
    y_all: np.ndarray
    era_all: np.ndarray
    era_all_int: np.ndarray
    id_all: np.ndarray
    bench_all: np.ndarray
    bench_cols: list[str]


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


def _download_with_numerapi(cfg: TrainRuntimeConfig, data_dir: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
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

    datasets = napi.list_datasets()
    required_files = (
        (_resolve_dataset_path(datasets, cfg.dataset_version, ("train",), cfg.use_int8_parquet), train_path),
        (_resolve_dataset_path(datasets, cfg.dataset_version, ("validation",), cfg.use_int8_parquet), validation_path),
        (f"{cfg.dataset_version}/{FEATURES_FILENAME}", features_path),
    )
    for dataset_path, local_path in required_files:
        if local_path.exists():
            logger.info("phase=dataset_reused path=%s", local_path)
            continue
        logger.info("phase=dataset_downloading dataset=%s path=%s", dataset_path, local_path)
        napi.download_dataset(dataset_path, str(local_path))

    benchmark_paths = download_benchmark_parquets(napi, cfg.dataset_version, version_data_dir / "benchmarks")
    return train_path, validation_path, features_path, benchmark_paths


def _resolve_dataset_path(
    datasets: list[str],
    dataset_version: str,
    split_tokens: tuple[str, ...],
    use_int8: bool,
) -> str:
    prefix = dataset_version.lower() + "/"
    token_set = tuple(token.lower() for token in split_tokens)
    parquet_matches = [
        ds
        for ds in datasets
        if ds.lower().startswith(prefix)
        and ds.lower().endswith(".parquet")
        and all(token in ds.lower() for token in token_set)
    ]
    if not use_int8:
        default_match = next((ds for ds in parquet_matches if "int8" not in ds.lower()), None)
        return default_match or f"{dataset_version}/{'_'.join(split_tokens)}.parquet"
    int8_match = next((ds for ds in parquet_matches if "int8" in ds.lower()), None)
    if int8_match:
        return int8_match
    logger.warning(
        "phase=int8_dataset_fallback split=%s dataset_version=%s reason=int8_not_found",
        "_".join(split_tokens),
        dataset_version,
    )
    default_match = next((ds for ds in parquet_matches if "int8" not in ds.lower()), None)
    return default_match or f"{dataset_version}/{'_'.join(split_tokens)}.parquet"


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


def _write_features_mapping(path: Path, payload: dict[str, list[str]]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _load_features_mapping(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid features mapping at {path}: expected object.")
    normalized: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, list) and all(isinstance(col, str) for col in value):
            normalized[key] = value
    return normalized


def _checkpoint_dir(cfg: TrainRuntimeConfig) -> Path:
    return cfg.numerai_data_dir / cfg.dataset_version / "checkpoints" / cfg.model_name


def _write_training_checkpoint(
    checkpoint_path: Path,
    cfg: TrainRuntimeConfig,
    lgb_params: dict[str, object],
    members: list[dict[str, object]],
    walkforward: dict[str, object] | None = None,
    postprocess: dict[str, object] | None = None,
) -> None:
    payload = {
        "dataset_version": cfg.dataset_version,
        "feature_set": cfg.feature_set_name,
        "seeds": list(cfg.lgbm_seeds),
        "lgb_params": lgb_params,
        "max_features_per_model": cfg.max_features_per_model,
        "feature_sampling_strategy": cfg.feature_sampling_strategy,
        "feature_sampling_master_seed": cfg.feature_sampling_master_seed,
        "completed_seeds": [int(member["seed"]) for member in members],
        "members": members,
    }
    if walkforward is not None:
        payload["walkforward"] = walkforward
    if postprocess is not None:
        payload["postprocess"] = postprocess
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(checkpoint_path)


def _load_training_checkpoint(
    checkpoint_path: Path,
    cfg: TrainRuntimeConfig,
    lgb_params: dict[str, object],
    expected_walkforward: dict[str, object] | None = None,
    expected_postprocess: dict[str, object] | None = None,
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
        "max_features_per_model": cfg.max_features_per_model,
        "feature_sampling_strategy": cfg.feature_sampling_strategy,
        "feature_sampling_master_seed": cfg.feature_sampling_master_seed,
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
    if (
        expected_postprocess is not None
        and payload.get("postprocess") is not None
        and payload.get("postprocess") != expected_postprocess
    ):
        raise RuntimeError(
            "Checkpoint mismatch for 'postprocess'. "
            f"Got {payload.get('postprocess')!r}, expected {expected_postprocess!r}. "
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
            features_key = str(member["features_key"]) if member.get("features_key") is not None else model_file
            n_features_used = int(member.get("n_features_used", 0))
            member_features_hash = str(member.get("features_hash", ""))
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
                "features_key": features_key,
                "n_features_used": n_features_used,
                "features_hash": member_features_hash,
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
            "payout_weight_corr": cfg.payout_weight_corr,
            "payout_weight_bmc": cfg.payout_weight_bmc,
            "blend_alpha_grid": list(cfg.blend_alpha_grid),
            "bench_neutralize_prop_grid": list(cfg.bench_neutralize_prop_grid),
            "blend_tune_seed": cfg.blend_tune_seed,
            "blend_use_windows": cfg.blend_use_windows,
            "max_features_per_model": cfg.max_features_per_model,
            "feature_sampling_strategy": cfg.feature_sampling_strategy,
            "feature_sampling_master_seed": cfg.feature_sampling_master_seed,
            "use_int8_parquet": cfg.use_int8_parquet,
            "load_backend": cfg.load_backend,
            "load_mode": cfg.load_mode,
            **lgb_params,
        },
    )


def load_train_valid_frames(
    cfg: TrainRuntimeConfig,
    train_path: Path,
    validation_path: Path,
    benchmark_paths: dict[str, Path],
    feature_cols: list[str],
) -> LoadedData:
    logger.info(
        "phase=feature_subset_loading dataset_version=%s data_dir=%s n_features=%d",
        cfg.dataset_version,
        cfg.numerai_data_dir,
        len(feature_cols),
    )
    x_train, y_train, era_train, id_train = load_split_numpy(
        train_path, feature_cols, cfg.id_col, cfg.era_col, cfg.target_col
    )
    x_valid, y_valid, era_valid, id_valid = load_split_numpy(
        validation_path, feature_cols, cfg.id_col, cfg.era_col, cfg.target_col
    )
    logger.info("phase=frame_loaded split=train rows=%d cols=%d", x_train.shape[0], x_train.shape[1] + 3)
    logger.info("phase=frame_loaded split=validation rows=%d cols=%d", x_valid.shape[0], x_valid.shape[1] + 3)

    x_all = np.concatenate([x_train, x_valid], axis=0)
    y_all = np.concatenate([y_train, y_valid], axis=0)
    era_all = np.concatenate([era_train, era_valid], axis=0)
    id_all = np.concatenate([id_train, id_valid], axis=0)
    era_all_int = era_to_int(era_all)

    order = np.argsort(era_all_int, kind="stable")
    x_all = x_all[order]
    y_all = y_all[order]
    era_all = era_all[order]
    era_all_int = era_all_int[order]
    id_all = id_all[order]

    bench_train_df = load_benchmark_frame(benchmark_paths["train"])
    bench_valid_df = load_benchmark_frame(benchmark_paths["validation"])
    bench_train, bench_cols = align_bench_to_ids(id_train, bench_train_df, cfg.id_col)
    bench_valid, bench_cols_valid = align_bench_to_ids(id_valid, bench_valid_df, cfg.id_col)
    if bench_cols_valid != bench_cols:
        raise RuntimeError(
            f"Benchmark column mismatch: train has {bench_cols}, validation has {bench_cols_valid}."
        )
    bench_all_df = pl.concat([bench_train_df, bench_valid_df], how="vertical")
    bench_all, bench_cols_all = align_bench_to_ids(id_all, bench_all_df, cfg.id_col)
    if bench_cols_all != bench_cols:
        raise RuntimeError(f"Benchmark column mismatch after concat: expected {bench_cols}, got {bench_cols_all}.")

    del bench_train_df, bench_valid_df, bench_all_df
    gc.collect()

    return LoadedData(
        feature_cols=feature_cols,
        x_train=x_train,
        y_train=y_train,
        era_train=era_train,
        id_train=id_train,
        bench_train=bench_train,
        x_valid=x_valid,
        y_valid=y_valid,
        era_valid=era_valid,
        id_valid=id_valid,
        bench_valid=bench_valid,
        x_all=x_all,
        y_all=y_all,
        era_all=era_all,
        era_all_int=era_all_int,
        id_all=id_all,
        bench_all=bench_all,
        bench_cols=bench_cols,
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

        x_train = data.x_all[train_idx]
        y_train = data.y_all[train_idx]
        x_valid = data.x_all[valid_idx]
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


def _collect_blend_windows(
    cfg: TrainRuntimeConfig,
    lgb_params: dict[str, object],
    data: LoadedData,
    wf_report: WalkforwardReport,
) -> list[dict[str, np.ndarray | int]]:
    if cfg.blend_tune_seed is None:
        raise RuntimeError(
            "Blend tuning requires BLEND_TUNE_SEED (defaults to WALKFORWARD_TUNE_SEED when configured)."
        )
    tune_seed = int(cfg.blend_tune_seed)
    selected_windows = wf_report.windows
    if cfg.blend_use_windows:
        selected_windows = selected_windows[-int(cfg.blend_use_windows) :]

    rows: list[dict[str, np.ndarray | int]] = []
    for row in selected_windows:
        train_idx = np.flatnonzero(data.era_all_int <= int(row["train_end"]))
        valid_idx = np.flatnonzero(
            (data.era_all_int >= int(row["val_start"])) & (data.era_all_int <= int(row["val_end"]))
        )
        if train_idx.size == 0 or valid_idx.size == 0:
            continue

        x_train = data.x_all[train_idx]
        y_train = data.y_all[train_idx]
        x_valid = data.x_all[valid_idx]
        y_valid = data.y_all[valid_idx]
        era_valid = data.era_all[valid_idx]
        bench_valid = data.bench_all[valid_idx]

        fit_params = dict(lgb_params)
        fit_params["seed"] = tune_seed
        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=data.feature_cols)
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=data.feature_cols)
        try:
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=cfg.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
        except lgb.basic.LightGBMError as exc:
            if fit_params.get("device") != "gpu" or "No OpenCL device found" not in str(exc):
                raise
            fit_params["device"] = "cpu"
            fit_params.pop("gpu_use_dp", None)
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=cfg.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
        best_iter = min(int(row["best_iter"]), int(model.current_iteration()))
        if best_iter != int(row["best_iter"]):
            logger.warning(
                "phase=blend_best_iter_clamped window_id=%d reported_best_iter=%d current_iteration=%d",
                int(row["window_id"]),
                int(row["best_iter"]),
                int(model.current_iteration()),
            )
        pred_raw = model.predict(x_valid, num_iteration=best_iter).astype(np.float32, copy=False)
        rows.append(
            {
                "window_id": int(row["window_id"]),
                "pred_raw": pred_raw,
                "target": y_valid.astype(np.float32, copy=False),
                "era": era_valid,
                "bench": bench_valid,
            }
        )
    if not rows:
        raise RuntimeError("No windows available for blend tuning.")
    return rows


def save_and_log_artifact(
    cfg: TrainRuntimeConfig,
    run: Any,
    lgb_params: dict[str, object],
    feature_cols: list[str],
    features_by_model: dict[str, list[str]],
    members: list[dict[str, object]],
    checkpoint_dir: Path,
    wf_report: WalkforwardReport | None = None,
    postprocess_config: dict[str, object] | None = None,
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

    features_union = sorted({col for cols in features_by_model.values() for col in cols}) or list(feature_cols)
    features_out = out_dir / FEATURES_FILENAME
    features_union_out = out_dir / FEATURES_UNION_FILENAME
    features_by_model_out = out_dir / FEATURES_BY_MODEL_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME
    wf_windows_out = out_dir / "walkforward_windows.json"
    postprocess_out = out_dir / "postprocess_config.json"
    features_out.write_text(json.dumps(features_union, indent=2))
    features_union_out.write_text(json.dumps(features_union, indent=2))
    features_by_model_out.write_text(json.dumps(features_by_model, indent=2))
    wf_windows_payload = wf_report.windows if wf_report is not None else []
    wf_windows_out.write_text(json.dumps(wf_windows_payload, indent=2))
    postprocess_out.write_text(json.dumps(postprocess_config or {}, indent=2))

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
                "n_features": len(features_union),
                "model_name": cfg.model_name,
                "model_file": str(members[0]["model_file"]),
                "model_files": [path.name for path in model_paths],
                "features_union_file": FEATURES_UNION_FILENAME,
                "features_by_model_file": FEATURES_BY_MODEL_FILENAME,
                "max_features_per_model": cfg.max_features_per_model,
                "sampling_strategy": cfg.feature_sampling_strategy,
                "sampling_master_seed": cfg.feature_sampling_master_seed,
                "lgb_params": lgb_params,
                "walkforward": walkforward_payload,
                "postprocess": postprocess_config,
            },
            indent=2,
        )
    )

    artifact = wandb.Artifact(name=cfg.model_name, type="model")
    for model_path in model_paths:
        artifact.add_file(str(model_path), name=model_path.name)
    artifact.add_file(str(features_out), name=FEATURES_FILENAME)
    artifact.add_file(str(features_union_out), name=FEATURES_UNION_FILENAME)
    artifact.add_file(str(features_by_model_out), name=FEATURES_BY_MODEL_FILENAME)
    artifact.add_file(str(manifest_out), name=MANIFEST_FILENAME)
    artifact.add_file(str(wf_windows_out), name=wf_windows_out.name)
    artifact.add_file(str(postprocess_out), name=postprocess_out.name)
    run.log_artifact(artifact, aliases=["latest", "candidate"])

    logger.info(
        "phase=artifact_uploaded artifact_name=%s aliases=latest,candidate model_count=%d",
        cfg.model_name,
        len(model_paths),
    )
    return SavedArtifact(model_paths=model_paths, features_path=features_out, manifest_path=manifest_out)


def _build_synthetic_dry_run_data(cfg: TrainRuntimeConfig, scratch_dir: Path) -> LoadedData:
    rng = np.random.default_rng(7)
    feature_cols = [f"feature_{idx:02d}" for idx in range(8)]
    features_path = scratch_dir / FEATURES_FILENAME
    features_path.write_text(json.dumps({"feature_sets": {"medium": feature_cols}}, indent=2))
    feature_cols = _load_feature_list(features_path, cfg.feature_set_name)

    n_train, n_valid = 120, 60
    total_rows = n_train + n_valid
    eras = np.array([f"era{(idx // 10) + 1}" for idx in range(total_rows)], dtype=object)
    ids = np.array([f"dry_{idx:04d}" for idx in range(total_rows)], dtype=object)
    x_all_np = rng.normal(size=(total_rows, len(feature_cols))).astype(np.float32)
    y_all = (
        (0.55 * x_all_np[:, 0]) - (0.25 * x_all_np[:, 1]) + (0.10 * x_all_np[:, 2]) + rng.normal(0.0, 0.03, total_rows)
    ).astype(np.float32)
    bench_all = np.stack(
        (
            x_all_np[:, 0] + rng.normal(0.0, 0.02, total_rows),
            x_all_np[:, 1] + rng.normal(0.0, 0.02, total_rows),
        ),
        axis=1,
    ).astype(np.float32)
    bench_cols = ["benchmark_1", "benchmark_2"]
    x_all = x_all_np
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    era_train = eras[:n_train]
    id_train = ids[:n_train]
    bench_train = bench_all[:n_train]

    x_valid = x_all[n_train:]
    y_valid = y_all[n_train:]
    era_valid = eras[n_train:]
    id_valid = ids[n_train:]
    bench_valid = bench_all[n_train:]

    era_all_int = era_to_int(eras)
    return LoadedData(
        feature_cols=feature_cols,
        x_train=x_train,
        y_train=y_train,
        era_train=era_train,
        id_train=id_train,
        bench_train=bench_train,
        x_valid=x_valid,
        y_valid=y_valid,
        era_valid=era_valid,
        id_valid=id_valid,
        bench_valid=bench_valid,
        x_all=x_all,
        y_all=y_all,
        era_all=eras,
        era_all_int=era_all_int,
        id_all=ids,
        bench_all=bench_all,
        bench_cols=bench_cols,
    )


def train_dry_run() -> None:
    cfg = TrainRuntimeConfig(
        dataset_version=os.getenv("NUMERAI_DATASET_VERSION", "v5.2").strip() or "v5.2",
        feature_set_name="medium",
        wandb_api_key="dry-run",
        lgbm_seeds=(42,),
        num_boost_round=30,
        early_stopping_rounds=10,
        walkforward_enabled=False,
    )
    lgb_params = _resolve_lgb_params(cfg)
    scratch_dir = Path("artifacts") / "_dry_run"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    data = _build_synthetic_dry_run_data(cfg, scratch_dir)
    windows = build_windows(data.era_all_int, chunk_size=6, purge_eras=1)
    if not windows:
        raise RuntimeError("TRAIN_DRY_RUN failed to build walk-forward windows.")

    fit_result = fit_lgbm(cfg, lgb_params, data, seed=cfg.lgbm_seeds[0])
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    model_filename = f"{cfg.model_name}_seed{cfg.lgbm_seeds[0]}_dry_run.txt"
    model_path = out_dir / model_filename
    fit_result.model.save_model(str(model_path), num_iteration=fit_result.best_iteration)
    features_by_model = {model_filename: list(data.feature_cols)}

    post_cfg = PostprocessConfig(
        schema_version=1,
        submission_transform="rank_01",
        blend_alpha=0.7,
        bench_neutralize_prop=0.3,
        payout_weight_corr=cfg.payout_weight_corr,
        payout_weight_bmc=cfg.payout_weight_bmc,
        bench_cols_used=tuple(data.bench_cols),
    )
    pred_raw = fit_result.model.predict(data.x_valid, num_iteration=fit_result.best_iteration).astype(np.float32, copy=False)
    _ = apply_postprocess(pred_raw=pred_raw, era=data.era_valid, cfg=post_cfg, bench=data.bench_valid)

    (out_dir / FEATURES_FILENAME).write_text(json.dumps(data.feature_cols, indent=2))
    (out_dir / FEATURES_UNION_FILENAME).write_text(json.dumps(data.feature_cols, indent=2))
    (out_dir / FEATURES_BY_MODEL_FILENAME).write_text(json.dumps(features_by_model, indent=2))
    (out_dir / "postprocess_config.json").write_text(
        json.dumps(
            {
                "schema_version": post_cfg.schema_version,
                "submission_transform": post_cfg.submission_transform,
                "blend_alpha": post_cfg.blend_alpha,
                "bench_neutralize_prop": post_cfg.bench_neutralize_prop,
                "payout_weight_corr": post_cfg.payout_weight_corr,
                "payout_weight_bmc": post_cfg.payout_weight_bmc,
                "bench_cols_used": list(post_cfg.bench_cols_used),
                "feature_neutralize_prop": post_cfg.feature_neutralize_prop,
                "feature_neutralize_n_features": post_cfg.feature_neutralize_n_features,
                "feature_neutralize_seed": post_cfg.feature_neutralize_seed,
            },
            indent=2,
        )
    )
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "dataset_version": cfg.dataset_version,
                "feature_set": cfg.feature_set_name,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_name": cfg.model_name,
                "model_file": model_filename,
                "model_files": [model_filename],
                "features_union_file": FEATURES_UNION_FILENAME,
                "features_by_model_file": FEATURES_BY_MODEL_FILENAME,
                "max_features_per_model": cfg.max_features_per_model,
                "sampling_strategy": cfg.feature_sampling_strategy,
                "sampling_master_seed": cfg.feature_sampling_master_seed,
            },
            indent=2,
        )
    )
    print(f"TRAIN_DRY_RUN_OK windows={len(windows)} artifact_dir={out_dir}")


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
    train_path, validation_path, features_path, benchmark_paths = _download_with_numerapi(cfg, cfg.numerai_data_dir)
    feature_pool = _load_feature_list(features_path, cfg.feature_set_name)
    logger.info(
        "phase=datasets_downloaded dataset_version=%s data_dir=%s n_features=%d",
        cfg.dataset_version,
        cfg.numerai_data_dir,
        len(feature_pool),
    )
    sampled_features_by_seed = {
        int(seed): sample_features_for_seed(
            feature_pool=feature_pool,
            seed=int(seed),
            model_index=idx,
            n_models=len(cfg.lgbm_seeds),
            max_features_per_model=int(cfg.max_features_per_model),
            master_seed=int(cfg.feature_sampling_master_seed),
            strategy=cfg.feature_sampling_strategy,
        )
        for idx, seed in enumerate(cfg.lgbm_seeds)
    }
    base_seed = int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0])
    base_data = load_train_valid_frames(
        cfg,
        train_path=train_path,
        validation_path=validation_path,
        benchmark_paths=benchmark_paths,
        feature_cols=sampled_features_by_seed[base_seed],
    )
    wf_report: WalkforwardReport | None = None
    blend_report: BlendTuneReport | None = None
    checkpoint_walkforward: dict[str, object] | None = None
    checkpoint_postprocess: dict[str, object] = {
        "schema_version": 1,
        "submission_transform": "rank_01",
        "blend_alpha": 1.0,
        "bench_neutralize_prop": 0.0,
        "payout_weight_corr": float(cfg.payout_weight_corr),
        "payout_weight_bmc": float(cfg.payout_weight_bmc),
        "bench_cols_used": base_data.bench_cols,
        "feature_neutralize_prop": 0.0,
        "feature_neutralize_n_features": 0,
        "feature_neutralize_seed": 0,
    }
    recommended_iter: int | None = None
    if cfg.walkforward_enabled:
        wf_report = evaluate_walkforward(cfg, lgb_params, base_data)
        recommended_iter = int(wf_report.recommended_num_iteration)
        blend_windows = _collect_blend_windows(cfg, lgb_params, base_data, wf_report)
        blend_report = tune_blend_on_windows(
            windows=blend_windows,
            alpha_grid=cfg.blend_alpha_grid,
            prop_grid=cfg.bench_neutralize_prop_grid,
            payout_weight_corr=float(cfg.payout_weight_corr),
            payout_weight_bmc=float(cfg.payout_weight_bmc),
        )
        checkpoint_walkforward = {
            "enabled": True,
            "chunk_size": int(cfg.walkforward_chunk_size),
            "purge_eras": int(cfg.walkforward_purge_eras),
            "max_windows": int(cfg.walkforward_max_windows),
            "tune_seed": int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0]),
            "recommended_num_iteration": recommended_iter,
        }
        checkpoint_postprocess = {
            "schema_version": 1,
            "submission_transform": "rank_01",
            "blend_alpha": float(blend_report.best_alpha),
            "bench_neutralize_prop": float(blend_report.best_prop),
            "payout_weight_corr": float(cfg.payout_weight_corr),
            "payout_weight_bmc": float(cfg.payout_weight_bmc),
            "bench_cols_used": base_data.bench_cols,
            "feature_neutralize_prop": 0.0,
            "feature_neutralize_n_features": 0,
            "feature_neutralize_seed": 0,
        }
        logger.info(
            "phase=walkforward_completed recommended_num_iteration=%d mean_corr=%.6f std_corr=%.6f sharpe=%.6f hit_rate=%.6f",
            recommended_iter,
            wf_report.mean_corr,
            wf_report.std_corr,
            wf_report.sharpe,
            wf_report.hit_rate,
        )
    del base_data
    gc.collect()

    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    features_by_model_path = checkpoint_dir / FEATURES_BY_MODEL_FILENAME

    members = _load_training_checkpoint(
        checkpoint_path,
        cfg,
        lgb_params,
        expected_walkforward=checkpoint_walkforward,
        expected_postprocess=checkpoint_postprocess,
    )
    completed_seeds = {int(member["seed"]) for member in members}
    features_by_model = _load_features_mapping(features_by_model_path)
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
        features_key = str(member.get("features_key") or member["model_file"])
        if features_key not in features_by_model:
            seed = int(member["seed"])
            features_by_model[features_key] = sampled_features_by_seed[seed]
    _write_features_mapping(features_by_model_path, features_by_model)

    for seed in cfg.lgbm_seeds:
        if seed in completed_seeds:
            logger.info("phase=seed_skipped_already_completed seed=%d", seed)
            continue
        seed_features = sampled_features_by_seed[int(seed)]
        data = load_train_valid_frames(
            cfg,
            train_path=train_path,
            validation_path=validation_path,
            benchmark_paths=benchmark_paths,
            feature_cols=seed_features,
        )
        if cfg.walkforward_enabled:
            if recommended_iter is None:
                raise RuntimeError("Walk-forward is enabled but recommended_num_iteration is not available.")
            model_file = f"{cfg.model_name}_seed{seed}.txt"
            final_model = fit_lgbm_final(
                lgb_params=lgb_params,
                x=data.x_all,
                y=data.y_all,
                feature_cols=seed_features,
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
                "features_key": model_file,
                "n_features_used": len(seed_features),
                "features_hash": features_hash(seed_features),
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
                "features_key": model_file,
                "n_features_used": len(seed_features),
                "features_hash": features_hash(seed_features),
            }
            _log_seed_observability(fit_result)
        members.append(member)
        completed_seeds.add(seed)
        features_by_model[model_file] = seed_features
        _write_features_mapping(features_by_model_path, features_by_model)
        del data
        gc.collect()
        _write_training_checkpoint(
            checkpoint_path,
            cfg,
            lgb_params,
            members,
            walkforward=checkpoint_walkforward,
            postprocess=checkpoint_postprocess,
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
            "n_features_union": len({col for cols in features_by_model.values() for col in cols}),
            "ensemble_members": summary_table,
        }
    )

    save_and_log_artifact(
        cfg,
        run,
        lgb_params,
        feature_pool,
        features_by_model,
        members,
        checkpoint_dir,
        wf_report=wf_report,
        postprocess_config=checkpoint_postprocess,
    )
    run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if _optional_bool_env("TRAIN_DRY_RUN", default=False):
        train_dry_run()
    else:
        train()
