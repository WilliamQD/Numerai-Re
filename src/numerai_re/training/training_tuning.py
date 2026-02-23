from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import wandb

from numerai_re.metrics.numerai_metrics import mean_per_era_numerai_corr
from numerai_re.training.walkforward import build_windows


@dataclass(frozen=True)
class WalkforwardReport:
    recommended_num_iteration: int
    mean_corr: float
    std_corr: float
    sharpe: float
    hit_rate: float
    windows: list[dict[str, float | int]]


def corr_scan_iterations(cfg: Any, max_iter: int) -> list[int]:
    if cfg.corr_scan_max_iters is not None:
        max_iter = min(max_iter, int(cfg.corr_scan_max_iters))
    max_iter = max(1, int(max_iter))
    scan_period = int(cfg.corr_scan_period)
    corr_scan_iters = list(range(scan_period, max_iter + 1, scan_period))
    if not corr_scan_iters or corr_scan_iters[-1] != max_iter:
        corr_scan_iters.append(max_iter)
    return corr_scan_iters


def best_corr_iteration(
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


def evaluate_walkforward(
    cfg: Any,
    lgb_params: dict[str, object],
    data: Any,
    *,
    logger: logging.Logger,
) -> WalkforwardReport:
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

        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=data.feature_cols, free_raw_data=True)
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=data.feature_cols, free_raw_data=True)
        del x_train

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
        del dtrain, dvalid

        corr_iters = corr_scan_iterations(cfg, int(model.current_iteration()))
        best_corr, best_iter, _ = best_corr_iteration(
            model=model,
            x_valid=x_valid,
            y_valid=y_valid,
            era_valid=era_valid,
            corr_scan_iters=corr_iters,
        )
        preds = model.predict(x_valid, num_iteration=best_iter)
        corr_mean_per_era = float(mean_per_era_numerai_corr(preds, y_valid, era_valid))
        del x_valid, model
        gc.collect()

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


def collect_blend_windows(
    cfg: Any,
    lgb_params: dict[str, object],
    data: Any,
    wf_report: WalkforwardReport,
    *,
    logger: logging.Logger,
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
        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=data.feature_cols, free_raw_data=True)
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=data.feature_cols, free_raw_data=True)
        del x_train
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
        del dtrain, dvalid
        gc.collect()
        best_iter = min(int(row["best_iter"]), int(model.current_iteration()))
        if best_iter != int(row["best_iter"]):
            logger.warning(
                "phase=blend_best_iter_clamped window_id=%d reported_best_iter=%d current_iteration=%d",
                int(row["window_id"]),
                int(row["best_iter"]),
                int(model.current_iteration()),
            )
        pred_raw = model.predict(x_valid, num_iteration=best_iter).astype(np.float32, copy=False)
        del x_valid, model
        gc.collect()
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
