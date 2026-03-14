from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import wandb

from numerai_re.shared import RuntimeStatusReporter, bmc_mean_per_era, gauss_rank_by_era, mean_per_era_numerai_corr
from numerai_re.training.checkpoints import load_signature_checkpoint, payload_list, write_signature_checkpoint
from numerai_re.training.walkforward import build_windows
from numerai_re.inference.postprocess import PostprocessConfig, apply_postprocess

logger = logging.getLogger(__name__)


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
    status: RuntimeStatusReporter | None = None,
    checkpoint_path: Path | None = None,
    resume_mode: str = "auto",
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
    wf_num_boost_round = int(getattr(cfg, "walkforward_num_boost_round", cfg.num_boost_round))
    logger.info("phase=walkforward_started n_windows=%d num_boost_round=%d", len(windows), wf_num_boost_round)
    stage_start = time.monotonic()

    tune_seed = int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0])
    rows: list[dict[str, float | int]] = []
    best_iters: list[int] = []
    window_corrs: list[float] = []
    total_windows = len(windows)
    rows_by_window_id: dict[int, dict[str, float | int]] = {}

    wf_signature = {
        "walkforward_chunk_size": int(cfg.walkforward_chunk_size),
        "walkforward_purge_eras": int(cfg.walkforward_purge_eras),
        "walkforward_max_windows": int(cfg.walkforward_max_windows),
        "walkforward_num_boost_round": int(wf_num_boost_round),
        "walkforward_tune_seed": int(tune_seed),
        "corr_scan_period": int(cfg.corr_scan_period),
        "corr_scan_max_iters": int(cfg.corr_scan_max_iters) if cfg.corr_scan_max_iters is not None else None,
        "early_stopping_rounds": int(cfg.early_stopping_rounds),
        "feature_set": str(cfg.feature_set_name),
        "n_rows": int(data.x_all.shape[0]),
        "n_features": int(data.x_all.shape[1]),
        "lgb_params": {str(key): str(value) for key, value in sorted(lgb_params.items())},
        "windows": [
            {
                "window_id": int(window.window_id),
                "train_end": int(window.train_end),
                "val_start": int(window.val_start),
                "val_end": int(window.val_end),
                "purge_start": int(window.purge_start),
                "purge_end": int(window.purge_end),
            }
            for window in windows
        ],
    }

    def _write_checkpoint() -> None:
        if checkpoint_path is None:
            return
        write_signature_checkpoint(
            checkpoint_path,
            wf_signature,
            {
                "rows": [rows_by_window_id[window.window_id] for window in windows if window.window_id in rows_by_window_id],
            },
        )

    payload = (
        load_signature_checkpoint(
            path=checkpoint_path,
            expected_signature=wf_signature,
            resume_mode=resume_mode,
            logger=logger,
            phase_name="walkforward",
        )
        if checkpoint_path is not None
        else None
    )
    for item in payload_list(payload, "rows"):
        if not isinstance(item, dict):
            continue
        try:
            window_id = int(item.get("window_id"))
            rows_by_window_id[window_id] = {
                "window_id": window_id,
                "train_end": int(item["train_end"]),
                "val_start": int(item["val_start"]),
                "val_end": int(item["val_end"]),
                "purge_start": int(item["purge_start"]),
                "purge_end": int(item["purge_end"]),
                "best_iter": int(item["best_iter"]),
                "best_corr": float(item["best_corr"]),
                "corr_mean_per_era": float(item["corr_mean_per_era"]),
            }
        except Exception:
            continue
    if rows_by_window_id:
        logger.info(
            "phase=walkforward_checkpoint_loaded path=%s completed=%d total=%d",
            checkpoint_path,
            len(rows_by_window_id),
            total_windows,
        )

    for window in windows:
        cached = rows_by_window_id.get(int(window.window_id))
        if cached is None:
            continue
        rows.append(cached)
        best_iters.append(int(cached["best_iter"]))
        window_corrs.append(float(cached["corr_mean_per_era"]))
        elapsed_s = max(0.0, time.monotonic() - stage_start)
        eta_s = (elapsed_s / len(rows)) * max(total_windows - len(rows), 0)
        logger.info(
            "phase=walkforward_window_restored window_id=%d completed=%d total=%d best_iter=%d corr_mean_per_era=%.6f elapsed_s=%.1f eta_s=%.1f",
            int(cached["window_id"]),
            len(rows),
            total_windows,
            int(cached["best_iter"]),
            float(cached["corr_mean_per_era"]),
            elapsed_s,
            eta_s,
        )

    for window in windows:
        if int(window.window_id) in rows_by_window_id:
            continue
        if status is not None:
            status.update(
                "walkforward_window",
                window=f"{int(window.window_id)}/{total_windows}",
                completed=len(rows),
                total=total_windows,
                force=True,
            )
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

        def _status_callback(env: Any) -> None:
            if status is None:
                return
            status.update(
                "walkforward_fit",
                window=f"{int(window.window_id)}/{total_windows}",
                iter=f"{int(env.iteration) + 1}/{int(env.end_iteration)}",
            )

        _status_callback.order = 0
        _status_callback.before_iteration = False

        try:
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=wf_num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(evals_result),
                    _status_callback,
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
                num_boost_round=wf_num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(evals_result),
                    _status_callback,
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
        rows_by_window_id[int(window.window_id)] = row
        best_iters.append(int(best_iter))
        window_corrs.append(corr_mean_per_era)
        _write_checkpoint()
        elapsed_s = max(0.0, time.monotonic() - stage_start)
        eta_s = (elapsed_s / len(rows)) * max(total_windows - len(rows), 0)
        logger.info(
            "phase=walkforward_window_completed window_id=%d completed=%d total=%d best_iter=%d corr_mean_per_era=%.6f elapsed_s=%.1f eta_s=%.1f",
            int(window.window_id),
            len(rows),
            total_windows,
            int(best_iter),
            corr_mean_per_era,
            elapsed_s,
            eta_s,
        )

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
    status: RuntimeStatusReporter | None = None,
    checkpoint_dir: Path | None = None,
    resume_mode: str = "auto",
) -> list[dict[str, np.ndarray | int]]:
    if cfg.blend_tune_seed is None:
        raise RuntimeError(
            "Blend tuning requires BLEND_TUNE_SEED (defaults to WALKFORWARD_TUNE_SEED when configured)."
        )
    tune_seed = int(cfg.blend_tune_seed)
    selected_windows = wf_report.windows
    if cfg.blend_use_windows:
        selected_windows = selected_windows[-int(cfg.blend_use_windows) :]
    logger.info("phase=blend_windows_collection_started n_windows=%d", len(selected_windows))
    wf_num_boost_round = int(getattr(cfg, "walkforward_num_boost_round", cfg.num_boost_round))
    stage_start = time.monotonic()

    cache_dir: Path | None = None
    cache_manifest_path: Path | None = None
    cache_signature: dict[str, object] | None = None
    cached_entries_by_window: dict[int, dict[str, object]] = {}
    if checkpoint_dir is not None:
        cache_dir = checkpoint_dir / "blend_windows"
        cache_manifest_path = checkpoint_dir / "blend_windows_checkpoint.json"
        cache_signature = {
            "blend_tune_seed": int(tune_seed),
            "walkforward_num_boost_round": int(wf_num_boost_round),
            "bench_min_covered_rows_per_window": int(cfg.bench_min_covered_rows_per_window),
            "bench_min_covered_eras_per_window": int(cfg.bench_min_covered_eras_per_window),
            "windows": [
                {
                    "window_id": int(row["window_id"]),
                    "train_end": int(row["train_end"]),
                    "val_start": int(row["val_start"]),
                    "val_end": int(row["val_end"]),
                    "best_iter": int(row["best_iter"]),
                }
                for row in selected_windows
            ],
        }
        payload = load_signature_checkpoint(
            path=cache_manifest_path,
            expected_signature=cache_signature,
            resume_mode=resume_mode,
            logger=logger,
            phase_name="blend_windows_cache",
        )
        for entry in payload_list(payload, "windows"):
            if not isinstance(entry, dict):
                continue
            try:
                window_id = int(entry.get("window_id"))
                file_name = str(entry.get("file"))
            except Exception:
                continue
            if file_name:
                cached_entries_by_window[window_id] = {
                    "window_id": window_id,
                    "file": file_name,
                }
        if cached_entries_by_window:
            logger.info(
                "phase=blend_windows_cache_loaded path=%s cached_windows=%d total=%d",
                cache_manifest_path,
                len(cached_entries_by_window),
                len(selected_windows),
            )

    def _write_cache_manifest(windows_rows: list[dict[str, np.ndarray | int]]) -> None:
        if cache_manifest_path is None or cache_signature is None or cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        windows_payload: list[dict[str, object]] = []
        for item in windows_rows:
            windows_payload.append(
                {
                    "window_id": int(item["window_id"]),
                    "file": f"blend_window_{int(item['window_id'])}.npz",
                }
            )
        write_signature_checkpoint(cache_manifest_path, cache_signature, {"windows": windows_payload})

    rows: list[dict[str, np.ndarray | int]] = []
    total_windows = len(selected_windows)
    for idx, row in enumerate(selected_windows, start=1):
        if status is not None:
            status.update(
                "blend_window",
                window=f"{idx}/{total_windows}",
                selected_window_id=int(row["window_id"]),
                completed=len(rows),
                force=True,
            )
        train_idx = np.flatnonzero(data.era_all_int <= int(row["train_end"]))
        valid_idx = np.flatnonzero(
            (data.era_all_int >= int(row["val_start"])) & (data.era_all_int <= int(row["val_end"]))
        )
        if train_idx.size == 0 or valid_idx.size == 0:
            continue

        bench_mask = data.bench_all_mask[valid_idx]
        covered_valid_idx = valid_idx[bench_mask]
        covered_rows = int(covered_valid_idx.size)
        if covered_rows < int(cfg.bench_min_covered_rows_per_window):
            logger.warning(
                "phase=blend_window_skipped window_id=%d reason=low_benchmark_row_coverage covered_rows=%d required_rows=%d",
                int(row["window_id"]),
                covered_rows,
                int(cfg.bench_min_covered_rows_per_window),
            )
            continue
        covered_eras = int(np.unique(data.era_all[covered_valid_idx]).size)
        if covered_eras < int(cfg.bench_min_covered_eras_per_window):
            logger.warning(
                "phase=blend_window_skipped window_id=%d reason=low_benchmark_era_coverage covered_eras=%d required_eras=%d",
                int(row["window_id"]),
                covered_eras,
                int(cfg.bench_min_covered_eras_per_window),
            )
            continue

        window_id = int(row["window_id"])
        if cache_dir is not None and window_id in cached_entries_by_window:
            cache_file = cache_dir / str(cached_entries_by_window[window_id]["file"])
            if cache_file.exists():
                try:
                    cached = np.load(cache_file, allow_pickle=False)
                    rows.append(
                        {
                            "window_id": window_id,
                            "pred_raw": cached["pred_raw"].astype(np.float32, copy=False),
                            "target": cached["target"].astype(np.float32, copy=False),
                            "era": cached["era"],
                            "bench": cached["bench"].astype(np.float32, copy=False),
                        }
                    )
                    elapsed_s = max(0.0, time.monotonic() - stage_start)
                    eta_s = (elapsed_s / len(rows)) * max(total_windows - len(rows), 0)
                    logger.info(
                        "phase=blend_window_restored selected_window_id=%d completed=%d total=%d elapsed_s=%.1f eta_s=%.1f",
                        window_id,
                        len(rows),
                        total_windows,
                        elapsed_s,
                        eta_s,
                    )
                    continue
                except Exception as exc:
                    if resume_mode == "strict":
                        raise
                    logger.warning(
                        "phase=blend_window_restore_failed selected_window_id=%d action=recompute reason=%s",
                        window_id,
                        exc,
                    )

        x_train = data.x_all[train_idx]
        y_train = data.y_all[train_idx]
        x_valid = data.x_all[covered_valid_idx]
        y_valid = data.y_all[covered_valid_idx]
        era_valid = data.era_all[covered_valid_idx]
        bench_valid = data.bench_all[covered_valid_idx]

        fit_params = dict(lgb_params)
        fit_params["seed"] = tune_seed
        dtrain = lgb.Dataset(x_train, label=y_train, feature_name=data.feature_cols, free_raw_data=True)
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, feature_name=data.feature_cols, free_raw_data=True)
        del x_train

        def _status_callback(env: Any) -> None:
            if status is None:
                return
            status.update(
                "blend_window_fit",
                window=f"{idx}/{total_windows}",
                selected_window_id=int(row["window_id"]),
                iter=f"{int(env.iteration) + 1}/{int(env.end_iteration)}",
            )

        _status_callback.order = 0
        _status_callback.before_iteration = False

        try:
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=wf_num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    _status_callback,
                ],
            )
        except lgb.basic.LightGBMError as exc:
            if fit_params.get("device") != "gpu" or "No OpenCL device found" not in str(exc):
                raise
            fit_params["device"] = "cpu"
            fit_params.pop("gpu_use_dp", None)
            model = lgb.train(
                params=fit_params,
                train_set=dtrain,
                num_boost_round=wf_num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    _status_callback,
                ],
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
                "window_id": window_id,
                "pred_raw": pred_raw,
                "target": y_valid.astype(np.float32, copy=False),
                "era": era_valid,
                "bench": bench_valid,
            }
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"blend_window_{window_id}.npz"
            np.savez_compressed(
                cache_file,
                pred_raw=pred_raw,
                target=y_valid.astype(np.float32, copy=False),
                era=era_valid,
                bench=bench_valid.astype(np.float32, copy=False),
            )
            _write_cache_manifest(rows)
        elapsed_s = max(0.0, time.monotonic() - stage_start)
        eta_s = (elapsed_s / len(rows)) * max(total_windows - len(rows), 0)
        logger.info(
            "phase=blend_window_completed selected_window_id=%d completed=%d total=%d covered_rows=%d elapsed_s=%.1f eta_s=%.1f",
            window_id,
            len(rows),
            total_windows,
            covered_rows,
            elapsed_s,
            eta_s,
        )
    if not rows:
        raise RuntimeError("No windows available for blend tuning.")
    return rows

# --- Blend tuning ---


@dataclass(frozen=True)
class BlendTuneReport:
    best_alpha: float
    best_prop: float
    best_hit_rate: float
    best_mean_score: float
    best_mean_corr: float
    best_mean_bmc: float
    search_rows: list[dict[str, float]]
    window_rows: list[dict[str, float | int]]


def blended_predictions(
    pred_raw: np.ndarray,
    era: np.ndarray,
    bench: np.ndarray,
    alpha: float,
    neutralize_prop: float,
    *,
    bench_gauss: np.ndarray | None = None,
) -> np.ndarray:
    cfg = PostprocessConfig(
        schema_version=1,
        submission_transform="gauss_rank",
        blend_alpha=float(alpha),
        bench_neutralize_prop=float(neutralize_prop),
    )
    return apply_postprocess(pred_raw.astype(np.float32, copy=False), era, cfg, bench=bench, bench_gauss=bench_gauss)


def tune_blend_on_windows(
    windows: list[dict[str, np.ndarray | int]],
    alpha_grid: tuple[float, ...],
    prop_grid: tuple[float, ...],
    payout_weight_corr: float,
    payout_weight_bmc: float,
    *,
    status: RuntimeStatusReporter | None = None,
    checkpoint_path: Path | None = None,
    resume_mode: str = "auto",
) -> BlendTuneReport:
    search_rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_metrics: tuple[float, float, float] | None = None
    total_combos = len(alpha_grid) * len(prop_grid)
    combo_index = 0
    tune_start = time.monotonic()

    def _combo_key(alpha: float, prop: float) -> str:
        return f"{float(alpha):.12g}|{float(prop):.12g}"

    def _window_signature(window: dict[str, np.ndarray | int]) -> dict[str, int]:
        bench = window["bench"]  # type: ignore[index]
        target = window["target"]  # type: ignore[index]
        return {
            "window_id": int(window["window_id"]),  # type: ignore[arg-type]
            "n_rows": int(target.shape[0]),
            "n_bench_cols": int(bench.shape[1]),
        }

    def _build_signature(prepared: list[dict[str, np.ndarray | int]]) -> dict[str, object]:
        return {
            "alpha_grid": [float(v) for v in alpha_grid],
            "prop_grid": [float(v) for v in prop_grid],
            "payout_weight_corr": float(payout_weight_corr),
            "payout_weight_bmc": float(payout_weight_bmc),
            "windows": [_window_signature(window) for window in prepared],
        }

    def _write_checkpoint(rows: list[dict[str, float]], signature: dict[str, object]) -> None:
        if checkpoint_path is None:
            return
        write_signature_checkpoint(checkpoint_path, signature, {"search_rows": rows})

    prepared_windows: list[dict[str, np.ndarray | int]] = []
    for window in windows:
        bench = window["bench"]  # type: ignore[index]
        era = window["era"]  # type: ignore[index]
        bench_gauss = np.empty_like(bench, dtype=np.float32)
        for idx in range(bench.shape[1]):
            bench_gauss[:, idx] = gauss_rank_by_era(bench[:, idx], era)
        prepared_windows.append({**window, "bench_gauss": bench_gauss})

    signature = _build_signature(prepared_windows)

    done_rows_by_key: dict[str, dict[str, float]] = {}
    payload = (
        load_signature_checkpoint(
            path=checkpoint_path,
            expected_signature=signature,
            resume_mode=resume_mode,
            logger=logger,
            phase_name="blend_tune",
        )
        if checkpoint_path is not None
        else None
    )
    for row in payload_list(payload, "search_rows"):
        if not isinstance(row, dict):
            continue
        alpha = float(row.get("alpha", np.nan))
        prop = float(row.get("prop", np.nan))
        if np.isnan(alpha) or np.isnan(prop):
            continue
        done_rows_by_key[_combo_key(alpha, prop)] = {
            "alpha": alpha,
            "prop": prop,
            "hit_rate": float(row.get("hit_rate", np.nan)),
            "mean_score": float(row.get("mean_score", np.nan)),
            "mean_corr": float(row.get("mean_corr", np.nan)),
            "mean_bmc": float(row.get("mean_bmc", np.nan)),
        }
    if done_rows_by_key:
        search_rows = list(done_rows_by_key.values())
        for row in search_rows:
            metrics = (row["hit_rate"], row["mean_score"], row["mean_corr"])
            if best_row is None or best_metrics is None or metrics > best_metrics:
                best_row = row
                best_metrics = metrics
        logger.info(
            "phase=blend_tune_checkpoint_loaded path=%s completed=%d total=%d",
            checkpoint_path,
            len(done_rows_by_key),
            total_combos,
        )

    logger.info(
        "phase=blend_tune_started n_windows=%d total_combos=%d payout_weight_corr=%.3f payout_weight_bmc=%.3f",
        len(prepared_windows),
        total_combos,
        float(payout_weight_corr),
        float(payout_weight_bmc),
    )

    for alpha in alpha_grid:
        for prop in prop_grid:
            combo_index += 1
            key = _combo_key(float(alpha), float(prop))
            if key in done_rows_by_key:
                continue
            combo_start = time.monotonic()
            scores: list[float] = []
            corrs: list[float] = []
            bmcs: list[float] = []
            for window_index, window in enumerate(prepared_windows, start=1):
                if status is not None:
                    status.update(
                        "blend_tune",
                        combo=f"{combo_index}/{total_combos}",
                        window=f"{window_index}/{len(windows)}",
                        alpha=f"{float(alpha):.2f}",
                        prop=f"{float(prop):.2f}",
                    )
                pred_raw = window["pred_raw"]  # type: ignore[index]
                target = window["target"]  # type: ignore[index]
                era = window["era"]  # type: ignore[index]
                bench = window["bench"]  # type: ignore[index]
                bench_gauss = window["bench_gauss"]  # type: ignore[index]
                blended = blended_predictions(
                    pred_raw,
                    era,
                    bench,
                    alpha=float(alpha),
                    neutralize_prop=float(prop),
                    bench_gauss=bench_gauss,
                )
                corr = float(mean_per_era_numerai_corr(blended, target, era))
                bmc = float(
                    bmc_mean_per_era(
                        blended,
                        target,
                        era,
                        bench,
                        neutralize_prop=float(prop),
                        bench_gauss=bench_gauss,
                    )
                )
                score = float((payout_weight_corr * corr) + (payout_weight_bmc * bmc))
                corrs.append(corr)
                bmcs.append(bmc)
                scores.append(score)

            score_arr = np.asarray(scores, dtype=np.float64)
            corr_arr = np.asarray(corrs, dtype=np.float64)
            bmc_arr = np.asarray(bmcs, dtype=np.float64)
            row = {
                "alpha": float(alpha),
                "prop": float(prop),
                "hit_rate": float(np.mean(score_arr > 0.0)),
                "mean_score": float(np.mean(score_arr)),
                "mean_corr": float(np.mean(corr_arr)),
                "mean_bmc": float(np.mean(bmc_arr)),
            }
            search_rows.append(row)
            done_rows_by_key[key] = row
            metrics = (row["hit_rate"], row["mean_score"], row["mean_corr"])
            if best_row is None or metrics > best_metrics:
                best_row = row
                best_metrics = metrics
            _write_checkpoint(search_rows, signature)

            elapsed_s = max(0.0, time.monotonic() - tune_start)
            combo_elapsed_s = max(0.0, time.monotonic() - combo_start)
            completed = len(done_rows_by_key)
            eta_s = (elapsed_s / completed) * max(total_combos - completed, 0) if completed > 0 else 0.0
            logger.info(
                "phase=blend_tune_progress combo=%d total=%d alpha=%.2f prop=%.2f combo_elapsed_s=%.1f elapsed_s=%.1f eta_s=%.1f elapsed_min=%.2f eta_min=%.2f",
                completed,
                total_combos,
                float(alpha),
                float(prop),
                combo_elapsed_s,
                elapsed_s,
                eta_s,
                elapsed_s / 60.0,
                eta_s / 60.0,
            )

    if best_row is None:
        raise RuntimeError("Blend tuning did not produce any candidate rows.")

    best_alpha = float(best_row["alpha"])
    best_prop = float(best_row["prop"])
    window_rows: list[dict[str, float | int]] = []
    for window in prepared_windows:
        pred_raw = window["pred_raw"]  # type: ignore[index]
        target = window["target"]  # type: ignore[index]
        era = window["era"]  # type: ignore[index]
        bench = window["bench"]  # type: ignore[index]
        bench_gauss = window["bench_gauss"]  # type: ignore[index]
        blended = blended_predictions(
            pred_raw,
            era,
            bench,
            alpha=best_alpha,
            neutralize_prop=best_prop,
            bench_gauss=bench_gauss,
        )
        corr = float(mean_per_era_numerai_corr(blended, target, era))
        bmc = float(
            bmc_mean_per_era(
                blended,
                target,
                era,
                bench,
                neutralize_prop=best_prop,
                bench_gauss=bench_gauss,
            )
        )
        score = float((payout_weight_corr * corr) + (payout_weight_bmc * bmc))
        window_rows.append(
            {
                "window_id": int(window["window_id"]),  # type: ignore[arg-type]
                "alpha": best_alpha,
                "prop": best_prop,
                "corr": corr,
                "bmc": bmc,
                "score": score,
                "hit": int(score > 0.0),
            }
        )

    search_table = wandb.Table(columns=["alpha", "prop", "hit_rate", "mean_score", "mean_corr", "mean_bmc"])
    for row in search_rows:
        search_table.add_data(
            row["alpha"],
            row["prop"],
            row["hit_rate"],
            row["mean_score"],
            row["mean_corr"],
            row["mean_bmc"],
        )
    best_windows_table = wandb.Table(columns=["window_id", "alpha", "prop", "corr", "bmc", "score", "hit"])
    for row in window_rows:
        best_windows_table.add_data(
            int(row["window_id"]),
            float(row["alpha"]),
            float(row["prop"]),
            float(row["corr"]),
            float(row["bmc"]),
            float(row["score"]),
            int(row["hit"]),
        )
    wandb.log(
        {
            "blend/search": search_table,
            "blend/windows": best_windows_table,
            "blend/best_alpha": best_alpha,
            "blend/best_prop": best_prop,
            "blend/best_hit_rate": float(best_row["hit_rate"]),
            "blend/best_mean_score": float(best_row["mean_score"]),
            "blend/best_mean_corr": float(best_row["mean_corr"]),
            "blend/best_mean_bmc": float(best_row["mean_bmc"]),
        }
    )
    return BlendTuneReport(
        best_alpha=best_alpha,
        best_prop=best_prop,
        best_hit_rate=float(best_row["hit_rate"]),
        best_mean_score=float(best_row["mean_score"]),
        best_mean_corr=float(best_row["mean_corr"]),
        best_mean_bmc=float(best_row["mean_bmc"]),
        search_rows=search_rows,
        window_rows=window_rows,
    )