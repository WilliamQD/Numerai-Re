from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numerapi import NumerAPI

from artifact_contract import resolve_model_files
from config import InferenceRuntimeConfig
from numerapi_datasets import pick_benchmark_models_parquet, resolve_split_parquet
from postprocess import PostprocessConfig, apply_postprocess


logger = logging.getLogger(__name__)
RANK_01_EPSILON = 1e-6


class DriftGuardError(RuntimeError):
    """Raised when quality gates fail and submission must abort safely."""


def _download_live_benchmark_dataset(napi: NumerAPI, dataset_version: str, out_path: Path) -> Path:
    try:
        dataset = pick_benchmark_models_parquet(napi.list_datasets(), dataset_version, "live")
    except RuntimeError as exc:
        raise DriftGuardError(f"Could not find live benchmark models dataset under {dataset_version}/") from exc
    napi.download_dataset(dataset, str(out_path))
    return out_path


def _download_live_dataset(napi: NumerAPI, dataset_version: str, out_path: Path, use_int8_parquet: bool) -> Path:
    selected = resolve_split_parquet(
        napi.list_datasets(),
        dataset_version,
        ("live",),
        use_int8=use_int8_parquet,
    )
    if use_int8_parquet and "int8" not in selected.lower():
        logger.warning("phase=int8_live_fallback dataset_version=%s reason=int8_not_found", dataset_version)
    napi.download_dataset(selected, str(out_path))
    return out_path


def apply_quality_gates(
    features_df: pd.DataFrame,
    preds: np.ndarray,
    cfg: InferenceRuntimeConfig,
    submission_transform: str,
) -> None:
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise DriftGuardError("Predictions contain NaN/Inf")
    if np.allclose(preds, 0.0):
        raise DriftGuardError("Predictions are all zero")
    if submission_transform == "rank_01":
        if (preds < RANK_01_EPSILON).any() or (preds > 1.0 - RANK_01_EPSILON).any():
            raise DriftGuardError("Postprocessed rank_01 predictions must lie strictly in (0,1).")

    pred_std = float(np.std(preds))
    if pred_std < cfg.min_pred_std:
        raise DriftGuardError(f"Prediction std ({pred_std:.8f}) below threshold {cfg.min_pred_std}")

    sampled_features = features_df
    sampled_preds = preds
    if len(features_df) > cfg.exposure_sample_rows:
        rng = np.random.default_rng(cfg.exposure_sample_seed)
        sampled_idx = np.sort(rng.choice(len(features_df), size=cfg.exposure_sample_rows, replace=False))
        sampled_features = features_df.iloc[sampled_idx]
        sampled_preds = preds[sampled_idx]

    exposures = sampled_features.corrwith(pd.Series(sampled_preds, index=sampled_features.index)).abs()
    feature_exposure = float(exposures.max(skipna=True))
    if np.isnan(feature_exposure):
        raise DriftGuardError("Feature exposure could not be computed (NaN).")
    if feature_exposure > cfg.max_abs_exposure:
        raise DriftGuardError(
            f"Max feature exposure {feature_exposure:.5f} exceeds threshold {cfg.max_abs_exposure:.5f}"
        )


def _model_feature_cols(
    model_filename: str,
    features_by_model: dict[str, list[str]],
    union_feature_cols: list[str],
) -> list[str]:
    return features_by_model.get(model_filename, union_feature_cols)


def _load_live_frame(live_path: Path, feature_cols: list[str], selected_dataset_version: str) -> pd.DataFrame:
    selected_live_cols = sorted(set(feature_cols + ["era", "id"]))
    try:
        live_df = pd.read_parquet(live_path, columns=selected_live_cols)
    except (ValueError, KeyError) as exc:
        logger.warning(
            "phase=live_parquet_column_fallback selected_cols=%s reason=%s",
            selected_live_cols,
            exc,
        )
        live_df = pd.read_parquet(live_path)
    logger.info(
        "phase=frame_loaded split=live rows=%d cols=%d n_features=%d selected_dataset_version=%s",
        len(live_df),
        len(live_df.columns),
        len(feature_cols),
        selected_dataset_version,
    )
    missing = [col for col in feature_cols if col not in live_df.columns]
    if missing:
        raise DriftGuardError(f"Live data missing required features: {missing[:10]}")
    if "era" not in live_df.columns:
        raise DriftGuardError("Live data is missing required 'era' column.")
    return live_df


def _resolve_live_id(live_df: pd.DataFrame) -> np.ndarray:
    if "id" in live_df.columns:
        return live_df["id"].to_numpy()
    if live_df.index.name == "id":
        return live_df.index.to_numpy()
    raise DriftGuardError("Live data is missing required 'id' (expected as column or index).")


def _align_live_benchmarks(live_bench_path: Path, live_id: np.ndarray, bench_cols_used: list[str]) -> np.ndarray:
    if not bench_cols_used:
        raise DriftGuardError("postprocess_config.json has empty bench_cols_used.")
    bench_df = pd.read_parquet(live_bench_path, columns=["id", *bench_cols_used]).set_index("id")
    try:
        bench_aligned_df = bench_df.loc[live_id, bench_cols_used]
    except KeyError as exc:
        raise DriftGuardError("Benchmark parquet is missing ids required by live.parquet.") from exc
    if bench_aligned_df.isna().any().any():
        raise DriftGuardError("Benchmark alignment produced NaNs.")
    bench_aligned = bench_aligned_df.to_numpy(dtype=np.float32, copy=False)
    logger.info("phase=benchmarks_loaded n_bench_cols=%d", bench_aligned.shape[1])
    return bench_aligned


def run_live_inference(
    cfg: InferenceRuntimeConfig,
    models: list[Any],
    feature_cols: list[str],
    features_by_model: dict[str, list[str]],
    manifest: dict,
    post_cfg: PostprocessConfig,
    *,
    use_int8_parquet: bool,
) -> tuple[Path, pd.DataFrame, str, str]:
    napi = NumerAPI(public_id=cfg.numerai_public_id, secret_key=cfg.numerai_secret_key)
    live_path = Path("live.parquet")
    live_bench_path = Path("live_benchmark_models.parquet")
    _download_live_dataset(napi, cfg.dataset_version, live_path, use_int8_parquet=use_int8_parquet)
    _download_live_benchmark_dataset(napi, cfg.dataset_version, live_bench_path)
    logger.info("phase=datasets_downloaded dataset_version=%s live_path=%s", cfg.dataset_version, live_path)

    live_df = _load_live_frame(live_path, feature_cols, str(manifest.get("dataset_version", cfg.dataset_version)))
    live_id = _resolve_live_id(live_df)
    model_files = resolve_model_files(manifest, label="inference runtime")

    model_preds = []
    for model, model_file in zip(models, model_files):
        model_cols = _model_feature_cols(str(model_file), features_by_model, feature_cols)
        missing_model_cols = [col for col in model_cols if col not in live_df.columns]
        if missing_model_cols:
            raise DriftGuardError(f"Live data missing model-specific features for {model_file}: {missing_model_cols[:10]}")
        model_preds.append(model.predict(live_df[model_cols]).astype(np.float32))

    bench_aligned = _align_live_benchmarks(live_bench_path, live_id, list(post_cfg.bench_cols_used))

    pred_raw = np.mean(np.vstack(model_preds), axis=0, dtype=np.float32)
    features_for_postprocess: np.ndarray | None = None
    if post_cfg.feature_neutralize_prop > 0.0 and post_cfg.feature_neutralize_n_features > 0:
        n_features = min(post_cfg.feature_neutralize_n_features, len(feature_cols))
        rng = np.random.default_rng(post_cfg.feature_neutralize_seed)
        selected = np.sort(rng.choice(len(feature_cols), size=n_features, replace=False))
        features_for_postprocess = live_df[feature_cols].iloc[:, selected].to_numpy(dtype=np.float32, copy=False)

    pred_final = apply_postprocess(
        pred_raw=pred_raw,
        era=live_df["era"].to_numpy(),
        cfg=post_cfg,
        bench=bench_aligned,
        features=features_for_postprocess,
    )
    logger.info(
        "phase=postprocess_applied pred_min=%.6f pred_max=%.6f pred_std=%.6f",
        float(np.min(pred_final)),
        float(np.max(pred_final)),
        float(np.std(pred_final)),
    )
    apply_quality_gates(live_df[feature_cols], pred_final, cfg, submission_transform=post_cfg.submission_transform)

    submission = pd.DataFrame({"id": live_id, "prediction": pred_final})
    submission_path = Path("submission.csv")
    submission.to_csv(submission_path, index=False)

    models_dict = napi.get_models()
    if cfg.numerai_model_name not in models_dict:
        available = ", ".join(sorted(models_dict.keys()))
        raise RuntimeError(
            f"Configured NUMERAI_MODEL_NAME '{cfg.numerai_model_name}' not found among Numerai models. "
            f"Available models: {available or 'none'}"
        )

    model_id = models_dict[cfg.numerai_model_name]
    submission_id = napi.upload_predictions(str(submission_path), model_id=model_id)
    return submission_path, submission, model_id, submission_id
