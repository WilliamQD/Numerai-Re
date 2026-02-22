"""Weekly inference + submission script for GitHub Actions."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from numerapi import NumerAPI

from config import InferenceRuntimeConfig, _optional_bool_env
from postprocess import PostprocessConfig, apply_postprocess


FEATURES_FILENAME = "features.json"
FEATURES_UNION_FILENAME = "features_union.json"
FEATURES_BY_MODEL_FILENAME = "features_by_model.json"
MANIFEST_FILENAME = "train_manifest.json"
POSTPROCESS_FILENAME = "postprocess_config.json"
REQUIRED_MANIFEST_KEYS = ("dataset_version", "feature_set")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
RANK_01_EPSILON = 1e-6


logger = logging.getLogger(__name__)


class DriftGuardError(RuntimeError):
    """Raised when quality gates fail and submission must abort safely."""


def _resolve_manifest(manifest_path: Path, model_name: str) -> tuple[dict, list[str]]:
    default_file = [f"{model_name}.txt"]
    if not manifest_path.exists():
        logger.warning("Manifest file '%s' is missing. Falling back to default model filename.", MANIFEST_FILENAME)
        return {}, default_file

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Manifest file '%s' is malformed JSON. Falling back to default model filename.", MANIFEST_FILENAME)
        return {}, default_file
    if not isinstance(manifest, dict):
        logger.warning("Manifest file '%s' is not a JSON object. Falling back to default model filename.", MANIFEST_FILENAME)
        return {}, default_file

    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_keys:
        logger.warning("Manifest file '%s' missing keys %s.", MANIFEST_FILENAME, missing_keys)

    model_files = manifest.get("model_files")
    if isinstance(model_files, list) and model_files and all(isinstance(f, str) and f.strip() for f in model_files):
        return manifest, model_files

    model_filename = manifest.get("model_file")
    if isinstance(model_filename, str) and model_filename.strip():
        return manifest, [model_filename]

    logger.warning("Manifest model filename keys are invalid. Falling back to default model filename.")
    return manifest, default_file


def _load_artifact_from_root(
    root: Path, cfg: InferenceRuntimeConfig, artifact_ref: str
) -> tuple[list[lgb.Booster], list[str], dict[str, list[str]], dict, PostprocessConfig]:
    features_path = root / FEATURES_FILENAME
    features_union_path = root / FEATURES_UNION_FILENAME
    features_by_model_path = root / FEATURES_BY_MODEL_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    postprocess_path = root / POSTPROCESS_FILENAME
    manifest, model_filenames = _resolve_manifest(manifest_path, cfg.wandb_model_name)

    if not features_path.exists() and not features_union_path.exists():
        raise RuntimeError(f"Model artifact is missing required file '{FEATURES_FILENAME}'.")
    if not postprocess_path.exists():
        raise RuntimeError(f"Model artifact is missing required file '{POSTPROCESS_FILENAME}'.")

    missing_models = [name for name in model_filenames if not (root / name).exists()]
    if missing_models:
        raise RuntimeError(f"Model artifact is missing model files: {missing_models}")

    feature_cols_path = features_union_path if features_union_path.exists() else features_path
    feature_cols = json.loads(feature_cols_path.read_text())
    if not isinstance(feature_cols, list) or not feature_cols or not all(isinstance(col, str) and col for col in feature_cols):
        raise RuntimeError(f"Invalid '{feature_cols_path.name}' in model artifact: expected non-empty list of features.")

    features_by_model: dict[str, list[str]] = {}
    if features_by_model_path.exists():
        raw_mapping = json.loads(features_by_model_path.read_text())
        if not isinstance(raw_mapping, dict):
            raise RuntimeError(f"Invalid '{FEATURES_BY_MODEL_FILENAME}' in model artifact: expected object.")
        for key, value in raw_mapping.items():
            if isinstance(key, str) and isinstance(value, list) and all(isinstance(col, str) and col for col in value):
                features_by_model[key] = value
    elif not cfg.allow_features_by_model_missing:
        raise RuntimeError(
            f"Model artifact is missing required file '{FEATURES_BY_MODEL_FILENAME}'. "
            "Set ALLOW_FEATURES_BY_MODEL_MISSING=true only for legacy artifacts."
        )
    if not features_by_model:
        features_by_model = {model_filename: feature_cols for model_filename in model_filenames}

    post_cfg = PostprocessConfig.from_json(postprocess_path)
    models = [lgb.Booster(model_file=str(root / model_filename)) for model_filename in model_filenames]
    logger.info(
        "phase=artifact_downloaded artifact_ref=%s selected_dataset_version=%s n_features=%d n_models=%d",
        artifact_ref,
        manifest.get("dataset_version", "unknown"),
        len(feature_cols),
        len(models),
    )
    logger.info(
        "phase=postprocess_loaded schema_version=%d blend_alpha=%.4f bench_prop=%.4f feat_neut=%.4f",
        post_cfg.schema_version,
        post_cfg.blend_alpha,
        post_cfg.bench_neutralize_prop,
        post_cfg.feature_neutralize_prop,
    )
    return models, feature_cols, features_by_model, manifest, post_cfg


def load_prod_model(
    cfg: InferenceRuntimeConfig,
) -> tuple[list[lgb.Booster], list[str], dict[str, list[str]], dict, PostprocessConfig]:
    ref = f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_model_name}:prod"
    api = wandb.Api()
    artifact = api.artifact(ref, type="model")
    root = Path(artifact.download(root="artifacts"))
    return _load_artifact_from_root(root, cfg, ref)


def _download_live_benchmark_dataset(napi: NumerAPI, dataset_version: str, out_path: Path) -> Path:
    prefix = dataset_version.lower() + "/"
    for dataset in napi.list_datasets():
        lowered = dataset.lower()
        if lowered.startswith(prefix) and all(token in lowered for token in ("live", "benchmark", "models")):
            napi.download_dataset(dataset, str(out_path))
            return out_path
    raise DriftGuardError(f"Could not find live benchmark models dataset under {dataset_version}/")


def _download_live_dataset(napi: NumerAPI, dataset_version: str, out_path: Path, use_int8_parquet: bool) -> Path:
    datasets = napi.list_datasets()
    prefix = dataset_version.lower() + "/"
    live_parquets = [
        dataset
        for dataset in datasets
        if dataset.lower().startswith(prefix) and dataset.lower().endswith(".parquet") and "live" in dataset.lower()
    ]
    selected = next((dataset for dataset in live_parquets if "int8" in dataset.lower()), None) if use_int8_parquet else None
    if selected is None:
        selected = next((dataset for dataset in live_parquets if "int8" not in dataset.lower()), None)
    if selected is None:
        selected = f"{dataset_version}/live.parquet"
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


def _build_mock_artifact_dir(root: Path, dataset_version: str, model_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"feature_{idx:02d}" for idx in range(6)]
    rng = np.random.default_rng(9)
    x = pd.DataFrame(rng.normal(size=(90, len(feature_cols))), columns=feature_cols)
    y = (0.6 * x[feature_cols[0]] - 0.2 * x[feature_cols[1]] + rng.normal(0.0, 0.02, len(x))).astype(np.float32)
    booster = lgb.train(
        params={"objective": "regression", "metric": "rmse", "learning_rate": 0.1, "num_leaves": 16, "verbosity": -1},
        train_set=lgb.Dataset(x, label=y, feature_name=feature_cols),
        num_boost_round=20,
    )
    model_file = f"{model_name}_dry_run.txt"
    booster.save_model(str(root / model_file))
    (root / FEATURES_FILENAME).write_text(json.dumps(feature_cols, indent=2))
    (root / FEATURES_UNION_FILENAME).write_text(json.dumps(feature_cols, indent=2))
    (root / FEATURES_BY_MODEL_FILENAME).write_text(json.dumps({model_file: feature_cols}, indent=2))
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "feature_set": "medium",
                "artifact_schema_version": 3,
                "model_file": model_file,
                "model_files": [model_file],
                "features_union_file": FEATURES_UNION_FILENAME,
                "features_by_model_file": FEATURES_BY_MODEL_FILENAME,
            },
            indent=2,
        )
    )
    (root / POSTPROCESS_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "submission_transform": "rank_01",
                "blend_alpha": 0.7,
                "bench_neutralize_prop": 0.3,
                "payout_weight_corr": 0.75,
                "payout_weight_bmc": 2.25,
                "bench_cols_used": ["benchmark_1", "benchmark_2"],
                "feature_neutralize_prop": 0.0,
                "feature_neutralize_n_features": 0,
                "feature_neutralize_seed": 0,
            },
            indent=2,
        )
    )


def inference_dry_run() -> int:
    dataset_version = os.getenv("NUMERAI_DATASET_VERSION", "v5.2").strip() or "v5.2"
    cfg = InferenceRuntimeConfig(
        numerai_public_id="dry-run",
        numerai_secret_key="dry-run",
        numerai_model_name="dry-run",
        wandb_entity="dry-run",
        wandb_project="dry-run",
        dataset_version=dataset_version,
        wandb_model_name="dry_run_model",
        min_pred_std=1e-8,
        max_abs_exposure=1.0,
        exposure_sample_rows=1000,
    )
    artifact_root = Path("artifacts") / "mock_prod"
    _build_mock_artifact_dir(artifact_root, cfg.dataset_version, cfg.wandb_model_name)
    models, feature_cols, features_by_model, manifest, post_cfg = _load_artifact_from_root(
        artifact_root, cfg, "local/mock:dry-run"
    )

    rng = np.random.default_rng(13)
    rows = 120
    live_df = pd.DataFrame(rng.normal(size=(rows, len(feature_cols))), columns=feature_cols)
    live_df.insert(0, "id", [f"id_{idx:04d}" for idx in range(rows)])
    live_df.insert(1, "era", [f"era{(idx // 15) + 1}" for idx in range(rows)])
    live_path = Path("live.parquet")
    live_df.to_parquet(live_path, index=False)
    live_id = live_df["id"].to_numpy()
    bench_df = pd.DataFrame(
        {
            "id": live_id,
            "benchmark_1": rng.normal(size=rows),
            "benchmark_2": rng.normal(size=rows),
        }
    )
    live_bench_path = Path("live_benchmark_models.parquet")
    bench_df.to_parquet(live_bench_path, index=False)

    model_files = manifest.get("model_files", [f"{cfg.wandb_model_name}_dry_run.txt"])
    model_preds = []
    for model, model_file in zip(models, model_files):
        model_cols = _model_feature_cols(str(model_file), features_by_model, feature_cols)
        model_preds.append(model.predict(live_df[model_cols]).astype(np.float32))
    bench_cols_used = list(post_cfg.bench_cols_used)
    bench_aligned = pd.read_parquet(live_bench_path, columns=["id", *bench_cols_used]).set_index("id").loc[live_id].to_numpy(
        dtype=np.float32, copy=False
    )
    pred_raw = np.mean(np.vstack(model_preds), axis=0, dtype=np.float32)
    pred_final = apply_postprocess(
        pred_raw=pred_raw,
        era=live_df["era"].to_numpy(),
        cfg=post_cfg,
        bench=bench_aligned,
        features=None,
    )
    apply_quality_gates(live_df[feature_cols], pred_final, cfg, submission_transform=post_cfg.submission_transform)
    submission = pd.DataFrame({"id": live_id, "prediction": pred_final})
    submission.to_csv("submission.csv", index=False)
    print(f"INFER_DRY_RUN_OK rows={len(submission)} manifest_dataset={manifest.get('dataset_version')}")
    return 0


def main() -> int:
    if _optional_bool_env("INFER_DRY_RUN", default=False):
        return inference_dry_run()
    cfg = InferenceRuntimeConfig.from_env()
    logger.info(
        "phase=config_loaded dataset_version=%s model_name=%s numerai_model_name=%s",
        cfg.dataset_version,
        cfg.wandb_model_name,
        cfg.numerai_model_name,
    )

    models, feature_cols, features_by_model, manifest, post_cfg = load_prod_model(cfg)
    model_files = manifest.get("model_files", [f"{cfg.wandb_model_name}.txt"])
    if len(model_files) != len(models):
        model_files = [f"{cfg.wandb_model_name}.txt" for _ in models]
    manifest_dataset_version = manifest.get("dataset_version")
    if manifest_dataset_version != cfg.dataset_version:
        message = (
            f"Dataset version mismatch: model manifest has {manifest_dataset_version!r}, "
            f"but runtime expects {cfg.dataset_version!r}."
        )
        if cfg.allow_dataset_version_mismatch:
            logger.warning("%s Proceeding due to ALLOW_DATASET_VERSION_MISMATCH=true.", message)
        else:
            raise RuntimeError(
                f"{message} Set ALLOW_DATASET_VERSION_MISMATCH=true only for intentional override."
            )

    napi = NumerAPI(public_id=cfg.numerai_public_id, secret_key=cfg.numerai_secret_key)
    live_path = Path("live.parquet")
    live_bench_path = Path("live_benchmark_models.parquet")
    _download_live_dataset(
        napi,
        cfg.dataset_version,
        live_path,
        use_int8_parquet=_optional_bool_env("USE_INT8_PARQUET", default=False),
    )
    _download_live_benchmark_dataset(napi, cfg.dataset_version, live_bench_path)
    logger.info("phase=datasets_downloaded dataset_version=%s live_path=%s", cfg.dataset_version, live_path)

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
        manifest.get("dataset_version", cfg.dataset_version),
    )
    missing = [c for c in feature_cols if c not in live_df.columns]
    if missing:
        raise DriftGuardError(f"Live data missing required features: {missing[:10]}")
    if "era" not in live_df.columns:
        raise DriftGuardError("Live data is missing required 'era' column.")

    model_preds = []
    for model, model_file in zip(models, model_files):
        model_cols = _model_feature_cols(str(model_file), features_by_model, feature_cols)
        missing_model_cols = [col for col in model_cols if col not in live_df.columns]
        if missing_model_cols:
            raise DriftGuardError(f"Live data missing model-specific features for {model_file}: {missing_model_cols[:10]}")
        model_preds.append(model.predict(live_df[model_cols]).astype(np.float32))

    if "id" in live_df.columns:
        live_id = live_df["id"].to_numpy()
    elif live_df.index.name == "id":
        live_id = live_df.index.to_numpy()
    else:
        raise DriftGuardError("Live data is missing required 'id' (expected as column or index).")

    bench_cols_used = list(post_cfg.bench_cols_used)
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

    logger.info(
        "phase=prediction_submitted submission_path=%s rows=%d model_id=%s ensemble_models=%d",
        submission_path,
        len(submission),
        model_id,
        len(models),
    )
    print(f"submission_id={submission_id}")
    if manifest:
        print(f"model_manifest={manifest}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        raise SystemExit(main())
    except DriftGuardError as exc:
        print(f"DRIFT_GUARD_ABORT: {exc}", file=sys.stderr)
        raise SystemExit(2)
