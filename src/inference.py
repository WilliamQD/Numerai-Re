"""Weekly inference + submission script for GitHub Actions."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from numerapi import NumerAPI

from config import InferenceRuntimeConfig


FEATURES_FILENAME = "features.json"
MANIFEST_FILENAME = "train_manifest.json"
REQUIRED_MANIFEST_KEYS = ("dataset_version", "feature_set")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


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


def load_prod_model(cfg: InferenceRuntimeConfig) -> tuple[list[lgb.Booster], list[str], dict]:
    ref = f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_model_name}:prod"

    api = wandb.Api()
    artifact = api.artifact(ref, type="model")
    root = Path(artifact.download(root="artifacts"))

    features_path = root / FEATURES_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    manifest, model_filenames = _resolve_manifest(manifest_path, cfg.wandb_model_name)

    if not features_path.exists():
        raise RuntimeError(f"Model artifact is missing required file '{FEATURES_FILENAME}'.")

    missing_models = [name for name in model_filenames if not (root / name).exists()]
    if missing_models:
        raise RuntimeError(f"Model artifact is missing model files: {missing_models}")

    feature_cols = json.loads(features_path.read_text())
    if not isinstance(feature_cols, list) or not feature_cols or not all(isinstance(col, str) and col for col in feature_cols):
        raise RuntimeError(f"Invalid '{FEATURES_FILENAME}' in model artifact: expected non-empty list of features.")

    models = [lgb.Booster(model_file=str(root / model_filename)) for model_filename in model_filenames]
    logger.info(
        "phase=artifact_downloaded artifact_ref=%s selected_dataset_version=%s n_features=%d n_models=%d",
        ref,
        manifest.get("dataset_version", "unknown"),
        len(feature_cols),
        len(models),
    )
    return models, feature_cols, manifest


def apply_quality_gates(features_df: pd.DataFrame, preds: np.ndarray, cfg: InferenceRuntimeConfig) -> None:
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise DriftGuardError("Predictions contain NaN/Inf")
    if np.allclose(preds, 0.0):
        raise DriftGuardError("Predictions are all zero")

    pred_std = float(np.std(preds))
    if pred_std < cfg.min_pred_std:
        raise DriftGuardError(f"Prediction std ({pred_std:.8f}) below threshold {cfg.min_pred_std}")

    exposures = features_df.corrwith(pd.Series(preds, index=features_df.index)).abs()
    feature_exposure = float(exposures.max(skipna=True))
    if np.isnan(feature_exposure):
        raise DriftGuardError("Feature exposure could not be computed (NaN).")
    if feature_exposure > cfg.max_abs_exposure:
        raise DriftGuardError(
            f"Max feature exposure {feature_exposure:.5f} exceeds threshold {cfg.max_abs_exposure:.5f}"
        )


def main() -> int:
    cfg = InferenceRuntimeConfig.from_env()
    logger.info(
        "phase=config_loaded dataset_version=%s model_name=%s numerai_model_name=%s",
        cfg.dataset_version,
        cfg.wandb_model_name,
        cfg.numerai_model_name,
    )

    napi = NumerAPI(public_id=cfg.numerai_public_id, secret_key=cfg.numerai_secret_key)

    live_path = Path("live.parquet")
    napi.download_dataset(f"{cfg.dataset_version}/live.parquet", str(live_path))
    logger.info("phase=datasets_downloaded dataset_version=%s live_path=%s", cfg.dataset_version, live_path)

    models, feature_cols, manifest = load_prod_model(cfg)

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

    x_live = live_df[feature_cols]
    model_preds = [model.predict(x_live).astype(np.float32) for model in models]
    preds = np.mean(np.vstack(model_preds), axis=0, dtype=np.float32)
    apply_quality_gates(x_live, preds, cfg)

    if "id" in live_df.columns:
        live_id = live_df["id"]
    elif live_df.index.name == "id":
        live_id = live_df.index
    else:
        raise DriftGuardError("Live data is missing required 'id' (expected as column or index).")

    submission = pd.DataFrame({"id": live_id, "prediction": preds})
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
