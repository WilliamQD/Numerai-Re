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
REQUIRED_MANIFEST_KEYS = ("model_file", "dataset_version", "feature_set")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


class DriftGuardError(RuntimeError):
    """Raised when quality gates fail and submission must abort safely."""


def _resolve_manifest(manifest_path: Path, model_name: str) -> tuple[dict, str]:
    if not manifest_path.exists():
        logger.warning(
            "Manifest file '%s' is missing. Falling back to default model filename.",
            MANIFEST_FILENAME,
        )
        return {}, f"{model_name}.txt"

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "Manifest file '%s' is malformed JSON. Falling back to default model filename.",
            MANIFEST_FILENAME,
        )
        return {}, f"{model_name}.txt"
    if not isinstance(manifest, dict):
        logger.warning(
            "Manifest file '%s' is not a JSON object. Falling back to default model filename.",
            MANIFEST_FILENAME,
        )
        return {}, f"{model_name}.txt"

    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_keys:
        logger.warning(
            "Manifest file '%s' is missing required keys %s. Falling back to default model filename.",
            MANIFEST_FILENAME,
            missing_keys,
        )
        return manifest, f"{model_name}.txt"

    model_filename = manifest.get("model_file")
    if not isinstance(model_filename, str) or not model_filename.strip():
        logger.warning(
            "Manifest key 'model_file' is invalid. Falling back to default model filename.",
        )
        return manifest, f"{model_name}.txt"

    return manifest, model_filename


def load_prod_model(cfg: InferenceRuntimeConfig) -> tuple[lgb.Booster, list[str], dict]:
    ref = f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_model_name}:prod"

    api = wandb.Api()
    artifact = api.artifact(ref, type="model")
    root = Path(artifact.download(root="artifacts"))

    features_path = root / FEATURES_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    manifest, model_filename = _resolve_manifest(manifest_path, cfg.wandb_model_name)
    model_path = root / model_filename

    if not model_path.exists() or not features_path.exists():
        raise RuntimeError(
            f"Model artifact is missing required files. "
            f"Expected model file '{model_filename}' and '{FEATURES_FILENAME}'."
        )

    feature_cols = json.loads(features_path.read_text())
    model = lgb.Booster(model_file=str(model_path))
    logger.info(
        "phase=artifact_uploaded artifact_ref=%s selected_dataset_version=%s n_features=%d",
        ref,
        manifest.get("dataset_version", "unknown"),
        len(feature_cols),
    )
    return model, feature_cols, manifest


def apply_quality_gates(features_df: pd.DataFrame, preds: np.ndarray, cfg: InferenceRuntimeConfig) -> None:
    """Abort inference when prediction quality or risk limits are violated."""
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

    model, feature_cols, manifest = load_prod_model(cfg)

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
    preds = model.predict(x_live).astype(np.float32)
    apply_quality_gates(x_live, preds, cfg)

    if "id" not in live_df.columns:
        raise DriftGuardError("Live data is missing required 'id' column.")
    submission = pd.DataFrame({"id": live_df["id"], "prediction": preds})
    submission_path = Path("submission.csv")
    submission.to_csv(submission_path, index=False)

    models = napi.get_models()
    if cfg.numerai_model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise RuntimeError(
            f"Configured NUMERAI_MODEL_NAME '{cfg.numerai_model_name}' not found among Numerai models. "
            f"Available models: {available or 'none'}"
        )
    model_id = models[cfg.numerai_model_name]
    submission_id = napi.upload_predictions(str(submission_path), model_id=model_id)

    logger.info(
        "phase=prediction_submitted submission_path=%s rows=%d model_id=%s",
        submission_path,
        len(submission),
        model_id,
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
