"""Weekly inference + submission script for GitHub Actions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from numerapi import NumerAPI


from src.config import InferenceRuntimeConfig


class DriftGuardError(RuntimeError):
    """Raised when quality gates fail and submission must abort safely."""




def _parse_env_float(name: str, default: str) -> float:
    """Return an environment-backed float with a consistent validation error."""
    value = os.getenv(name, default).strip()
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {name} value {value!r}: must be a valid float"
        ) from exc


def load_prod_model() -> tuple[lgb.Booster, list[str], dict]:
    entity = _env("WANDB_ENTITY")
    project = _env("WANDB_PROJECT")
    model_name = os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v43")

    api = wandb.Api()
    ref = f"{entity}/{project}/{model_name}:prod"
    artifact = api.artifact(ref, type="model")
    root = Path(artifact.download(root="artifacts"))

    features_path = root / "features.json"
    manifest_path = root / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_filename = manifest.get("model_file", f"{model_name}.txt")
    model_path = root / model_filename

    if not model_path.exists() or not features_path.exists():
        raise RuntimeError(
            f"Model artifact is missing required files. "
            f"Expected model file '{model_filename}' and 'features.json'."
        )

    feature_cols = json.loads(features_path.read_text())
    model = lgb.Booster(model_file=str(model_path))
    return model, feature_cols, manifest


def apply_quality_gates(features_df: pd.DataFrame, preds: np.ndarray) -> None:
    """Abort inference when prediction quality or risk limits are violated.

    MIN_PRED_STD defines the minimum required prediction standard deviation so
    submissions retain enough dispersion to be actionable.

    MAX_ABS_EXPOSURE defines the maximum allowed absolute feature exposure,
    limiting unintended concentration to any single input feature.
    """
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise DriftGuardError("Predictions contain NaN/Inf")
    if np.allclose(preds, 0.0):
        raise DriftGuardError("Predictions are all zero")

    pred_std = float(np.std(preds))
    min_std = _parse_env_float("MIN_PRED_STD", "1e-6")
    if pred_std < min_std:
        raise DriftGuardError(f"Prediction std ({pred_std:.8f}) below threshold {min_std}")

    max_abs_exposure = _parse_env_float("MAX_ABS_EXPOSURE", "0.30")
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

    napi = NumerAPI(public_id=cfg.numerai_public_id, secret_key=cfg.numerai_secret_key)

    live_path = Path("live.parquet")
    napi.download_dataset(f"{cfg.dataset_version}/live.parquet", str(live_path))

    model, feature_cols, manifest = load_prod_model(cfg)

    live_df = pd.read_parquet(live_path)
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

    print(f"submission_id={submission_id}")
    if manifest:
        print(f"model_manifest={manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DriftGuardError as exc:
        print(f"DRIFT_GUARD_ABORT: {exc}", file=sys.stderr)
        raise SystemExit(2)
