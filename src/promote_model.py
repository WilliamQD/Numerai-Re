"""Promote a trained candidate artifact to production after integrity checks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import wandb


FEATURES_FILENAME = "features.json"
MANIFEST_FILENAME = "train_manifest.json"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    entity = _required_env("WANDB_ENTITY")
    project = _required_env("WANDB_PROJECT")
    model_name = os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v52").strip() or "lgbm_numerai_v52"
    candidate_alias = os.getenv("WANDB_CANDIDATE_ALIAS", "candidate").strip() or "candidate"
    prod_alias = os.getenv("WANDB_PROD_ALIAS", "prod").strip() or "prod"
    expected_dataset_version = os.getenv("NUMERAI_DATASET_VERSION", "").strip() or None

    ref = f"{entity}/{project}/{model_name}:{candidate_alias}"
    api = wandb.Api()
    artifact = api.artifact(ref, type="model")
    root = Path(artifact.download(root="artifacts_promote"))

    features_path = root / FEATURES_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    if not features_path.exists():
        raise RuntimeError(f"Candidate artifact is missing required file '{FEATURES_FILENAME}'.")
    if not manifest_path.exists():
        raise RuntimeError(f"Candidate artifact is missing required file '{MANIFEST_FILENAME}'.")

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Invalid '{MANIFEST_FILENAME}': expected JSON object.")

    manifest_dataset_version = manifest.get("dataset_version")
    if expected_dataset_version and manifest_dataset_version != expected_dataset_version:
        raise RuntimeError(
            f"Dataset version mismatch: manifest has {manifest_dataset_version!r}, "
            f"expected {expected_dataset_version!r}."
        )

    model_files = manifest.get("model_files")
    if isinstance(model_files, list) and model_files and all(isinstance(name, str) and name.strip() for name in model_files):
        filenames = model_files
    else:
        model_file = manifest.get("model_file")
        if not isinstance(model_file, str) or not model_file.strip():
            raise RuntimeError("Manifest missing valid 'model_files' or 'model_file'.")
        filenames = [model_file]

    missing_models = [name for name in filenames if not (root / name).exists()]
    if missing_models:
        raise RuntimeError(f"Candidate artifact is missing model files: {missing_models}")

    run = wandb.init(project=project, entity=entity, job_type="promote")
    run.log_artifact(artifact, aliases=[prod_alias])
    run.finish()
    logger.info("phase=artifact_promoted artifact_ref=%s promoted_alias=%s", ref, prod_alias)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    raise SystemExit(main())
