from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FEATURES_FILENAME = "features.json"
FEATURES_UNION_FILENAME = "features_union.json"
FEATURES_BY_MODEL_FILENAME = "features_by_model.json"
MANIFEST_FILENAME = "train_manifest.json"
POSTPROCESS_FILENAME = "postprocess_config.json"
REQUIRED_MANIFEST_KEYS = ("dataset_version", "feature_set", "artifact_schema_version")
REQUIRED_POSTPROCESS_KEYS = {
    "schema_version",
    "submission_transform",
    "blend_alpha",
    "bench_neutralize_prop",
    "payout_weight_corr",
    "payout_weight_bmc",
    "bench_cols_used",
}


def _read_json(path: Path, *, label: str) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing required file '{path.name}' for {label}.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in '{path.name}' for {label}: {exc}.") from exc


def load_manifest(root: Path, *, label: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_FILENAME
    payload = _read_json(manifest_path, label=label)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid '{MANIFEST_FILENAME}' for {label}: expected JSON object.")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"Invalid '{MANIFEST_FILENAME}' for {label}: missing keys {missing}.")
    return payload


def resolve_model_files(manifest: dict[str, Any], *, label: str) -> list[str]:
    model_files = manifest.get("model_files")
    if not isinstance(model_files, list) or not model_files:
        raise RuntimeError(f"Invalid manifest for {label}: 'model_files' must be a non-empty list[str].")
    filenames = [name.strip() for name in model_files if isinstance(name, str) and name.strip()]
    if len(filenames) != len(model_files):
        raise RuntimeError(f"Invalid manifest for {label}: 'model_files' contains empty/non-string entries.")
    return filenames


def load_union_features(root: Path, manifest: dict[str, Any], *, label: str) -> list[str]:
    union_file = manifest.get("features_union_file")
    if not isinstance(union_file, str) or not union_file.strip():
        raise RuntimeError(f"Invalid manifest for {label}: missing non-empty 'features_union_file'.")
    payload = _read_json(root / union_file, label=label)
    if not isinstance(payload, list) or not payload or not all(isinstance(col, str) and col for col in payload):
        raise RuntimeError(f"Invalid '{union_file}' for {label}: expected non-empty list[str].")
    return payload


def load_features_by_model(
    root: Path,
    manifest: dict[str, Any],
    model_files: list[str],
    *,
    label: str,
) -> dict[str, list[str]]:
    by_model_file = manifest.get("features_by_model_file")
    if not isinstance(by_model_file, str) or not by_model_file.strip():
        raise RuntimeError(f"Invalid manifest for {label}: missing non-empty 'features_by_model_file'.")

    file_path = root / by_model_file
    if not file_path.exists():
        raise RuntimeError(f"Missing required file '{by_model_file}' for {label}.")

    payload = _read_json(file_path, label=label)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid '{by_model_file}' for {label}: expected JSON object.")

    normalized: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, list) and all(isinstance(col, str) and col for col in value):
            normalized[key] = value

    if not normalized:
        raise RuntimeError(f"Invalid '{by_model_file}' for {label}: no valid model->feature mapping entries.")
    return normalized


def validate_model_files_exist(root: Path, model_files: list[str], *, label: str) -> None:
    missing = [name for name in model_files if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing model files for {label}: {missing}")


def validate_dataset_version(manifest: dict[str, Any], expected: str, *, label: str) -> None:
    manifest_version = manifest.get("dataset_version")
    if manifest_version != expected:
        raise RuntimeError(
            f"Dataset version mismatch for {label}: manifest has {manifest_version!r}, expected {expected!r}."
        )
