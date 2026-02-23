from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from numerai_re.contracts.artifact_contract import (
    FEATURES_BY_MODEL_FILENAME,
    FEATURES_FILENAME,
    FEATURES_UNION_FILENAME,
    MANIFEST_FILENAME,
    POSTPROCESS_FILENAME,
)


@dataclass(frozen=True)
class SavedArtifact:
    model_paths: list[Path]
    features_path: Path
    manifest_path: Path


def save_and_log_artifact(
    cfg: Any,
    run: Any,
    lgb_params: dict[str, object],
    feature_cols: list[str],
    features_by_model: dict[str, list[str]],
    members: list[dict[str, object]],
    checkpoint_dir: Path,
    artifact_schema_version: int,
    logger: logging.Logger,
    wf_report: Any | None = None,
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
    features_legacy_out = out_dir / FEATURES_FILENAME
    features_union_out = out_dir / FEATURES_UNION_FILENAME
    features_by_model_out = out_dir / FEATURES_BY_MODEL_FILENAME
    manifest_out = out_dir / MANIFEST_FILENAME
    wf_windows_out = out_dir / "walkforward_windows.json"
    postprocess_out = out_dir / POSTPROCESS_FILENAME
    features_legacy_out.write_text(json.dumps(features_union, indent=2))
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
                "artifact_schema_version": artifact_schema_version,
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
    artifact.add_file(str(features_legacy_out), name=FEATURES_FILENAME)
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
    return SavedArtifact(model_paths=model_paths, features_path=features_legacy_out, manifest_path=manifest_out)
