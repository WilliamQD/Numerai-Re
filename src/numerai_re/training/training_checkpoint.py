from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def member_features_key(member: dict[str, object]) -> str:
    features_key = member.get("features_key")
    if isinstance(features_key, str) and features_key:
        return features_key
    return str(member["model_file"])


def checkpoint_dir(cfg: Any) -> Path:
    base = getattr(cfg, "training_checkpoint_dir", None)
    if base is None:
        base = Path(cfg.numerai_data_dir)
    return Path(base) / str(cfg.dataset_version) / "checkpoints" / str(cfg.model_name)


def write_training_checkpoint(
    checkpoint_path: Path,
    cfg: Any,
    lgb_params: dict[str, object],
    members: list[dict[str, object]],
    walkforward: dict[str, object] | None = None,
    postprocess: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "dataset_version": cfg.dataset_version,
        "feature_set": cfg.feature_set_name,
        "seeds": list(cfg.lgbm_seeds),
        "lgb_params": lgb_params,
        "max_features_per_model": cfg.max_features_per_model,
        "feature_sampling_strategy": cfg.feature_sampling_strategy,
        "feature_sampling_master_seed": cfg.feature_sampling_master_seed,
        "completed_seeds": [int(member["seed"]) for member in members],
        "members": members,
    }
    if walkforward is not None:
        payload["walkforward"] = walkforward
    if postprocess is not None:
        payload["postprocess"] = postprocess
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: Path,
    cfg: Any,
    lgb_params: dict[str, object],
    expected_walkforward: dict[str, object] | None = None,
    expected_postprocess: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if not checkpoint_path.exists():
        return []

    payload = json.loads(checkpoint_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload at {checkpoint_path}: expected JSON object.")

    expected_meta = {
        "dataset_version": cfg.dataset_version,
        "feature_set": cfg.feature_set_name,
        "seeds": list(cfg.lgbm_seeds),
        "lgb_params": lgb_params,
        "max_features_per_model": cfg.max_features_per_model,
        "feature_sampling_strategy": cfg.feature_sampling_strategy,
        "feature_sampling_master_seed": cfg.feature_sampling_master_seed,
    }
    for key, expected_value in expected_meta.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(
                f"Checkpoint mismatch for '{key}': got {payload.get(key)!r}, expected {expected_value!r}. "
                f"Delete {checkpoint_path} to retrain from scratch."
            )
    if expected_walkforward is not None and payload.get("walkforward") != expected_walkforward:
        raise RuntimeError(
            "Checkpoint mismatch for 'walkforward'. "
            f"Got {payload.get('walkforward')!r}, expected {expected_walkforward!r}. "
            f"Delete {checkpoint_path} to retrain from scratch."
        )
    if (
        expected_postprocess is not None
        and payload.get("postprocess") is not None
        and payload.get("postprocess") != expected_postprocess
    ):
        raise RuntimeError(
            "Checkpoint mismatch for 'postprocess'. "
            f"Got {payload.get('postprocess')!r}, expected {expected_postprocess!r}. "
            f"Delete {checkpoint_path} to retrain from scratch."
        )

    members = payload.get("members")
    if not isinstance(members, list):
        raise RuntimeError(f"Invalid checkpoint payload at {checkpoint_path}: expected 'members' list.")

    normalized_members: list[dict[str, object]] = []
    for member in members:
        if not isinstance(member, dict):
            raise RuntimeError(f"Invalid checkpoint member in {checkpoint_path}: expected object, got {type(member)!r}.")
        try:
            seed = int(member["seed"])
            model_file = str(member["model_file"])
            best_iteration = int(member["best_iteration"])
            best_valid_rmse = float(member["best_valid_rmse"])
            best_valid_corr = float(member.get("best_valid_corr", np.nan))
            corr_scan_period = int(member["corr_scan_period"]) if member.get("corr_scan_period") is not None else None
            train_mode = str(member["train_mode"]) if member.get("train_mode") is not None else None
            recommended_num_iteration = (
                int(member["recommended_num_iteration"]) if member.get("recommended_num_iteration") is not None else None
            )
            features_key = member_features_key(member)
            n_features_used = int(member.get("n_features_used", 0))
            member_features_hash = str(member.get("features_hash", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid checkpoint member schema in {checkpoint_path}: {member!r}") from exc
        if not model_file:
            raise RuntimeError(f"Invalid checkpoint member model_file in {checkpoint_path}: {member!r}")
        normalized_members.append(
            {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": best_iteration,
                "best_valid_rmse": best_valid_rmse,
                "best_valid_corr": best_valid_corr,
                "corr_scan_period": corr_scan_period,
                "train_mode": train_mode,
                "recommended_num_iteration": recommended_num_iteration,
                "features_key": features_key,
                "n_features_used": n_features_used,
                "features_hash": member_features_hash,
            }
        )
    return normalized_members
