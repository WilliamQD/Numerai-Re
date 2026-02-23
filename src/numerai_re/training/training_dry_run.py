from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from numerai_re.contracts.artifact_contract import (
    FEATURES_BY_MODEL_FILENAME,
    FEATURES_FILENAME,
    FEATURES_UNION_FILENAME,
    MANIFEST_FILENAME,
    POSTPROCESS_FILENAME,
)
from numerai_re.runtime.config import TrainRuntimeConfig
from numerai_re.common.era_utils import era_to_int
from numerai_re.training.training_runtime import LoadedData, fit_lgbm, load_feature_list, resolve_lgb_params
from numerai_re.training.walkforward import build_windows


ARTIFACT_SCHEMA_VERSION = 4


def _build_synthetic_dry_run_data(cfg: TrainRuntimeConfig, scratch_dir: Path) -> LoadedData:
    rng = np.random.default_rng(7)
    feature_cols = [f"feature_{idx:02d}" for idx in range(8)]
    features_path = scratch_dir / FEATURES_FILENAME
    features_path.write_text(json.dumps({"feature_sets": {"medium": feature_cols}}, indent=2))
    feature_cols = load_feature_list(features_path, cfg.feature_set_name)

    n_train, n_valid = 120, 60
    total_rows = n_train + n_valid
    eras = np.array([f"era{(idx // 10) + 1}" for idx in range(total_rows)], dtype=object)
    ids = np.array([f"dry_{idx:04d}" for idx in range(total_rows)], dtype=object)
    x_all_np = rng.normal(size=(total_rows, len(feature_cols))).astype(np.float32)
    y_all = (
        (0.55 * x_all_np[:, 0]) - (0.25 * x_all_np[:, 1]) + (0.10 * x_all_np[:, 2]) + rng.normal(0.0, 0.03, total_rows)
    ).astype(np.float32)
    bench_all = np.stack(
        (
            x_all_np[:, 0] + rng.normal(0.0, 0.02, total_rows),
            x_all_np[:, 1] + rng.normal(0.0, 0.02, total_rows),
        ),
        axis=1,
    ).astype(np.float32)
    bench_cols = ["benchmark_1", "benchmark_2"]
    x_all = x_all_np
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    era_train = eras[:n_train]
    id_train = ids[:n_train]
    bench_train = bench_all[:n_train]

    x_valid = x_all[n_train:]
    y_valid = y_all[n_train:]
    era_valid = eras[n_train:]
    id_valid = ids[n_train:]
    bench_valid = bench_all[n_train:]

    era_all_int = era_to_int(eras)
    return LoadedData(
        feature_cols=feature_cols,
        x_train=x_train,
        y_train=y_train,
        era_train=era_train,
        id_train=id_train,
        bench_train=bench_train,
        x_valid=x_valid,
        y_valid=y_valid,
        era_valid=era_valid,
        id_valid=id_valid,
        bench_valid=bench_valid,
        x_all=x_all,
        y_all=y_all,
        era_all=eras,
        era_all_int=era_all_int,
        id_all=ids,
        bench_all=bench_all,
        bench_cols=bench_cols,
    )


def train_dry_run() -> None:
    cfg = TrainRuntimeConfig(
        dataset_version=os.getenv("NUMERAI_DATASET_VERSION", "v5.2").strip() or "v5.2",
        feature_set_name="medium",
        wandb_api_key="dry-run",
        lgbm_seeds=(42,),
        num_boost_round=30,
        early_stopping_rounds=10,
        walkforward_enabled=False,
    )
    lgb_params = resolve_lgb_params(cfg)
    scratch_dir = Path("artifacts") / "_dry_run"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    data = _build_synthetic_dry_run_data(cfg, scratch_dir)
    windows = build_windows(data.era_all_int, chunk_size=6, purge_eras=1)
    if not windows:
        raise RuntimeError("TRAIN_DRY_RUN failed to build walk-forward windows.")

    fit_result = fit_lgbm(cfg, lgb_params, data, seed=cfg.lgbm_seeds[0])
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    model_filename = f"{cfg.model_name}_seed{cfg.lgbm_seeds[0]}_dry_run.txt"
    model_path = out_dir / model_filename
    fit_result.model.save_model(str(model_path), num_iteration=fit_result.best_iteration)
    features_by_model = {model_filename: list(data.feature_cols)}

    post_cfg = {
        "schema_version": 1,
        "submission_transform": "rank_01",
        "blend_alpha": 0.7,
        "bench_neutralize_prop": 0.3,
        "payout_weight_corr": cfg.payout_weight_corr,
        "payout_weight_bmc": cfg.payout_weight_bmc,
        "bench_cols_used": list(data.bench_cols),
        "feature_neutralize_prop": 0.0,
        "feature_neutralize_n_features": 0,
        "feature_neutralize_seed": 0,
    }
    pred_raw = fit_result.model.predict(data.x_valid, num_iteration=fit_result.best_iteration).astype(np.float32, copy=False)
    if np.isnan(pred_raw).any() or np.isinf(pred_raw).any():
        raise RuntimeError("TRAIN_DRY_RUN produced invalid predictions (NaN/Inf).")

    (out_dir / FEATURES_FILENAME).write_text(json.dumps(data.feature_cols, indent=2))
    (out_dir / FEATURES_UNION_FILENAME).write_text(json.dumps(data.feature_cols, indent=2))
    (out_dir / FEATURES_BY_MODEL_FILENAME).write_text(json.dumps(features_by_model, indent=2))
    (out_dir / POSTPROCESS_FILENAME).write_text(
        json.dumps(post_cfg, indent=2)
    )
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "dataset_version": cfg.dataset_version,
                "feature_set": cfg.feature_set_name,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_name": cfg.model_name,
                "model_file": model_filename,
                "model_files": [model_filename],
                "features_union_file": FEATURES_UNION_FILENAME,
                "features_by_model_file": FEATURES_BY_MODEL_FILENAME,
                "max_features_per_model": cfg.max_features_per_model,
                "sampling_strategy": cfg.feature_sampling_strategy,
                "sampling_master_seed": cfg.feature_sampling_master_seed,
            },
            indent=2,
        )
    )
    print(f"TRAIN_DRY_RUN_OK windows={len(windows)} artifact_dir={out_dir}")
