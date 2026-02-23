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

from artifact_contract import (
    FEATURES_BY_MODEL_FILENAME,
    FEATURES_FILENAME,
    FEATURES_UNION_FILENAME,
    MANIFEST_FILENAME,
    POSTPROCESS_FILENAME,
    load_features_by_model,
    load_manifest,
    load_union_features,
    resolve_model_files,
    validate_dataset_version,
    validate_model_files_exist,
)
from config import InferenceRuntimeConfig, _optional_bool_env
from inference_runtime import (
    RANK_01_EPSILON,
    DriftGuardError,
    _download_live_benchmark_dataset,
    _download_live_dataset,
    _model_feature_cols,
    apply_quality_gates,
    run_live_inference,
)
from postprocess import PostprocessConfig, apply_postprocess


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


logger = logging.getLogger(__name__)


def _load_artifact_from_root(
    root: Path, cfg: InferenceRuntimeConfig, artifact_ref: str
) -> tuple[list[lgb.Booster], list[str], dict[str, list[str]], dict, PostprocessConfig]:
    manifest = load_manifest(root, label=artifact_ref)
    model_filenames = resolve_model_files(manifest, label=artifact_ref)
    validate_model_files_exist(root, model_filenames, label=artifact_ref)
    feature_cols = load_union_features(root, manifest, label=artifact_ref)
    features_by_model = load_features_by_model(
        root,
        manifest,
        model_filenames,
        label=artifact_ref,
    )

    if not (root / POSTPROCESS_FILENAME).exists():
        raise RuntimeError(f"Model artifact is missing required file '{POSTPROCESS_FILENAME}'.")

    postprocess_path = root / POSTPROCESS_FILENAME

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
                "artifact_schema_version": 4,
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
    validate_dataset_version(manifest, cfg.dataset_version, label="inference runtime")

    submission_path, submission, model_id, submission_id = run_live_inference(
        cfg,
        models,
        feature_cols,
        features_by_model,
        manifest,
        post_cfg,
        use_int8_parquet=_optional_bool_env("USE_INT8_PARQUET", default=False),
    )

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
