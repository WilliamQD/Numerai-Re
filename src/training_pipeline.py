from __future__ import annotations

import gc
import logging
from pathlib import Path

from bench_matrix_builder import BenchmarkAlignmentError
from config import TrainRuntimeConfig
from feature_sampling import features_hash
from training_artifact import save_and_log_artifact
from training_checkpoint import (
    checkpoint_dir as _checkpoint_dir,
    load_training_checkpoint as _load_training_checkpoint,
    member_features_key as _member_features_key,
    write_training_checkpoint as _write_training_checkpoint,
)
from training_runtime import (
    download_with_numerapi,
    fit_lgbm,
    fit_lgbm_final,
    init_wandb_run,
    load_feature_list,
    load_features_mapping,
    load_train_valid_frames,
    log_seed_observability,
    resolve_lgb_params,
    sample_features_by_seed,
    write_features_mapping,
)
from training_seed_runner import (
    hydrate_checkpoint_features,
    log_member_summary,
    run_seed_training_loop,
)
from training_tuning import WalkforwardReport, collect_blend_windows, evaluate_walkforward
from tune_blend import BlendTuneReport, tune_blend_on_windows

from artifact_contract import FEATURES_BY_MODEL_FILENAME


logger = logging.getLogger(__name__)


CHECKPOINT_FILENAME = "training_checkpoint.json"
ARTIFACT_SCHEMA_VERSION = 4


def train() -> None:
    cfg = TrainRuntimeConfig.from_env()
    logger.info(
        "phase=config_loaded dataset_version=%s feature_set=%s model_name=%s lgbm_device=%s seeds=%s",
        cfg.dataset_version,
        cfg.feature_set_name,
        cfg.model_name,
        cfg.lgbm_device,
        list(cfg.lgbm_seeds),
    )

    lgb_params = resolve_lgb_params(cfg)
    run = init_wandb_run(cfg, lgb_params)
    train_path, validation_path, features_path, benchmark_paths = download_with_numerapi(cfg, cfg.numerai_data_dir)
    feature_pool = load_feature_list(features_path, cfg.feature_set_name)
    logger.info(
        "phase=datasets_downloaded dataset_version=%s data_dir=%s n_features=%d",
        cfg.dataset_version,
        cfg.numerai_data_dir,
        len(feature_pool),
    )
    sampled_features_by_seed = sample_features_by_seed(cfg, feature_pool)
    base_seed = int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0])
    try:
        base_data = load_train_valid_frames(
            cfg,
            train_path=train_path,
            validation_path=validation_path,
            benchmark_paths=benchmark_paths,
            feature_cols=sampled_features_by_seed[base_seed],
        )
    except BenchmarkAlignmentError as first_exc:
        logger.warning(
            "phase=bench_alignment_retry reason=%s action=force_redownload",
            first_exc,
        )
        _, _, _, benchmark_paths = download_with_numerapi(
            cfg,
            cfg.numerai_data_dir,
            force_benchmark_redownload=True,
        )
        try:
            base_data = load_train_valid_frames(
                cfg,
                train_path=train_path,
                validation_path=validation_path,
                benchmark_paths=benchmark_paths,
                feature_cols=sampled_features_by_seed[base_seed],
            )
        except BenchmarkAlignmentError as second_exc:
            raise RuntimeError(
                "Benchmark alignment failed after one forced benchmark redownload. "
                f"initial_error={first_exc}; retry_error={second_exc}"
            ) from second_exc
    wf_report: WalkforwardReport | None = None
    blend_report: BlendTuneReport | None = None
    checkpoint_walkforward: dict[str, object] | None = None
    checkpoint_postprocess: dict[str, object] = {
        "schema_version": 1,
        "submission_transform": "rank_01",
        "blend_alpha": 1.0,
        "bench_neutralize_prop": 0.0,
        "payout_weight_corr": float(cfg.payout_weight_corr),
        "payout_weight_bmc": float(cfg.payout_weight_bmc),
        "bench_cols_used": base_data.bench_cols,
        "feature_neutralize_prop": 0.0,
        "feature_neutralize_n_features": 0,
        "feature_neutralize_seed": 0,
    }
    recommended_iter: int | None = None
    if cfg.walkforward_enabled:
        wf_report = evaluate_walkforward(cfg, lgb_params, base_data, logger=logger)
        recommended_iter = int(wf_report.recommended_num_iteration)
        blend_windows = collect_blend_windows(cfg, lgb_params, base_data, wf_report, logger=logger)
        blend_report = tune_blend_on_windows(
            windows=blend_windows,
            alpha_grid=cfg.blend_alpha_grid,
            prop_grid=cfg.bench_neutralize_prop_grid,
            payout_weight_corr=float(cfg.payout_weight_corr),
            payout_weight_bmc=float(cfg.payout_weight_bmc),
        )
        checkpoint_walkforward = {
            "enabled": True,
            "chunk_size": int(cfg.walkforward_chunk_size),
            "purge_eras": int(cfg.walkforward_purge_eras),
            "max_windows": int(cfg.walkforward_max_windows),
            "tune_seed": int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0]),
            "recommended_num_iteration": recommended_iter,
        }
        checkpoint_postprocess = {
            "schema_version": 1,
            "submission_transform": "rank_01",
            "blend_alpha": float(blend_report.best_alpha),
            "bench_neutralize_prop": float(blend_report.best_prop),
            "payout_weight_corr": float(cfg.payout_weight_corr),
            "payout_weight_bmc": float(cfg.payout_weight_bmc),
            "bench_cols_used": base_data.bench_cols,
            "feature_neutralize_prop": 0.0,
            "feature_neutralize_n_features": 0,
            "feature_neutralize_seed": 0,
        }
        logger.info(
            "phase=walkforward_completed recommended_num_iteration=%d mean_corr=%.6f std_corr=%.6f sharpe=%.6f hit_rate=%.6f",
            recommended_iter,
            wf_report.mean_corr,
            wf_report.std_corr,
            wf_report.sharpe,
            wf_report.hit_rate,
        )
    del base_data
    gc.collect()

    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    features_by_model_path = checkpoint_dir / FEATURES_BY_MODEL_FILENAME

    members = _load_training_checkpoint(
        checkpoint_path,
        cfg,
        lgb_params,
        expected_walkforward=checkpoint_walkforward,
        expected_postprocess=checkpoint_postprocess,
    )
    completed_seeds = {int(member["seed"]) for member in members}
    features_by_model = load_features_mapping(features_by_model_path)
    if completed_seeds:
        logger.info(
            "phase=checkpoint_loaded checkpoint_path=%s completed_seeds=%s",
            checkpoint_path,
            sorted(completed_seeds),
        )

    features_by_model = hydrate_checkpoint_features(
        members,
        checkpoint_dir=checkpoint_dir,
        features_by_model=features_by_model,
        sampled_features_by_seed=sampled_features_by_seed,
        member_features_key_fn=_member_features_key,
    )
    write_features_mapping(features_by_model_path, features_by_model)

    members, features_by_model = run_seed_training_loop(
        cfg=cfg,
        lgb_params=lgb_params,
        members=members,
        features_by_model=features_by_model,
        sampled_features_by_seed=sampled_features_by_seed,
        benchmark_paths=benchmark_paths,
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        features_by_model_path=features_by_model_path,
        checkpoint_walkforward=checkpoint_walkforward,
        checkpoint_postprocess=checkpoint_postprocess,
        recommended_iter=recommended_iter,
        wf_report_mean_corr=float(wf_report.mean_corr) if wf_report is not None else None,
        load_train_valid_frames_fn=load_train_valid_frames,
        fit_lgbm_fn=fit_lgbm,
        fit_lgbm_final_fn=fit_lgbm_final,
        log_seed_observability_fn=log_seed_observability,
        feature_hash_fn=features_hash,
        write_features_mapping_fn=write_features_mapping,
        write_training_checkpoint_fn=_write_training_checkpoint,
        logger=logger,
    )

    if len(members) != len(cfg.lgbm_seeds):
        raise RuntimeError(
            f"Incomplete checkpoint state: expected {len(cfg.lgbm_seeds)} seeds, got {len(members)} members."
        )

    log_member_summary(members, features_by_model)

    save_and_log_artifact(
        cfg,
        run,
        lgb_params,
        feature_pool,
        features_by_model,
        members,
        checkpoint_dir,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        logger=logger,
        wf_report=wf_report,
        postprocess_config=checkpoint_postprocess,
    )
    run.finish()
