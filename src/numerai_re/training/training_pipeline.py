from __future__ import annotations

import gc
import logging
from pathlib import Path
from time import perf_counter

from numerai_re.common.status_reporter import RuntimeStatusReporter
from numerai_re.data.bench_matrix_builder import BenchmarkAlignmentError
from numerai_re.runtime.config import TrainRuntimeConfig
from numerai_re.features.feature_sampling import features_hash
from numerai_re.training.training_artifact import save_and_log_artifact
from numerai_re.training.training_checkpoint import (
    checkpoint_dir as _checkpoint_dir,
    load_training_checkpoint as _load_training_checkpoint,
    member_features_key as _member_features_key,
    write_training_checkpoint as _write_training_checkpoint,
)
from numerai_re.training.training_runtime import (
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
from numerai_re.training.training_seed_runner import (
    hydrate_checkpoint_features,
    log_member_summary,
    run_seed_training_loop,
)
from numerai_re.training.training_tuning import WalkforwardReport, collect_blend_windows, evaluate_walkforward
from numerai_re.training.tune_blend import BlendTuneReport, tune_blend_on_windows
from numerai_re.training.checkpoint_io import count_list_items

from numerai_re.contracts.artifact_contract import FEATURES_BY_MODEL_FILENAME


logger = logging.getLogger(__name__)


CHECKPOINT_FILENAME = "training_checkpoint.json"
ARTIFACT_SCHEMA_VERSION = 4


def train() -> None:
    total_start = perf_counter()
    phase_seconds: dict[str, float] = {}
    completed_phases = 0
    total_phases = 6

    def _record_phase(phase_name: str, started_at: float) -> None:
        nonlocal completed_phases
        elapsed = perf_counter() - started_at
        phase_seconds[phase_name] = phase_seconds.get(phase_name, 0.0) + elapsed
        logger.info("phase=train_timing phase_name=%s elapsed_seconds=%.2f", phase_name, elapsed)
        completed_phases += 1
        elapsed_total = perf_counter() - total_start
        projected_total = (elapsed_total / completed_phases) * total_phases
        eta_seconds = max(0.0, projected_total - elapsed_total)
        logger.info(
            "phase=train_eta_progress stage=%s completed=%d total=%d elapsed_seconds=%.2f projected_total_seconds=%.2f eta_seconds=%.2f elapsed_minutes=%.2f projected_total_minutes=%.2f eta_minutes=%.2f",
            phase_name,
            completed_phases,
            total_phases,
            elapsed_total,
            projected_total,
            eta_seconds,
            elapsed_total / 60.0,
            projected_total / 60.0,
            eta_seconds / 60.0,
        )

    cfg = TrainRuntimeConfig.from_env()
    total_phases = 6 if cfg.walkforward_enabled else 5
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
    status = RuntimeStatusReporter(logger=logger, interval_seconds=float(cfg.status_update_seconds), name="train")
    logger.info(
        "phase=runtime_io_config data_dir=%s load_mode=%s load_backend=%s status_update_seconds=%d",
        cfg.numerai_data_dir,
        cfg.load_mode,
        cfg.load_backend,
        cfg.status_update_seconds,
    )
    dataset_download_started = perf_counter()
    train_path, validation_path, features_path, benchmark_paths, dataset_selection = download_with_numerapi(
        cfg,
        cfg.numerai_data_dir,
    )
    _record_phase("dataset_download", dataset_download_started)
    logger.info(
        "phase=dataset_selection_summary requested_int8=%s train_selected=%s validation_selected=%s feature_dtype=%s",
        cfg.use_int8_parquet,
        dataset_selection.train_dataset,
        dataset_selection.validation_dataset,
        dataset_selection.feature_dtype.__name__,
    )
    feature_pool = load_feature_list(features_path, cfg.feature_set_name)
    logger.info(
        "phase=datasets_downloaded dataset_version=%s data_dir=%s n_features=%d",
        cfg.dataset_version,
        cfg.numerai_data_dir,
        len(feature_pool),
    )
    sampled_features_by_seed = sample_features_by_seed(cfg, feature_pool)
    base_seed = int(cfg.walkforward_tune_seed if cfg.walkforward_tune_seed is not None else cfg.lgbm_seeds[0])
    base_data_load_started = perf_counter()
    try:
        base_data = load_train_valid_frames(
            cfg,
            train_path=train_path,
            validation_path=validation_path,
            benchmark_paths=benchmark_paths,
            feature_cols=sampled_features_by_seed[base_seed],
            feature_dtype_override=dataset_selection.feature_dtype,
            status=status,
        )
    except BenchmarkAlignmentError as first_exc:
        logger.warning(
            "phase=bench_alignment_retry reason=%s action=force_redownload",
            first_exc,
        )
        _, _, _, benchmark_paths, _ = download_with_numerapi(
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
                feature_dtype_override=dataset_selection.feature_dtype,
                status=status,
            )
        except BenchmarkAlignmentError as second_exc:
            raise RuntimeError(
                "Benchmark alignment failed after one forced benchmark redownload. "
                f"initial_error={first_exc}; retry_error={second_exc}"
            ) from second_exc
    _record_phase("base_data_load", base_data_load_started)
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
    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    walkforward_checkpoint_path = checkpoint_dir / "walkforward_checkpoint.json"
    blend_windows_checkpoint_path = checkpoint_dir / "blend_windows_checkpoint.json"
    blend_tune_checkpoint_path = checkpoint_dir / "blend_tune_checkpoint.json"
    seed_checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    logger.info(
        "phase=resume_state_summary checkpoint_dir=%s walkforward_windows_restored=%d blend_windows_restored=%d blend_combos_restored=%d seed_models_restored=%d",
        checkpoint_dir,
        count_list_items(walkforward_checkpoint_path, "rows"),
        count_list_items(blend_windows_checkpoint_path, "windows"),
        count_list_items(blend_tune_checkpoint_path, "search_rows"),
        count_list_items(seed_checkpoint_path, "members"),
    )
    recommended_iter: int | None = None
    if cfg.walkforward_enabled:
        walkforward_started = perf_counter()
        wf_report = evaluate_walkforward(
            cfg,
            lgb_params,
            base_data,
            logger=logger,
            status=status,
            checkpoint_path=walkforward_checkpoint_path,
            resume_mode=cfg.walkforward_resume_mode,
        )
        recommended_iter = int(wf_report.recommended_num_iteration)
        blend_windows = collect_blend_windows(
            cfg,
            lgb_params,
            base_data,
            wf_report,
            logger=logger,
            status=status,
            checkpoint_dir=checkpoint_dir,
            resume_mode=cfg.blend_tune_resume_mode,
        )
        blend_report = tune_blend_on_windows(
            windows=blend_windows,
            alpha_grid=cfg.blend_alpha_grid,
            prop_grid=cfg.bench_neutralize_prop_grid,
            payout_weight_corr=float(cfg.payout_weight_corr),
            payout_weight_bmc=float(cfg.payout_weight_bmc),
            status=status,
            checkpoint_path=blend_tune_checkpoint_path,
            resume_mode=cfg.blend_tune_resume_mode,
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
        _record_phase("walkforward_and_blend_tune", walkforward_started)
    del base_data
    gc.collect()

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    features_by_model_path = checkpoint_dir / FEATURES_BY_MODEL_FILENAME

    checkpoint_prepare_started = perf_counter()
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
    _record_phase("checkpoint_prepare", checkpoint_prepare_started)

    seed_training_started = perf_counter()
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
        feature_dtype=dataset_selection.feature_dtype,
        load_train_valid_frames_fn=load_train_valid_frames,
        fit_lgbm_fn=fit_lgbm,
        fit_lgbm_final_fn=fit_lgbm_final,
        log_seed_observability_fn=log_seed_observability,
        feature_hash_fn=features_hash,
        write_features_mapping_fn=write_features_mapping,
        write_training_checkpoint_fn=_write_training_checkpoint,
        logger=logger,
        status=status,
    )
    _record_phase("seed_training", seed_training_started)

    if len(members) != len(cfg.lgbm_seeds):
        raise RuntimeError(
            f"Incomplete checkpoint state: expected {len(cfg.lgbm_seeds)} seeds, got {len(members)} members."
        )

    log_member_summary(members, features_by_model)

    artifact_save_started = perf_counter()
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
    _record_phase("artifact_save", artifact_save_started)
    total_elapsed = perf_counter() - total_start
    summary = " ".join(f"{name}={seconds:.2f}s" for name, seconds in sorted(phase_seconds.items()))
    logger.info("phase=train_timing_summary total_seconds=%.2f %s", total_elapsed, summary)
    status.clear()
    run.finish()
