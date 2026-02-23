from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import wandb


def hydrate_checkpoint_features(
    members: list[dict[str, object]],
    *,
    checkpoint_dir: Path,
    features_by_model: dict[str, list[str]],
    sampled_features_by_seed: dict[int, list[str]],
    member_features_key_fn: Callable[[dict[str, object]], str],
) -> dict[str, list[str]]:
    for member in members:
        model_path = checkpoint_dir / str(member["model_file"])
        if not model_path.exists():
            raise RuntimeError(f"Checkpoint references missing model file: {model_path}")
        features_key = member_features_key_fn(member)
        if features_key not in features_by_model:
            seed = int(member["seed"])
            features_by_model[features_key] = sampled_features_by_seed[seed]
    return features_by_model


def run_seed_training_loop(
    *,
    cfg: Any,
    lgb_params: dict[str, object],
    members: list[dict[str, object]],
    features_by_model: dict[str, list[str]],
    sampled_features_by_seed: dict[int, list[str]],
    benchmark_paths: dict[str, Path],
    train_path: Path,
    validation_path: Path,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    features_by_model_path: Path,
    checkpoint_walkforward: dict[str, object] | None,
    checkpoint_postprocess: dict[str, object],
    recommended_iter: int | None,
    wf_report_mean_corr: float | None,
    load_train_valid_frames_fn: Callable[..., Any],
    fit_lgbm_fn: Callable[..., Any],
    fit_lgbm_final_fn: Callable[..., Any],
    log_seed_observability_fn: Callable[[Any], None],
    feature_hash_fn: Callable[[list[str]], str],
    write_features_mapping_fn: Callable[[Path, dict[str, list[str]]], None],
    write_training_checkpoint_fn: Callable[..., None],
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    completed_seeds = {int(member["seed"]) for member in members}

    for seed in cfg.lgbm_seeds:
        if seed in completed_seeds:
            logger.info("phase=seed_skipped_already_completed seed=%d", seed)
            continue

        seed_features = sampled_features_by_seed[int(seed)]
        data = load_train_valid_frames_fn(
            cfg,
            train_path=train_path,
            validation_path=validation_path,
            benchmark_paths=benchmark_paths,
            feature_cols=seed_features,
        )
        if cfg.walkforward_enabled:
            if recommended_iter is None:
                raise RuntimeError("Walk-forward is enabled but recommended_num_iteration is not available.")
            model_file = f"{cfg.model_name}_seed{seed}.txt"
            final_model = fit_lgbm_final_fn(
                lgb_params=lgb_params,
                x=data.x_all,
                y=data.y_all,
                feature_cols=seed_features,
                seed=seed,
                num_boost_round=recommended_iter,
            )
            final_model.save_model(str(checkpoint_dir / model_file), num_iteration=recommended_iter)
            member = {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": recommended_iter,
                "best_valid_rmse": float("nan"),
                "best_valid_corr": float(wf_report_mean_corr if wf_report_mean_corr is not None else np.nan),
                "corr_scan_period": cfg.corr_scan_period,
                "train_mode": "walkforward_final",
                "recommended_num_iteration": recommended_iter,
                "features_key": model_file,
                "n_features_used": len(seed_features),
                "features_hash": feature_hash_fn(seed_features),
            }
            wandb.log(
                {
                    f"seed/{seed}/best_iteration": recommended_iter,
                    f"seed/{seed}/train_mode": "walkforward_final",
                }
            )
        else:
            fit_result = fit_lgbm_fn(cfg, lgb_params, data, seed)
            model_file = f"{cfg.model_name}_seed{seed}.txt"
            fit_result.model.save_model(str(checkpoint_dir / model_file), num_iteration=fit_result.best_iteration)
            member = {
                "seed": seed,
                "model_file": model_file,
                "best_iteration": fit_result.best_iteration,
                "best_valid_rmse": fit_result.best_valid_rmse,
                "best_valid_corr": fit_result.best_valid_corr,
                "corr_scan_period": cfg.corr_scan_period,
                "features_key": model_file,
                "n_features_used": len(seed_features),
                "features_hash": feature_hash_fn(seed_features),
            }
            log_seed_observability_fn(fit_result)

        members.append(member)
        completed_seeds.add(seed)
        features_by_model[model_file] = seed_features
        write_features_mapping_fn(features_by_model_path, features_by_model)

        del data
        gc.collect()

        write_training_checkpoint_fn(
            checkpoint_path,
            cfg,
            lgb_params,
            members,
            walkforward=checkpoint_walkforward,
            postprocess=checkpoint_postprocess,
        )
        logger.info(
            "phase=seed_checkpoint_saved checkpoint_path=%s seed=%d completed=%d total=%d",
            checkpoint_path,
            seed,
            len(completed_seeds),
            len(cfg.lgbm_seeds),
        )

    return members, features_by_model


def log_member_summary(members: list[dict[str, object]], features_by_model: dict[str, list[str]]) -> None:
    summary_table = wandb.Table(columns=["seed", "best_iteration", "best_valid_rmse", "best_valid_corr", "model_file"])
    for member in members:
        summary_table.add_data(
            int(member["seed"]),
            int(member["best_iteration"]),
            float(member["best_valid_rmse"]),
            float(member.get("best_valid_corr", np.nan)),
            str(member["model_file"]),
        )
    valid_corr_values = [float(member.get("best_valid_corr", np.nan)) for member in members]
    best_corr_member = max(
        members,
        key=lambda member: float(member.get("best_valid_corr", float("-inf"))),
    )
    wandb.log(
        {
            "best_iteration_mean": float(np.mean([float(member["best_iteration"]) for member in members])),
            "best_valid_rmse_mean": float(np.mean([float(member["best_valid_rmse"]) for member in members])),
            "best_valid_corr_mean": float(np.nanmean(valid_corr_values)),
            "best_valid_corr_max": float(np.nanmax(valid_corr_values)),
            "best_valid_corr_best_seed": int(best_corr_member["seed"]),
            "n_models": len(members),
            "n_features_union": len({col for cols in features_by_model.values() for col in cols}),
            "ensemble_members": summary_table,
        }
    )
