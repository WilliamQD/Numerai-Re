from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import wandb
from numerapi import NumerAPI

from numerai_re.contracts import FEATURES_FILENAME
from numerai_re.shared import RuntimeStatusReporter, era_to_int, sample_features_for_seed
from numerai_re.data.benchmarks import download_benchmark_parquets
from numerai_re.config import TrainRuntimeConfig
from numerai_re.data.loading import load_split_numpy
from numerai_re.data.numerapi_datasets import SplitParquetResolution, resolve_split_parquet_with_report
from numerai_re.data.benchmarks import load_and_align_benchmarks


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
	feature_cols: list[str]
	x_train: Any
	y_train: np.ndarray
	era_train: np.ndarray
	id_train: np.ndarray
	bench_train: np.ndarray
	bench_train_mask: np.ndarray
	x_valid: Any
	y_valid: np.ndarray
	era_valid: np.ndarray
	id_valid: np.ndarray
	bench_valid: np.ndarray
	bench_valid_mask: np.ndarray
	x_all: Any
	y_all: np.ndarray
	era_all: np.ndarray
	era_all_int: np.ndarray
	id_all: np.ndarray
	bench_all: np.ndarray
	bench_all_mask: np.ndarray
	bench_cols: list[str]


@dataclass(frozen=True)
class FitResult:
	seed: int
	model: lgb.Booster
	best_iteration: int
	best_valid_rmse: float
	best_valid_corr: float
	train_rmse_curve: list[float]
	valid_rmse_curve: list[float]
	valid_corr_curve: list[float]
	corr_scan_iters: list[int]


@dataclass(frozen=True)
class DatasetSelection:
	train_dataset: str
	validation_dataset: str
	train_is_int8: bool
	validation_is_int8: bool
	feature_dtype: type


def _log_split_resolution(split: str, resolution: SplitParquetResolution) -> None:
	candidate_preview = "|".join(resolution.candidates[:5]) if resolution.candidates else "<none>"
	logger.info(
		"phase=dataset_resolution split=%s requested_int8=%s selected=%s selected_is_int8=%s total_candidates=%d int8_candidates=%d candidates_preview=%s",
		split,
		resolution.use_int8_requested,
		resolution.selected,
		resolution.selected_is_int8,
		len(resolution.candidates),
		len(resolution.int8_candidates),
		candidate_preview,
	)
	if resolution.use_int8_requested and not resolution.int8_candidates:
		logger.debug(
			"phase=int8_named_variant_absent split=%s dataset_version=%s selected=%s candidate_count=%d candidates_preview=%s",
			split,
			resolution.dataset_version,
			resolution.selected,
			len(resolution.candidates),
			candidate_preview,
		)


def _is_integer_like_dtype(dtype: Any) -> bool:
	dtype_name = str(dtype).lower()
	return "int" in dtype_name


def _split_feature_dtype_name(parquet_path: Path) -> str:
	try:
		schema = pl.read_parquet_schema(str(parquet_path))
	except Exception:
		return "unknown"
	feature_dtypes = [dtype for column, dtype in schema.items() if column.startswith("feature")]
	if not feature_dtypes:
		return "unknown"
	if all(_is_integer_like_dtype(dtype) for dtype in feature_dtypes):
		return "int8"
	return "float32"


def _lgb_status_callback(
	status: RuntimeStatusReporter | None,
	phase: str,
	*,
	seed: int | None = None,
	window_id: int | None = None,
	window_total: int | None = None,
) -> Any:
	def _callback(env: Any) -> None:
		if status is None:
			return
		payload: dict[str, object] = {
			"iter": f"{int(env.iteration) + 1}/{int(env.end_iteration)}",
		}
		if seed is not None:
			payload["seed"] = seed
		if window_id is not None and window_total is not None:
			payload["window"] = f"{window_id}/{window_total}"
		status.update(phase, **payload)

	_callback.order = 0
	_callback.before_iteration = False
	return _callback


BASE_LGB_PARAMS = {
	"objective": "regression",
	"metric": "rmse",
	"learning_rate": 0.02,
	"num_leaves": 128,
	"feature_fraction": 0.7,
	"bagging_fraction": 0.8,
	"bagging_freq": 1,
	"min_data_in_leaf": 1000,
	"max_depth": -1,
	"verbosity": -1,
	"device": "cpu",
}


def resolve_lgb_params(cfg: TrainRuntimeConfig) -> dict[str, object]:
	params = dict(BASE_LGB_PARAMS)
	params.update(
		{
			"learning_rate": cfg.lgbm_learning_rate,
			"num_leaves": cfg.lgbm_num_leaves,
			"feature_fraction": cfg.lgbm_feature_fraction,
			"bagging_fraction": cfg.lgbm_bagging_fraction,
			"bagging_freq": cfg.lgbm_bagging_freq,
			"min_data_in_leaf": cfg.lgbm_min_data_in_leaf,
			"device": cfg.lgbm_device,
		}
	)
	if cfg.lgbm_device == "gpu":
		params["gpu_use_dp"] = False
	else:
		params.pop("gpu_use_dp", None)
	return params


def download_with_numerapi(
	cfg: TrainRuntimeConfig,
	data_dir: Path,
	*,
	force_benchmark_redownload: bool = False,
) -> tuple[Path, Path, Path, dict[str, Path], DatasetSelection]:
	version_data_dir = data_dir / cfg.dataset_version
	version_data_dir.mkdir(parents=True, exist_ok=True)
	numerapi_kwargs: dict[str, str] = {}
	if cfg.numerai_public_id and cfg.numerai_secret_key:
		numerapi_kwargs = {
			"public_id": cfg.numerai_public_id,
			"secret_key": cfg.numerai_secret_key,
		}
	napi = NumerAPI(**numerapi_kwargs)

	train_path = version_data_dir / "train.parquet"
	validation_path = version_data_dir / "validation.parquet"
	features_path = version_data_dir / FEATURES_FILENAME

	datasets = napi.list_datasets()
	train_resolution = resolve_split_parquet_with_report(
		datasets,
		cfg.dataset_version,
		("train",),
		use_int8=cfg.use_int8_parquet,
	)
	validation_resolution = resolve_split_parquet_with_report(
		datasets,
		cfg.dataset_version,
		("validation",),
		use_int8=cfg.use_int8_parquet,
	)
	_log_split_resolution("train", train_resolution)
	_log_split_resolution("validation", validation_resolution)

	train_dataset = train_resolution.selected
	validation_dataset = validation_resolution.selected

	required_files = (
		(train_dataset, train_path),
		(validation_dataset, validation_path),
		(f"{cfg.dataset_version}/{FEATURES_FILENAME}", features_path),
	)
	for dataset_path, local_path in required_files:
		if local_path.exists():
			logger.info("phase=dataset_reused path=%s", local_path)
			continue
		logger.info("phase=dataset_downloading dataset=%s path=%s", dataset_path, local_path)
		napi.download_dataset(dataset_path, str(local_path))

	train_dtype_name = _split_feature_dtype_name(train_path)
	validation_dtype_name = _split_feature_dtype_name(validation_path)
	use_int8_effective = (
		cfg.use_int8_parquet
		and train_dtype_name == "int8"
		and validation_dtype_name == "int8"
	)
	if cfg.use_int8_parquet and use_int8_effective:
		logger.info(
			"phase=int8_dtype_enabled reason=parquet_schema_detected train_feature_dtype=%s validation_feature_dtype=%s",
			train_dtype_name,
			validation_dtype_name,
		)
	elif cfg.use_int8_parquet:
		logger.info(
			"phase=int8_dtype_disabled reason=parquet_schema_not_int8 train_feature_dtype=%s validation_feature_dtype=%s",
			train_dtype_name,
			validation_dtype_name,
		)

	benchmark_paths = download_benchmark_parquets(
		napi,
		cfg.dataset_version,
		version_data_dir / "benchmarks",
		force_redownload=force_benchmark_redownload,
	)
	selection = DatasetSelection(
		train_dataset=train_dataset,
		validation_dataset=validation_dataset,
		train_is_int8=train_dtype_name == "int8",
		validation_is_int8=validation_dtype_name == "int8",
		feature_dtype=np.int8 if use_int8_effective else np.float32,
	)
	return train_path, validation_path, features_path, benchmark_paths, selection


def load_feature_list(features_path: Path, feature_set_name: str) -> list[str]:
	payload = json.loads(features_path.read_text())
	if not isinstance(payload, dict):
		raise ValueError(f"Unexpected JSON structure in features file {features_path}: expected object.")

	feature_sets = payload.get("feature_sets")
	if not isinstance(feature_sets, dict):
		raise ValueError(f'Missing or invalid "feature_sets" key in features file {features_path}.')

	if feature_set_name not in feature_sets:
		available = ", ".join(sorted(feature_sets.keys()))
		raise ValueError(f"Unknown feature set: {feature_set_name}. Available: {available}")

	feature_list = feature_sets[feature_set_name]
	if not isinstance(feature_list, list) or not all(isinstance(f, str) for f in feature_list):
		raise ValueError(f'Feature set "{feature_set_name}" in file {features_path} is invalid.')
	return feature_list


def write_features_mapping(path: Path, payload: dict[str, list[str]]) -> None:
	tmp_path = path.with_suffix(".tmp")
	tmp_path.write_text(json.dumps(payload, indent=2))
	tmp_path.replace(path)


def load_features_mapping(path: Path) -> dict[str, list[str]]:
	if not path.exists():
		return {}
	payload = json.loads(path.read_text())
	if not isinstance(payload, dict):
		raise RuntimeError(f"Invalid features mapping at {path}: expected object.")
	normalized: dict[str, list[str]] = {}
	for key, value in payload.items():
		if isinstance(key, str) and isinstance(value, list) and all(isinstance(col, str) for col in value):
			normalized[key] = value
	return normalized


def log_seed_observability(result: FitResult) -> None:
	seed_label = str(result.seed)
	wandb.log(
		{
			f"seed/{seed_label}/best_iteration": result.best_iteration,
			f"seed/{seed_label}/best_valid_rmse": result.best_valid_rmse,
			f"seed/{seed_label}/best_valid_corr": result.best_valid_corr,
			f"seed/{seed_label}/learning_curve": wandb.plot.line_series(
				xs=list(range(1, len(result.train_rmse_curve) + 1)),
				ys=[result.train_rmse_curve, result.valid_rmse_curve],
				keys=["train_rmse", "valid_rmse"],
				title=f"Seed {seed_label} RMSE",
				xname="iteration",
			),
			f"seed/{seed_label}/corr_curve": wandb.plot.line_series(
				xs=result.corr_scan_iters,
				ys=[result.valid_corr_curve],
				keys=["valid_corr_mean_per_era"],
				title=f"Seed {seed_label} Numerai CORR (scan)",
				xname="iteration",
			),
		}
	)


def init_wandb_run(cfg: TrainRuntimeConfig, lgb_params: dict[str, object]) -> Any:
	wandb.login()
	return wandb.init(
		project=cfg.wandb_project,
		entity=cfg.wandb_entity,
		job_type="train",
		tags=["numerai", cfg.dataset_version, "colab", "lightgbm", "ensemble"],
		config={
			"dataset_version": cfg.dataset_version,
			"feature_set": cfg.feature_set_name,
			"target_col": cfg.target_col,
			"era_col": cfg.era_col,
			"num_boost_round": cfg.num_boost_round,
			"early_stopping_rounds": cfg.early_stopping_rounds,
			"lgbm_seeds": list(cfg.lgbm_seeds),
			"corr_scan_period": cfg.corr_scan_period,
			"corr_scan_max_iters": cfg.corr_scan_max_iters,
			"select_best_by": cfg.select_best_by,
			"walkforward_enabled": cfg.walkforward_enabled,
			"walkforward_chunk_size": cfg.walkforward_chunk_size,
			"walkforward_purge_eras": cfg.walkforward_purge_eras,
			"walkforward_max_windows": cfg.walkforward_max_windows,
			"walkforward_tune_seed": cfg.walkforward_tune_seed,
			"payout_weight_corr": cfg.payout_weight_corr,
			"payout_weight_bmc": cfg.payout_weight_bmc,
			"blend_alpha_grid": list(cfg.blend_alpha_grid),
			"bench_neutralize_prop_grid": list(cfg.bench_neutralize_prop_grid),
			"blend_tune_seed": cfg.blend_tune_seed,
			"blend_use_windows": cfg.blend_use_windows,
			"max_features_per_model": cfg.max_features_per_model,
			"feature_sampling_strategy": cfg.feature_sampling_strategy,
			"feature_sampling_master_seed": cfg.feature_sampling_master_seed,
			"use_int8_parquet": cfg.use_int8_parquet,
			"load_backend": cfg.load_backend,
			"load_mode": cfg.load_mode,
			**lgb_params,
		},
	)


def load_train_valid_frames(
	cfg: TrainRuntimeConfig,
	train_path: Path,
	validation_path: Path,
	benchmark_paths: dict[str, Path],
	feature_cols: list[str],
	*,
	feature_dtype_override: type | None = None,
	status: RuntimeStatusReporter | None = None,
) -> LoadedData:
	logger.info(
		"phase=feature_subset_loading dataset_version=%s data_dir=%s n_features=%d",
		cfg.dataset_version,
		cfg.numerai_data_dir,
		len(feature_cols),
	)
	feature_dtype = feature_dtype_override if feature_dtype_override is not None else (
		np.int8 if cfg.use_int8_parquet else np.float32
	)
	logger.info("phase=feature_dtype_selected dtype=%s", np.dtype(feature_dtype).name)
	use_cache = cfg.load_mode == "cached"
	load_start = time.monotonic()
	if status is not None:
		status.update("data_load", step="train_split", cache=use_cache, force=True)

	x_train, y_train, era_train, id_train = load_split_numpy(
		train_path,
		feature_cols,
		cfg.id_col,
		cfg.era_col,
		cfg.target_col,
		feature_dtype=feature_dtype,
		use_cache=use_cache,
	)
	if status is not None:
		status.update(
			"data_load",
			step="train_loaded",
			rows=int(len(y_train)),
			elapsed_s=f"{time.monotonic() - load_start:.1f}",
			force=True,
		)
	if status is not None:
		status.update("data_load", step="validation_split", cache=use_cache, force=True)
	x_valid, y_valid, era_valid, id_valid = load_split_numpy(
		validation_path,
		feature_cols,
		cfg.id_col,
		cfg.era_col,
		cfg.target_col,
		feature_dtype=feature_dtype,
		use_cache=use_cache,
	)
	if status is not None:
		status.update(
			"data_load",
			step="validation_loaded",
			rows=int(len(y_valid)),
			elapsed_s=f"{time.monotonic() - load_start:.1f}",
			force=True,
		)
	n_train = len(y_train)
	logger.info("phase=frame_loaded split=train rows=%d cols=%d", n_train, len(feature_cols) + 3)
	logger.info("phase=frame_loaded split=validation rows=%d cols=%d", len(y_valid), len(feature_cols) + 3)

	x_all = np.concatenate([x_train, x_valid], axis=0)
	y_all = np.concatenate([y_train, y_valid], axis=0)
	era_all = np.concatenate([era_train, era_valid], axis=0)
	id_all = np.concatenate([id_train, id_valid], axis=0)
	era_all_int = era_to_int(era_all)

	aligned_bench = load_and_align_benchmarks(
		id_train=id_train,
		id_valid=id_valid,
		id_all=id_all,
		benchmark_paths=benchmark_paths,
		id_col=cfg.id_col,
		drop_sparse_columns=cfg.bench_drop_sparse_columns,
		max_null_ratio_per_column=cfg.bench_max_null_ratio_per_column,
		min_benchmark_columns=cfg.bench_min_columns,
	)

	return LoadedData(
		feature_cols=feature_cols,
		x_train=x_train,
		y_train=y_train,
		era_train=era_train,
		id_train=id_train,
		bench_train=aligned_bench.train,
		bench_train_mask=aligned_bench.train_mask,
		x_valid=x_valid,
		y_valid=y_valid,
		era_valid=era_valid,
		id_valid=id_valid,
		bench_valid=aligned_bench.valid,
		bench_valid_mask=aligned_bench.valid_mask,
		x_all=x_all,
		y_all=y_all,
		era_all=era_all,
		era_all_int=era_all_int,
		id_all=id_all,
		bench_all=aligned_bench.all,
		bench_all_mask=aligned_bench.all_mask,
		bench_cols=aligned_bench.cols,
	)


def fit_lgbm(cfg: TrainRuntimeConfig, lgb_params: dict[str, object], data: LoadedData, seed: int) -> FitResult:
	from numerai_re.training.tuning import best_corr_iteration as _best_corr_iteration
	from numerai_re.training.tuning import corr_scan_iterations as _corr_scan_iterations

	dtrain = lgb.Dataset(data.x_train, label=data.y_train, feature_name=data.feature_cols, free_raw_data=True)
	dvalid = lgb.Dataset(
		data.x_valid,
		label=data.y_valid,
		reference=dtrain,
		feature_name=data.feature_cols,
		free_raw_data=True,
	)

	fit_params = dict(lgb_params)
	fit_params["seed"] = seed
	callbacks = [
		lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
		lgb.log_evaluation(100),
	]
	evals_result: dict[str, dict[str, list[float]]] = {}
	callbacks.append(lgb.record_evaluation(evals_result))

	try:
		model = lgb.train(
			params=fit_params,
			train_set=dtrain,
			num_boost_round=cfg.num_boost_round,
			valid_sets=[dtrain, dvalid],
			valid_names=["train", "valid"],
			callbacks=callbacks,
		)
	except lgb.basic.LightGBMError as exc:
		if fit_params.get("device") != "gpu" or "No OpenCL device found" not in str(exc):
			raise
		logger.warning("phase=lgbm_gpu_unavailable requested_device=gpu fallback_device=cpu reason=%s", exc)
		fit_params["device"] = "cpu"
		fit_params.pop("gpu_use_dp", None)
		evals_result = {}
		callbacks = [
			lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
			lgb.log_evaluation(100),
			lgb.record_evaluation(evals_result),
		]
		model = lgb.train(
			params=fit_params,
			train_set=dtrain,
			num_boost_round=cfg.num_boost_round,
			valid_sets=[dtrain, dvalid],
			valid_names=["train", "valid"],
			callbacks=callbacks,
		)
		lgb_params["device"] = "cpu"
		lgb_params.pop("gpu_use_dp", None)

	best_iter = int(model.best_iteration)
	best_rmse = float(evals_result["valid"]["rmse"][max(0, best_iter - 1)])
	train_curve = [float(v) for v in evals_result["train"]["rmse"]]
	valid_curve = [float(v) for v in evals_result["valid"]["rmse"]]
	max_iter = int(model.current_iteration())
	corr_scan_iters = _corr_scan_iterations(cfg, max_iter)
	best_corr, best_corr_iter, corr_curve = _best_corr_iteration(
		model=model,
		x_valid=data.x_valid,
		y_valid=data.y_valid,
		era_valid=data.era_valid,
		corr_scan_iters=corr_scan_iters,
	)

	selected_best_iter = best_corr_iter if cfg.select_best_by == "corr" else best_iter
	logger.info(
		"phase=model_trained seed=%d best_iteration_rmse=%d best_iteration_corr=%d selected_best_iteration=%d "
		"best_valid_rmse=%.6f best_valid_corr=%.6f select_best_by=%s",
		seed,
		best_iter,
		best_corr_iter,
		selected_best_iter,
		best_rmse,
		best_corr,
		cfg.select_best_by,
	)
	return FitResult(
		seed=seed,
		model=model,
		best_iteration=selected_best_iter,
		best_valid_rmse=best_rmse,
		best_valid_corr=best_corr,
		train_rmse_curve=train_curve,
		valid_rmse_curve=valid_curve,
		valid_corr_curve=corr_curve,
		corr_scan_iters=corr_scan_iters,
	)


def fit_lgbm_final(
	lgb_params: dict[str, object],
	x: Any,
	y: np.ndarray,
	feature_cols: list[str],
	seed: int,
	num_boost_round: int,
	*,
	status: RuntimeStatusReporter | None = None,
) -> lgb.Booster:
	dtrain = lgb.Dataset(x, label=y, feature_name=feature_cols, free_raw_data=True)
	params = dict(lgb_params)
	params["seed"] = seed
	callbacks: list[Any] = []
	if status is not None:
		callbacks.append(_lgb_status_callback(status, "seed_fit_final", seed=seed))
	return lgb.train(params=params, train_set=dtrain, num_boost_round=num_boost_round, callbacks=callbacks)


def sample_features_by_seed(cfg: TrainRuntimeConfig, feature_pool: list[str]) -> dict[int, list[str]]:
	return {
		int(seed): sample_features_for_seed(
			feature_pool=feature_pool,
			seed=int(seed),
			model_index=idx,
			n_models=len(cfg.lgbm_seeds),
			max_features_per_model=int(cfg.max_features_per_model),
			master_seed=int(cfg.feature_sampling_master_seed),
			strategy=cfg.feature_sampling_strategy,
		)
		for idx, seed in enumerate(cfg.lgbm_seeds)
	}


# --- Seed training loop ---


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
    feature_dtype: type,
    load_train_valid_frames_fn: Callable[..., Any],
    fit_lgbm_fn: Callable[..., Any],
    fit_lgbm_final_fn: Callable[..., Any],
    log_seed_observability_fn: Callable[[Any], None],
    feature_hash_fn: Callable[[list[str]], str],
    write_features_mapping_fn: Callable[[Path, dict[str, list[str]]], None],
    write_training_checkpoint_fn: Callable[..., None],
    logger: logging.Logger,
    status: RuntimeStatusReporter | None = None,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    completed_seeds = {int(member["seed"]) for member in members}

    for seed in cfg.lgbm_seeds:
        if status is not None:
            status.update(
                "seed_loop",
                seed=int(seed),
                completed=len(completed_seeds),
                total=len(cfg.lgbm_seeds),
                force=True,
            )
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
            feature_dtype_override=feature_dtype,
            status=status,
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
                status=status,
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