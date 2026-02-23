from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import wandb
from numerapi import NumerAPI

from numerai_re.contracts.artifact_contract import FEATURES_FILENAME
from numerai_re.data.benchmarks import download_benchmark_parquets
from numerai_re.runtime.config import TrainRuntimeConfig
from numerai_re.data.data_loading import load_split_numpy
from numerai_re.common.era_utils import era_to_int
from numerai_re.features.feature_sampling import sample_features_for_seed
from numerai_re.data.numerapi_datasets import resolve_split_parquet
from numerai_re.data.train_benchmark_data import load_and_align_benchmarks


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedData:
	feature_cols: list[str]
	x_train: Any
	y_train: np.ndarray
	era_train: np.ndarray
	id_train: np.ndarray
	bench_train: np.ndarray
	x_valid: Any
	y_valid: np.ndarray
	era_valid: np.ndarray
	id_valid: np.ndarray
	bench_valid: np.ndarray
	x_all: Any
	y_all: np.ndarray
	era_all: np.ndarray
	era_all_int: np.ndarray
	id_all: np.ndarray
	bench_all: np.ndarray
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
) -> tuple[Path, Path, Path, dict[str, Path]]:
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
	train_dataset = resolve_split_parquet(
		datasets,
		cfg.dataset_version,
		("train",),
		use_int8=cfg.use_int8_parquet,
	)
	validation_dataset = resolve_split_parquet(
		datasets,
		cfg.dataset_version,
		("validation",),
		use_int8=cfg.use_int8_parquet,
	)
	if cfg.use_int8_parquet and "int8" not in train_dataset.lower():
		logger.warning(
			"phase=int8_dataset_fallback split=train dataset_version=%s reason=int8_not_found",
			cfg.dataset_version,
		)
	if cfg.use_int8_parquet and "int8" not in validation_dataset.lower():
		logger.warning(
			"phase=int8_dataset_fallback split=validation dataset_version=%s reason=int8_not_found",
			cfg.dataset_version,
		)

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

	benchmark_paths = download_benchmark_parquets(
		napi,
		cfg.dataset_version,
		version_data_dir / "benchmarks",
		force_redownload=force_benchmark_redownload,
	)
	return train_path, validation_path, features_path, benchmark_paths


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
) -> LoadedData:
	logger.info(
		"phase=feature_subset_loading dataset_version=%s data_dir=%s n_features=%d",
		cfg.dataset_version,
		cfg.numerai_data_dir,
		len(feature_cols),
	)
	feature_dtype = np.int8 if cfg.use_int8_parquet else np.float32
	use_cache = cfg.load_mode == "cached"

	x_train, y_train, era_train, id_train = load_split_numpy(
		train_path,
		feature_cols,
		cfg.id_col,
		cfg.era_col,
		cfg.target_col,
		feature_dtype=feature_dtype,
		use_cache=use_cache,
	)
	x_valid, y_valid, era_valid, id_valid = load_split_numpy(
		validation_path,
		feature_cols,
		cfg.id_col,
		cfg.era_col,
		cfg.target_col,
		feature_dtype=feature_dtype,
		use_cache=use_cache,
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
		x_valid=x_valid,
		y_valid=y_valid,
		era_valid=era_valid,
		id_valid=id_valid,
		bench_valid=aligned_bench.valid,
		x_all=x_all,
		y_all=y_all,
		era_all=era_all,
		era_all_int=era_all_int,
		id_all=id_all,
		bench_all=aligned_bench.all,
		bench_cols=aligned_bench.cols,
	)


def fit_lgbm(cfg: TrainRuntimeConfig, lgb_params: dict[str, object], data: LoadedData, seed: int) -> FitResult:
	from numerai_re.training.training_tuning import best_corr_iteration as _best_corr_iteration
	from numerai_re.training.training_tuning import corr_scan_iterations as _corr_scan_iterations

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
) -> lgb.Booster:
	dtrain = lgb.Dataset(x, label=y, feature_name=feature_cols, free_raw_data=True)
	params = dict(lgb_params)
	params["seed"] = seed
	return lgb.train(params=params, train_set=dtrain, num_boost_round=num_boost_round)


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

