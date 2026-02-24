# Environment Variable Reference

Canonical runtime parsers:
- `src/numerai_re/runtime/config.py` (`TrainRuntimeConfig.from_env`, `InferenceRuntimeConfig.from_env`)

## Required

| Variable | Scope | Purpose |
| --- | --- | --- |
| `WANDB_API_KEY` | training | Authenticate W&B for training logs/artifacts. |
| `NUMERAI_PUBLIC_ID` | inference | NumerAI API public key for live data/submission. |
| `NUMERAI_SECRET_KEY` | inference | NumerAI API secret key paired with public key. |
| `NUMERAI_MODEL_NAME` | inference | NumerAI model receiving submissions. |
| `WANDB_ENTITY` | inference | W&B entity owning artifacts. |
| `WANDB_PROJECT` | inference | W&B project containing model registry artifacts. |

## Shared Optional

| Variable | Default | Purpose |
| --- | --- | --- |
| `WANDB_MODEL_NAME` | `lgbm_numerai_v52` | W&B model registry name used by train/inference. |
| `USE_INT8_PARQUET` | `true` | Prefer int8 feature loading; runtime infers feature-column integer schema from parquet content instead of filename suffix. |
| `STATUS_UPDATE_SECONDS` | `60` | Long-phase status refresh cadence (single-line if interactive, compact log fallback otherwise). |
| `TRAIN_DRY_RUN` | `false` | Run synthetic training smoke path and print `TRAIN_DRY_RUN_OK`. |
| `INFER_DRY_RUN` | `false` | Run inference smoke path and print `INFER_DRY_RUN_OK`. |

## Training Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUMERAI_DATA_DIR` | environment-specific | Dataset/cache/checkpoint root path. |
| `NUMERAI_PUBLIC_ID` | unset | Optional authenticated download key for training data. |
| `NUMERAI_SECRET_KEY` | unset | Optional authenticated download secret for training data. |
| `NUMERAI_FEATURE_SET` | `all` | Feature set key from Numerai `features.json` (`small`/`medium`/`all`). |
| `WANDB_PROJECT` | `numerai-mlops` | Override training project. |
| `WANDB_ENTITY` | unset | Override training entity. |
| `LGBM_DEVICE` | `cpu` | LightGBM device (`cpu`/`gpu`, with safe fallback behavior). |
| `TRAIN_PROFILE` | `full` | Optional training preset for Colab entrypoint; `balanced` applies conservative speed-focused defaults via `setdefault`, `full` leaves env unchanged. |
| `RUN_LGBM_GPU_PROBE` | `false` | Run a tiny LightGBM CPU/GPU probe before training and auto-resolve `LGBM_DEVICE` to `gpu` on success, otherwise `cpu`. |
| `LGBM_GPU_PROBE_MIN_SPEEDUP` | `1.15` | Minimum GPU-vs-CPU probe speedup required to keep `LGBM_DEVICE=gpu`; otherwise runtime uses CPU for the run. |
| `LGBM_SEEDS` | `42,1337,2026` | Multi-model seed list for ensembling. |
| `MAX_FEATURES_PER_MODEL` | `1200` | Per-seed feature cap (`<=0` disables sampling). |
| `FEATURE_SAMPLING_STRATEGY` | `sharded_shuffle` | Feature selection strategy. |
| `FEATURE_SAMPLING_MASTER_SEED` | `0` | Master seed controlling deterministic feature assignment. |
| `LOAD_BACKEND` | `polars` | Dataset loading backend. |
| `LOAD_MODE` | `in_memory` | `in_memory` or cached preprocessing mode. |
| `BENCH_DROP_SPARSE_COLUMNS` | `true` | Drop benchmark columns above null-ratio threshold before BMC alignment. |
| `BENCH_MAX_NULL_RATIO_PER_COLUMN` | `0.0` | Maximum allowed null ratio per benchmark column (`0.0-1.0`). |
| `BENCH_MIN_COLUMNS` | `1` | Minimum benchmark columns required after sparse-column filtering. |
| `BENCH_MIN_COVERED_ROWS_PER_WINDOW` | `512` | Minimum benchmark-covered validation rows required per blend-tuning window. |
| `BENCH_MIN_COVERED_ERAS_PER_WINDOW` | `8` | Minimum benchmark-covered eras required per blend-tuning window. |
| `LGBM_NUM_LEAVES` | `128` | LightGBM tree leaves. |
| `LGBM_MIN_DATA_IN_LEAF` | `1000` | LightGBM regularization parameter. |
| `LGBM_FEATURE_FRACTION` | `0.7` | Column subsampling ratio. |
| `LGBM_BAGGING_FRACTION` | `0.8` | Row subsampling ratio. |
| `LGBM_BAGGING_FREQ` | `1` | Bagging frequency. |
| `LGBM_LEARNING_RATE` | `0.02` | Learning rate. |
| `LGBM_NUM_BOOST_ROUND` | `5000` | Maximum boosting rounds. |
| `WALKFORWARD_NUM_BOOST_ROUND` | `LGBM_NUM_BOOST_ROUND` | Boosting-round cap used only for walk-forward and blend-window tuning fits. |
| `LGBM_EARLY_STOPPING_ROUNDS` | `300` | Early stopping patience. |
| `CORR_SCAN_PERIOD` | `100` | Iteration interval for CORR scanning. |
| `CORR_SCAN_MAX_ITERS` | unset | Optional scan upper bound. |
| `SELECT_BEST_BY` | `corr` | Checkpoint metric selector (`corr`/`rmse`). |
| `NUMERAI_ID_COL` | `id` | Row id column for benchmark alignment. |
| `PAYOUT_WEIGHT_CORR` | `0.75` | Payout objective CORR weight for blend tuning. |
| `PAYOUT_WEIGHT_BMC` | `2.25` | Payout objective BMC weight for blend tuning. |
| `BLEND_ALPHA_GRID` | `0.0,0.1,...,1.0` | Alpha candidates for blending raw/neutralized predictions. |
| `BENCH_NEUTRALIZE_PROP_GRID` | `0.0,0.25,0.5,0.75,1.0` | Neutralization strength candidates. |
| `BLEND_TUNE_SEED` | `WALKFORWARD_TUNE_SEED` | Seed used for blend tuning windows. |
| `BLEND_USE_WINDOWS` | `WALKFORWARD_MAX_WINDOWS` | Number of windows used in blend tuning. |
| `WALKFORWARD_ENABLED` | `true` | Enable walk-forward pre-fit tuning. |
| `WALKFORWARD_CHUNK_SIZE` | `156` | Eras per walk-forward chunk. |
| `WALKFORWARD_PURGE_ERAS` | `8` | Era gap between train/validation windows. |
| `WALKFORWARD_MAX_WINDOWS` | `4` | Max recent windows to evaluate. |
| `WALKFORWARD_TUNE_SEED` | first `LGBM_SEEDS` | Seed used for walk-forward tuning. |
| `WALKFORWARD_LOG_MODELS` | `false` | Whether to log per-window models. |

## Inference Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUMERAI_DATASET_VERSION` | `v5.2` | Runtime dataset version guard. |
| `EXPOSURE_SAMPLE_ROWS` | `200000` | Rows sampled for exposure drift checks. |
| `EXPOSURE_SAMPLE_SEED` | `0` | Sampling seed for drift checks. |
| `MIN_PRED_STD` | `1e-6` | Minimum prediction standard deviation gate. |
| `MAX_ABS_EXPOSURE` | `0.30` | Maximum allowed absolute feature exposure. |

## Colab Notebook Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `REPO_REF` | unset | Optional pinned 40-char commit SHA checkout. |
| `REPO_DIR` | `/content/Numerai-Re` | Repo clone path in Colab runtime. |
| `PERSISTENT_ROOT` | `/content/drive/MyDrive/Numerai-Re` | Persistent drive root used by notebook setup. |
| `COLAB_ENV_PATH` | `/content/drive/MyDrive/Numerai-Re/.env.colab` | Drive-first env file loaded by the Colab notebook setup cell. |
| `COLAB_DRIVE_DATA_ROOT` | `/content/drive/MyDrive/Numerai-Re/datasets/numerai` | Drive dataset cache root used as sync source for local training data. |
| `COLAB_SYNC_DATA_FROM_DRIVE` | `true` | When enabled, notebook setup copies required v5.x parquet files from Drive cache to local `NUMERAI_DATA_DIR` when missing. |
