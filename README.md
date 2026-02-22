# NumerAI Hybrid-Cloud MLOps

This repository implements a **remote-train / auto-submit** pipeline:

- **Training engine (Colab Pro, 51GB RAM)**: `src/train_colab.py`
  - Uses `NumerAPI` (official NumerAI flow) to download train/validation/features datasets.
  - Uses Polars + float downcasting for efficient loading.
  - Trains LightGBM with per-seed checkpoint/resume and logs candidate model artifacts to W&B (`latest`, `candidate`).
- **Inference agent (GitHub Actions)**: `src/inference.py`
  - Pulls the `prod` model artifact from W&B.
  - Downloads live data via `NumerAPI`.
  - Enforces manifest/runtime dataset-version compatibility by default.
  - Applies drift/quality gates before submission.
  - Uploads predictions to NumerAI.
- **Promotion gate (GitHub Actions)**: `.github/workflows/promote-model.yml`
  - Promotes a checked `candidate` artifact to `prod` only after integrity/compatibility checks.
- **Drift guard**: workflow opens a GitHub issue on failures.

## 60-second quickstart

### Train in Colab

1. Open `notebooks/train_colab.ipynb` in Colab.
2. Set notebook env vars in the first cell:
   - `REPO_REF` (optional)
   - `REPO_DIR` (optional, defaults to `/content/Numerai-Re`)
   - Repo URL is fixed to `https://github.com/WilliamQD/Numerai-Re.git`.
3. Set Colab Secret `WANDB_API_KEY`.
4. Run the setup cell (clone/update + dependency install), then run the training cell (`src/train_colab.py`).

### Run inference via GitHub Actions

1. Add the required GitHub repository secrets (see environment table below).
2. Ensure your NumerAI model name in `NUMERAI_MODEL_NAME` exists in your NumerAI account.
3. Trigger `.github/workflows/submit.yml` (manual run or scheduled run).
4. Inspect workflow logs for drift-guard status and submission result.

## Files

- `.github/workflows/submit.yml`: Weekly CI job and failure alert.
- `.github/workflows/code-safety.yml`: Fast syntax + lint checks on PRs/pushes.
- `.github/workflows/promote-model.yml`: Manual candidate-to-prod promotion workflow.
- `notebooks/train_colab.ipynb`: Colab runner notebook (safe repo sync + dependency install + launch).
- `scripts/colab_bootstrap.sh`: Clones fixed public repo into Colab runtime, enforces origin/ref safety checks, installs deps.
- `src/train_colab.py`: Main training script.
- `src/inference.py`: Weekly inference + submit script.

## Colab workflow (recommended)

> **Do not edit code in Colab.**
> Edit on GitHub (locally/Copilot/Codex), then use Colab only as a disposable execution runner.

1. Open `notebooks/train_colab.ipynb` in Colab.
2. Set notebook environment variables in the first code cell:
   - `REPO_REF` (optional): full 40-character commit SHA override. Leave empty to run latest `main`.
   - `REPO_DIR` (optional): clone path, default `/content/Numerai-Re`.
   - Repo source is fixed to `https://github.com/WilliamQD/Numerai-Re.git`.
3. In Colab **Secrets** (`🔑` sidebar), set:
    - `WANDB_API_KEY` (required for training)
    - `NUMERAI_PUBLIC_ID` + `NUMERAI_SECRET_KEY` (optional; only if you want authenticated dataset downloads during training)
    - For manual self-inference in Colab, also add `NUMERAI_MODEL_NAME`, `WANDB_ENTITY`, and `WANDB_PROJECT`.
4. Run setup cell to clone/update repo + install `requirements-train.txt`.
5. Run training cell to execute `src/train_colab.py` from the cloned repo.

### Notes on secrets

- GitHub Actions exposes NumerAI/W&B secrets only to the single inference step (not the whole job).
- Python dependencies are pinned to exact versions in requirements files to reduce supply-chain drift.
- Colab setup verifies that any pre-existing repository in `REPO_DIR` points to the expected origin before running updates.
- If `REPO_REF` is empty, Colab setup updates and runs the latest `main` branch.
- `src/train_colab.py` fails fast if `WANDB_API_KEY` is missing.
- `src/train_colab.py` downloads train/validation/features via `NumerAPI`; these public datasets work without NumerAI keys, but you can optionally set `NUMERAI_PUBLIC_ID` + `NUMERAI_SECRET_KEY` in Colab Secrets for authenticated downloads.
- Colab setup mounts Google Drive and defaults persistent storage to `/content/drive/MyDrive/Numerai-Re`.
- If `NUMERAI_DATA_DIR` is unset, Colab setup uses `/content/drive/MyDrive/Numerai-Re/datasets/numerai` and training reuses existing required files instead of re-downloading.
- Training checkpoints are persisted under `NUMERAI_DATA_DIR/<dataset_version>/checkpoints/<WANDB_MODEL_NAME>/` so interrupted Colab runs resume remaining seeds only.
- Notebook setup auto-loads `WANDB_API_KEY`, `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`, `NUMERAI_MODEL_NAME`, `WANDB_ENTITY`, and `WANDB_PROJECT` from Colab Secrets when those environment variables are unset; missing secrets are skipped.
- Keep secrets in Colab Secrets or environment variables; never hardcode keys into notebook/code.

## Environment variables

### Required

| Variable | Script | Purpose |
| --- | --- | --- |
| `WANDB_API_KEY` | `src/train_colab.py` (Colab) | Required to authenticate and log artifacts to Weights & Biases during training. |
| `NUMERAI_PUBLIC_ID` | `src/inference.py` (GitHub Actions) | NumerAI API public ID for live data download and submission. |
| `NUMERAI_SECRET_KEY` | `src/inference.py` (GitHub Actions) | NumerAI API secret key paired with `NUMERAI_PUBLIC_ID`. |
| `NUMERAI_MODEL_NAME` | `src/inference.py` (GitHub Actions) | Target NumerAI model name to receive submissions. |
| `WANDB_ENTITY` | `src/inference.py` (GitHub Actions) | W&B entity that owns the model artifact. |
| `WANDB_PROJECT` | `src/inference.py` (GitHub Actions) | W&B project containing the model artifact. |

### Optional

| Variable | Script | Default | Purpose |
| --- | --- | --- | --- |
| `REPO_REF` | `notebooks/train_colab.ipynb` setup | unset | Optional full 40-character commit SHA override; when unset, setup runs latest `main`. |
| `REPO_DIR` | `notebooks/train_colab.ipynb` setup | `/content/Numerai-Re` | Optional clone destination inside Colab runtime. |
| `NUMERAI_DATA_DIR` | `src/train_colab.py` | `/content/drive/MyDrive/Numerai-Re/datasets/numerai` (Colab setup default) | Override NumerAI dataset download path in Colab. |
| `PERSISTENT_ROOT` | `notebooks/train_colab.ipynb`, `scripts/colab_bootstrap.sh` | `/content/drive/MyDrive/Numerai-Re` | Persistent root inside Google Drive used by Colab setup for cached data. |
| `NUMERAI_PUBLIC_ID` | `src/train_colab.py` | unset | Optional NumerAI public ID for authenticated training dataset downloads (public dataset download also works without it). |
| `NUMERAI_SECRET_KEY` | `src/train_colab.py` | unset | Optional NumerAI secret key paired with `NUMERAI_PUBLIC_ID` for authenticated training dataset downloads. |
| `NUMERAI_FEATURE_SET` | `src/train_colab.py` | `all` | Select feature set from NumerAI `features.json` (`small`, `medium`, or `all`). |
| `WANDB_MODEL_NAME` | `src/train_colab.py`, `src/inference.py` | `lgbm_numerai_v52` | Override model artifact name for logging/loading. |
| `WANDB_PROJECT` | `src/train_colab.py` | `numerai-mlops` | Override W&B project for training logs. |
| `WANDB_ENTITY` | `src/train_colab.py` | unset | Optional W&B entity override for training run ownership. |
| `LGBM_DEVICE` | `src/train_colab.py` | `cpu` | Choose CPU-first training (`cpu`, default) or optional `gpu` acceleration (auto-fallbacks to CPU if OpenCL GPU is unavailable). |
| `LGBM_SEEDS` | `src/train_colab.py` | `42,1337,2026` | Comma-separated seeds used for multi-model training; predictions are ensembled by mean. |
| `LGBM_NUM_LEAVES` | `src/train_colab.py` | `128` | LightGBM leaves for CPU-focused Numerai baseline. |
| `LGBM_MIN_DATA_IN_LEAF` | `src/train_colab.py` | `1000` | LightGBM minimum data in leaf for regularization on large tabular training sets. |
| `LGBM_FEATURE_FRACTION` | `src/train_colab.py` | `0.7` | Column subsampling fraction per tree. |
| `LGBM_BAGGING_FRACTION` | `src/train_colab.py` | `0.8` | Row subsampling fraction for bagging. |
| `LGBM_BAGGING_FREQ` | `src/train_colab.py` | `1` | Bagging frequency. |
| `LGBM_LEARNING_RATE` | `src/train_colab.py` | `0.02` | Learning rate for boosting. |
| `LGBM_NUM_BOOST_ROUND` | `src/train_colab.py` | `5000` | Number of boosting rounds (CPU baseline target range: 3000-6000). |
| `LGBM_EARLY_STOPPING_ROUNDS` | `src/train_colab.py` | `300` | Early stopping rounds (CPU baseline target range: 200-400). |
| `CORR_SCAN_PERIOD` | `src/train_colab.py` | `100` | Iteration interval used to scan validation Numerai CORR for checkpoint selection/logging. |
| `CORR_SCAN_MAX_ITERS` | `src/train_colab.py` | unset | Optional cap on max iteration to scan for validation Numerai CORR. |
| `SELECT_BEST_BY` | `src/train_colab.py` | `corr` | Checkpoint selection metric (`corr` or `rmse`); default uses Numerai CORR. |
| `NUMERAI_ID_COL` | `src/train_colab.py` | `id` | Row identifier column used to align benchmark-model parquet predictions. |
| `PAYOUT_WEIGHT_CORR` | `src/train_colab.py` | `0.75` | Payout-score CORR weight used in PR3 blend tuning (`wC*corr + wM*bmc`). |
| `PAYOUT_WEIGHT_BMC` | `src/train_colab.py` | `2.25` | Payout-score BMC weight used in PR3 blend tuning (`wC*corr + wM*bmc`). |
| `BLEND_ALPHA_GRID` | `src/train_colab.py` | `0.0,0.1,0.2,...,1.0` | Comma-separated blend alpha candidates for raw vs. benchmark-neutralized predictions. |
| `BENCH_NEUTRALIZE_PROP_GRID` | `src/train_colab.py` | `0.0,0.25,0.5,0.75,1.0` | Comma-separated benchmark neutralization strengths searched during blend tuning. |
| `BLEND_TUNE_SEED` | `src/train_colab.py` | `WALKFORWARD_TUNE_SEED` | Seed used for blend tuning over walk-forward windows. |
| `BLEND_USE_WINDOWS` | `src/train_colab.py` | `WALKFORWARD_MAX_WINDOWS` | Number of most recent walk-forward windows used for PR3 blend tuning. |
| `NUMERAI_DATASET_VERSION` | `src/inference.py` | `v5.2` | Override NumerAI dataset version for live data. |
| `ALLOW_DATASET_VERSION_MISMATCH` | `src/inference.py` | `false` | Explicitly allow inference with manifest/runtime dataset-version mismatch (not recommended). |
| `MIN_PRED_STD` | `src/inference.py` | `1e-6` | Drift guard minimum prediction standard deviation threshold. |
| `MAX_ABS_EXPOSURE` | `src/inference.py` | `0.30` | Drift guard maximum absolute feature exposure threshold. |

## Operational checks and failure modes

- **Drift guard aborts**: inference exits with `DRIFT_GUARD_ABORT` when predictions are invalid (NaN/Inf/all-zero), too flat (`MIN_PRED_STD`), too exposed (`MAX_ABS_EXPOSURE`), or required live columns are missing.
- **Missing NumerAI model**: if `NUMERAI_MODEL_NAME` does not exist in your NumerAI account, inference fails before upload and reports available model names.
- **Incomplete W&B artifact**: if the downloaded `prod` model artifact is missing `features.json` or the model file referenced by `train_manifest.json`, inference aborts with a required-files error.
- **Failure visibility**: when the workflow fails, drift-guard handling opens a GitHub issue so failures are visible asynchronously.

## Local checks

Run from the repository root (`/workspace/Numerai-Re`):

```bash
python -m py_compile src/train_colab.py src/inference.py src/config.py src/promote_model.py
ruff check src
```
