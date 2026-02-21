# NumerAI Hybrid-Cloud MLOps

This repository implements a **remote-train / auto-submit** pipeline:

- **Training engine (Colab Pro, 51GB RAM)**: `src/train_colab.py`
  - Uses `NumerAPI` (official NumerAI flow) to download train/validation/features datasets.
  - Uses Polars + float downcasting for efficient loading.
  - Trains LightGBM and logs model artifacts to W&B with aliases (`latest`, `prod`).
- **Inference agent (GitHub Actions)**: `src/inference.py`
  - Pulls the `prod` model artifact from W&B.
  - Downloads live data via `NumerAPI`.
  - Applies drift/quality gates before submission.
  - Uploads predictions to NumerAI.
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
4. Run setup cell to clone/update repo + install `requirements-train.txt`.
5. Run training cell to execute `src/train_colab.py` from the cloned repo.

### Notes on secrets

- GitHub Actions exposes NumerAI/W&B secrets only to the single inference step (not the whole job).
- Python dependencies are pinned to exact versions in requirements files to reduce supply-chain drift.
- Colab setup verifies that any pre-existing repository in `REPO_DIR` points to the expected origin before running updates.
- If `REPO_REF` is empty, Colab setup updates and runs the latest `main` branch.
- `src/train_colab.py` fails fast if `WANDB_API_KEY` is missing.
- `src/train_colab.py` downloads train/validation/features via `NumerAPI`; these public datasets work without NumerAI keys, but you can optionally set `NUMERAI_PUBLIC_ID` + `NUMERAI_SECRET_KEY` in Colab Secrets for authenticated downloads.
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
| `NUMERAI_DATA_DIR` | `src/train_colab.py` | `/content/numerai_data` | Override NumerAI dataset download path in Colab. |
| `NUMERAI_PUBLIC_ID` | `src/train_colab.py` | unset | Optional NumerAI public ID for authenticated training dataset downloads (public dataset download also works without it). |
| `NUMERAI_SECRET_KEY` | `src/train_colab.py` | unset | Optional NumerAI secret key paired with `NUMERAI_PUBLIC_ID` for authenticated training dataset downloads. |
| `NUMERAI_FEATURE_SET` | `src/train_colab.py` | `medium` | Select feature set from NumerAI `features.json`. |
| `WANDB_MODEL_NAME` | `src/train_colab.py`, `src/inference.py` | `lgbm_numerai_v43` | Override model artifact name for logging/loading. |
| `WANDB_PROJECT` | `src/train_colab.py` | `numerai-mlops` | Override W&B project for training logs. |
| `WANDB_ENTITY` | `src/train_colab.py` | unset | Optional W&B entity override for training run ownership. |
| `LGBM_DEVICE` | `src/train_colab.py` | `gpu` | Choose `gpu` (default) or `cpu` fallback for LightGBM. |
| `NUMERAI_DATASET_VERSION` | `src/inference.py` | `v4.3` | Override NumerAI dataset version for live data. |
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
python -m py_compile src/train_colab.py src/inference.py
```
