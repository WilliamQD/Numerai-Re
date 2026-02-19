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

## Files

- `.github/workflows/submit.yml`: Weekly CI job and failure alert.
- `notebooks/train_colab.ipynb`: Thin Colab runner notebook (bootstrap + launch script).
- `scripts/colab_bootstrap.sh`: Clones repo into Colab runtime, checks out optional ref, installs deps.
- `src/train_colab.py`: Main training script.
- `src/inference.py`: Weekly inference + submit script.

## Colab workflow (recommended)

> **Do not edit code in Colab.**
> Edit on GitHub (locally/Copilot/Codex), then use Colab only as a disposable execution runner.

1. Open `notebooks/train_colab.ipynb` in Colab.
2. Set notebook environment variables in the first code cell:
   - `REPO_URL`: your GitHub clone URL for this repo.
   - `REPO_REF` (optional): branch/tag/commit SHA to run (pin a SHA for reproducibility).
   - `REPO_DIR` (optional): clone path, default `/content/Numerai-Re`.
3. In Colab **Secrets** (`🔑` sidebar), set:
   - `WANDB_API_KEY` (required for training)
   - `GH_TOKEN` (optional; only needed for private repo clone over HTTPS)
4. Run bootstrap cell to clone repo + install `requirements-train.txt`.
5. Run training cell to execute `src/train_colab.py` from the cloned repo.

### Notes on secrets

- `src/train_colab.py` fails fast if `WANDB_API_KEY` is missing.
- Keep secrets in Colab Secrets or environment variables; never hardcode keys into notebook/code.

## Optional runtime knobs

- `NUMERAI_DATA_DIR`: override dataset download path (default `/content/numerai_data`).
- `NUMERAI_FEATURE_SET`: choose feature set (default `medium`).
- `WANDB_MODEL_NAME`: override logged model artifact name.
- `LGBM_DEVICE`: `gpu` (default) or `cpu` for fallback environments.

## Secrets required in GitHub Actions

- `NUMERAI_PUBLIC_ID`
- `NUMERAI_SECRET_KEY`
- `NUMERAI_MODEL_NAME`
- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`

## Local checks

```bash
python -m py_compile src/train_colab.py src/inference.py
```
