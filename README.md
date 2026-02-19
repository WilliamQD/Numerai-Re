# NumerAI Hybrid-Cloud MLOps

This repository implements a **remote-train / auto-submit** pipeline:

- **Training engine (Colab Pro, 51GB RAM)**: `src/train_colab.py`
  - Uses `NumerAPI` (official NumerAI flow) to download train/validation/features datasets.
  - Uses Polars + float downcasting for efficient loading.
  - Trains GPU LightGBM and logs model artifacts to W&B with aliases (`latest`, `prod`).
- **Inference agent (GitHub Actions)**: `src/inference.py`
  - Pulls the `prod` model artifact from W&B.
  - Downloads live data via `NumerAPI`.
  - Applies drift/quality gates before submission.
  - Uploads predictions to NumerAI.
- **Drift guard**: workflow opens a GitHub issue on failures.

## Files

- `.github/workflows/submit.yml`: Weekly CI job and failure alert.
- `notebooks/train_colab.ipynb`: Colab notebook wrapper.
- `src/train_colab.py`: Main training script.
- `src/inference.py`: Weekly inference + submit script.

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
