# Smoke Tests

Run from repository root (`/home/runner/work/Numerai-Re/Numerai-Re`).

```bash
python -c "import src.config, src.postprocess, src.train_colab, src.inference"
python src/train_colab.py
python -m tools.validate_pipeline --dry-run
TRAIN_DRY_RUN=true python src/train_colab.py
INFER_DRY_RUN=true python src/inference.py
```
