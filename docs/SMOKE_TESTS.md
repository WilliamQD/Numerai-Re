# Smoke Tests

Run from the repository root.

```bash
PYTHONPATH=src python -c "import numerai_re.runtime.config, numerai_re.inference.postprocess, numerai_re.cli.train_colab, numerai_re.cli.inference"
PYTHONPATH=src python -m tools.validate_pipeline --dry-run
PYTHONPATH=src TRAIN_DRY_RUN=true python -m numerai_re.cli.train_colab
PYTHONPATH=src INFER_DRY_RUN=true python -m numerai_re.cli.inference
```

Notes:
- `python -m tools.validate_pipeline --dry-run` now skips artifact-file checks when `artifacts/` is absent and still validates runtime/source contracts.
