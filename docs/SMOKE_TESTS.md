# Smoke Tests

Run from the repository root.

```bash
PYTHONPATH=src python -c "import numerai_re.runtime.config, numerai_re.inference.postprocess, numerai_re.cli.train_colab, numerai_re.cli.inference"
PYTHONPATH=src python -m tools.validate_pipeline --dry-run --artifact-dir artifacts/mock_prod
PYTHONPATH=src TRAIN_DRY_RUN=true python -m numerai_re.cli.train_colab
PYTHONPATH=src INFER_DRY_RUN=true python -m numerai_re.cli.inference
```

Notes:
- Use `--artifact-dir artifacts/mock_prod` when validating the local mock artifact payload in this repository layout.
