# Copilot Instructions for Numerai-Re

## Big Picture
- Remote-train / auto-submit NumerAI system: training (Colab) logs model artifacts to W&B; inference (GitHub Actions) pulls promoted artifacts and submits.
- Core boundary: `training/` produces artifact payloads; `inference/` consumes them; `contracts/artifact_contract.py` defines the handshake.
- Treat `artifacts/*.json` (manifest/features/postprocess files) as API contracts.

## Key Paths
- Train: `python -m numerai_re.cli.train_colab` -> `src/numerai_re/training/training_pipeline.py`
- Infer: `python -m numerai_re.cli.inference` -> `src/numerai_re/inference/inference_runtime.py`
- Promote: `python -m numerai_re.cli.promote_model` (`candidate` -> `prod` after checks)
- Runtime env parsing is centralized in `src/numerai_re/runtime/config.py`.

## Working Style (Open Guidance)
- Keep full autonomy; aim for changes that improve correctness without unnecessary complexity.
- Prefer updating existing code when it stays readable, but add new code when it clearly helps.
- For simple read/print/transform paths, straightforward fail-fast behavior is often enough.
- Spend extra defensive effort on high-risk boundaries (artifact contract, train/infer interface, NumerAPI/W&B, drift gates).
- Keep readability and code size in mind as practical tradeoffs, not hard constraints.
- Always remember to update relevant docs (including .env.colab.example) if needed after changes.

## Fast Validation
- Run with `PYTHONPATH=src` from repo root.
- Useful smoke checks:
  - `PYTHONPATH=src python -m tools.validate_pipeline --dry-run`
  - `PYTHONPATH=src TRAIN_DRY_RUN=true python -m numerai_re.cli.train_colab`
  - `PYTHONPATH=src INFER_DRY_RUN=true python -m numerai_re.cli.inference`
