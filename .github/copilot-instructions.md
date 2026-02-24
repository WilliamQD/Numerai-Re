# Copilot Instructions for Numerai-Re

## Big Picture
- This repo is a **remote-train / auto-submit** NumerAI system: Colab training publishes W&B model artifacts; GitHub Actions inference pulls the promoted artifact and submits.
- Main boundaries:
  - `src/numerai_re/training/` builds/checkpoints ensemble members and writes artifact payloads.
  - `src/numerai_re/inference/` loads promoted artifacts, runs postprocess + drift gates, uploads predictions.
  - `src/numerai_re/contracts/artifact_contract.py` is the schema gate between train and inference.
- Treat `artifacts/` JSON files as API contracts (`train_manifest.json`, `features_union.json`, `features_by_model.json`, `postprocess_config.json`).

## Entry Points and Data Flow
- Training entrypoint: `python -m numerai_re.cli.train_colab` (`src/numerai_re/cli/train_colab.py`) -> `training/training_pipeline.py`.
- Inference entrypoint: `python -m numerai_re.cli.inference` (`src/numerai_re/cli/inference.py`) -> `inference/inference_runtime.py`.
- Promotion entrypoint: `python -m numerai_re.cli.promote_model` (`src/numerai_re/cli/promote_model.py`) promotes `candidate` -> `prod` after manifest/model-file checks.
- Runtime config is env-only and centralized in `src/numerai_re/runtime/config.py`; avoid duplicating env parsing elsewhere.

## Project-Specific Conventions
- Always run modules with `PYTHONPATH=src` from repo root.
- Keep phase-based structured logs (`phase=...`) used across train/infer/loader code.
- Dry-run behavior is first-class and CI-relevant:
  - `TRAIN_DRY_RUN=true` must print `TRAIN_DRY_RUN_OK`.
  - `INFER_DRY_RUN=true` must print `INFER_DRY_RUN_OK`.
- Artifact compatibility is strict:
  - Use helpers like `load_manifest`, `resolve_model_files`, `validate_model_files_exist`, `load_union_features`.
  - Inference assumes `artifact_schema_version` and postprocess keys are present.
- Checkpoint resume is strict on metadata parity (`dataset_version`, seeds, lgb params, sampling settings) in `training/training_checkpoint.py`; mismatches require retrain (delete checkpoint).
- Feature sampling strategy is intentionally deterministic (`sharded_shuffle`) and keyed by seeds/master seed (`features/feature_sampling.py`).

## Complexity Budget (Very Important)
- Keep full autonomy, but actively consider whether added code is necessary for the request and contract.
- Bias toward the simplest readable solution that satisfies requirements; prefer modifying existing code when practical.
- Treat defensive branches and fallbacks as optional by default; add them when they materially improve correctness in real runtime scenarios.
- For trivial operations (simple read/print/transform), simple fail-fast behavior is usually preferred over layered fallback logic.
- Reserve deeper validation and handling for high-risk boundaries (artifact contracts, train/infer handshake, NumerAPI/W&B integration, drift/quality gates).
- Keep readability and maintainability in view when balancing correctness, clarity, and code size.

## Critical Workflows (Use These)
- Install deps: `pip install -r requirements.txt` (or train/infer-specific files).
- Smoke checks (from `docs/SMOKE_TESTS.md`):
  - `PYTHONPATH=src python -m tools.validate_pipeline --dry-run`
  - `PYTHONPATH=src TRAIN_DRY_RUN=true python -m numerai_re.cli.train_colab`
  - `PYTHONPATH=src INFER_DRY_RUN=true python -m numerai_re.cli.inference`
- Contract/lint guard: `tools/validate_pipeline.py` validates artifact schema + runtime contract + syntax AST checks.
- Unit/integration tests live under `tests/unit` and `tests/integration`; keep new tests close to changed subsystem.

## Integration Points
- W&B:
  - Training logs artifact aliases `latest` and `candidate` (`training/training_artifact.py`).
  - Inference fetches `:prod`; promotion workflow is explicit and manual.
- NumerAPI:
  - Training downloads train/validation + benchmarks with optional authenticated access.
  - Inference downloads live + live benchmark parquet and uploads `submission.csv`.
- Dataset handling:
  - Dataset version guard is enforced (`NUMERAI_DATASET_VERSION`).
  - `USE_INT8_PARQUET` resolution is schema-aware (not just filename suffix).

## Change Guidance for Agents
- Prefer minimal, contract-preserving edits over broad refactors.
- Before adding new code, consider whether existing code can be adjusted or simplified directly.
- If touching train/infer handshake, update both producer (`training_artifact.py`) and consumer (`artifact_contract.py`/`cli/inference.py`) plus tests.
- When adding env knobs, wire through `runtime/config.py` and document in `docs/env_reference.md`.
- Preserve public CLI module paths and dry-run semantics used by CI/tests.
