# NumerAI Hybrid-Cloud MLOps

This repository runs a **remote-train / auto-submit** system:
- Training runs in Colab and logs versioned model artifacts to W&B.
- Inference runs in GitHub Actions, pulls the promoted artifact, applies quality gates, and submits to NumerAI.
- Promotion is a manual workflow that validates candidate artifacts before moving them to `prod`.

## Quickstart

### Train in Colab
1. Open `notebooks/train_colab.ipynb`.
2. Set Colab secret `WANDB_API_KEY`.
3. Run setup cell, then run training cell (launches `python -m numerai_re.cli.train_colab`).

### Run inference in GitHub Actions
1. Configure required repo secrets: `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`, `NUMERAI_MODEL_NAME`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`.
2. Trigger `.github/workflows/submit.yml` manually or wait for schedule.
  - Scheduled cadence is aligned to Numerai round windows (Tue-Fri multiple runs in 13:00-14:00 UTC, Saturday early + follow-up pass).
  - Policy is single attempt per workflow run; failures create a GitHub issue alert.
  - Manual dispatch supports `submission_mode=dry_run` for safe workflow testing (no Numerai upload) and `submission_mode=live` for real submissions.
  - Each run uploads `infer.log` as a workflow artifact and writes submission metadata to the job summary.

### Run inference locally (first-time friendly)
1. Follow the manual runbook in `inference_local_workflow.md`.
2. Optional quick path from repo root (PowerShell):
  - `.\.venv\Scripts\Activate.ps1`
  - `$env:PYTHONPATH="src"`
  - `python -m tools.validate_pipeline --dry-run --artifact-dir artifacts/mock_prod`
  - `python -m numerai_re.cli.promote_model`
  - `cmd /c "python -m numerai_re.cli.inference 2>&1" | Tee-Object -FilePath infer_live.log`

## Repository Structure

```text
.
├─ artifacts/
├─ docs/
│  ├─ env_reference.md
│  └─ SMOKE_TESTS.md
├─ notebooks/
├─ src/
│  └─ numerai_re/
│     ├─ cli/            # train/infer/promote entry modules
│     ├─ contracts/      # artifact and manifest contract checks
│     ├─ runtime/        # runtime config parsing from env
│     ├─ data/           # dataset and benchmark loading/alignment
│     ├─ features/       # feature sampling
│     ├─ metrics/        # NumerAI metrics and neutralization math
│     ├─ common/         # shared utilities
│     ├─ training/       # training pipeline internals
│     └─ inference/      # inference runtime and postprocess
├─ tests/
│  ├─ integration/
│  └─ unit/
└─ tools/
```

## Train Pipeline Structure

```mermaid
flowchart TD
  A[CLI: numerai_re.cli.train_colab] --> B[Runtime Config: runtime.config.TrainRuntimeConfig]
  B --> C[Pipeline Orchestrator: training.training_pipeline]
  C --> D[Data Layer: data_loading / numerapi_datasets / benchmarks]
  C --> E[Feature Layer: features.feature_sampling]
  C --> F[Model Layer: training_runtime + training_seed_runner]
  C --> G[Tuning Layer: training_tuning + walkforward + tune_blend]
  C --> H[Checkpoint Layer: training_checkpoint]
  C --> I[Artifact Layer: training_artifact + contracts.artifact_contract]
  I --> J[W&B Model Registry candidate/latest]
```

- Contains: config parsing, dataset acquisition, benchmark alignment, feature subset strategy, walk-forward tuning, seed ensemble training, checkpoint/resume, and artifact publication.
- Output: trained LightGBM model files + manifest/features/postprocess contract files in artifact storage.

## Inference Pipeline Structure

```mermaid
flowchart TD
  A[CLI: numerai_re.cli.inference] --> B[Runtime Config: runtime.config.InferenceRuntimeConfig]
  B --> C[Artifact Contract Load: contracts.artifact_contract]
  C --> D[Inference Runtime: inference.inference_runtime]
  D --> E[Live Data Loader: data.numerapi_datasets]
  D --> F[Model Predictors]
  F --> G[Postprocess: inference.postprocess]
  G --> H[Quality Gates: std/exposure/schema checks]
  H --> I[Submission Writer + NumerAPI upload]
```

- Contains: promoted artifact retrieval, manifest compatibility checks, live feature loading, ensemble prediction, postprocessing, and drift/quality gating.
- Output: validated submission predictions uploaded to NumerAI (or dry-run artifacts locally).

## Environment Variables

- Full reference moved to `docs/env_reference.md`.
- Runtime defaults and parsing logic are defined in:
  - `src/numerai_re/runtime/config.py`
- Minimum required by context:
  - Training (Colab): `WANDB_API_KEY`
  - Inference (GitHub Actions): `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`, `NUMERAI_MODEL_NAME`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`

## Local Checks

Run from repository root:

```bash
PYTHONPATH=src python -m py_compile src/numerai_re/cli/train_colab.py src/numerai_re/cli/inference.py src/numerai_re/runtime/config.py src/numerai_re/cli/promote_model.py
ruff check src
PYTHONPATH=src python -m tools.validate_pipeline --dry-run --artifact-dir artifacts/mock_prod
PYTHONPATH=src TRAIN_DRY_RUN=true python -m numerai_re.cli.train_colab
PYTHONPATH=src INFER_DRY_RUN=true python -m numerai_re.cli.inference
```

## CI Policy

- Pull requests and pushes to `main` run `.github/workflows/ci.yml` with two required jobs:
  - `lint-compile` (`compileall` + `ruff check src`)
  - `dry-run-pipeline` (`TRAIN_DRY_RUN`, `validate_pipeline`, `INFER_DRY_RUN`)
- Code changes should merge only after CI is green. If CI fails due to legitimate workflow drift, update workflow definitions in the same change set.
- Configure GitHub branch protection to require these CI checks before merge.

## Live Ops Cadence

- Inference submissions: run every round window (automation handles this via submit workflow schedule).
- Retraining: run weekly, then promote only validated candidates to `prod`.
- Score timing: live diagnostics usually appear after several days; fully settled round scoring can take about a month.
- Practical implication: do not pause submissions while waiting for first score updates.
