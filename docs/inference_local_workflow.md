# Inference Local Workflow

This runbook keeps secrets in `.env.inference.local` and loads them into the current shell.

Automation note:
- Production submission automation is handled by `.github/workflows/submit.yml`.
- Use this local workflow for first-time setup, smoke checks, or manual backfill/debug runs.
- For GitHub Actions verification without live upload, run the submit workflow via `workflow_dispatch` with `submission_mode=dry_run`.
- It is normal for new submissions to be recorded before full performance metrics are visible.

## 0) Create local env file (do not commit)

Create `.env.inference.local` at repo root with:

```env
PYTHONPATH=src
WANDB_API_KEY=YOUR_WANDB_API_KEY
WANDB_ENTITY=YOUR_WANDB_ENTITY
WANDB_PROJECT=numerai-mlops
WANDB_MODEL_NAME=lgbm_numerai_v52
NUMERAI_PUBLIC_ID=YOUR_NUMERAI_PUBLIC_ID
NUMERAI_SECRET_KEY=YOUR_NUMERAI_SECRET_KEY
NUMERAI_MODEL_NAME=YOUR_NUMERAI_MODEL_NAME
```

---

## Windows PowerShell

1. Open PowerShell at repo root

```powershell
cd C:\Users\w4343\OneDrive\桌面\Numerai-Re
```

2. Activate venv

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Load env from `.env.inference.local`

```powershell
Get-Content .\.env.inference.local |
  Where-Object { $_ -and $_ -notmatch '^\s*#' } |
  ForEach-Object {
    $key, $value = $_.Split('=', 2)
    if ($key -and $value) {
      Set-Item -Path "Env:$($key.Trim())" -Value $value.Trim()
    }
  }
```

4. Preflight validation

```powershell
python -m tools.validate_pipeline --dry-run --artifact-dir artifacts/mock_prod
```

5. Optional dry-run inference

```powershell
$env:INFER_DRY_RUN="true"
python -m numerai_re.cli.inference
Remove-Item Env:INFER_DRY_RUN -ErrorAction SilentlyContinue
```

6. Promote candidate artifact to prod

```powershell
python -m numerai_re.cli.promote_model
```

7. Live inference with log file (PowerShell-safe)

```powershell
cmd /c "python -m numerai_re.cli.inference 2>&1" | Tee-Object -FilePath infer_live.log
```

8. Optional log highlights

```powershell
Select-String -Path infer_live.log -Pattern "phase=artifact_downloaded|phase=frame_loaded|postprocess_loaded|DRIFT_GUARD_ABORT|Traceback|submit|upload|submission"
```

9. If blocked by exposure gate (temporary override)

```powershell
$env:MAX_ABS_EXPOSURE="0.35"
cmd /c "python -m numerai_re.cli.inference 2>&1" | Tee-Object -FilePath infer_live.log
Remove-Item Env:MAX_ABS_EXPOSURE -ErrorAction SilentlyContinue
```

---

## macOS (zsh/bash)

1. Open terminal at repo root

```bash
cd ~/path/to/Numerai-Re
```

2. Activate venv

```bash
source .venv/bin/activate
```

3. Load env from `.env.inference.local`

```bash
set -a
source .env.inference.local
set +a
```

4. Preflight validation

```bash
python -m tools.validate_pipeline --dry-run --artifact-dir artifacts/mock_prod
```

5. Optional dry-run inference

```bash
export INFER_DRY_RUN=true
python -m numerai_re.cli.inference
unset INFER_DRY_RUN
```

6. Promote candidate artifact to prod

```bash
python -m numerai_re.cli.promote_model
```

7. Live inference with log file

```bash
python -m numerai_re.cli.inference 2>&1 | tee infer_live.log
```

8. Optional log highlights

```bash
grep -E "phase=artifact_downloaded|phase=frame_loaded|postprocess_loaded|DRIFT_GUARD_ABORT|Traceback|submit|upload|submission" infer_live.log
```

9. If blocked by exposure gate (temporary override)

```bash
export MAX_ABS_EXPOSURE=0.35
python -m numerai_re.cli.inference 2>&1 | tee infer_live.log
unset MAX_ABS_EXPOSURE
```
