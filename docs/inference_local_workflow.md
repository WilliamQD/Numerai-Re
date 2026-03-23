# Inference Local Workflow

This runbook uses PowerShell Secret Store as the preferred local credential source.
Use `.env.inference.local` only as a fallback when Secret Store is unavailable.

Automation note:
- Production submission automation is handled by `.github/workflows/submit.yml`.
- Use this local workflow for first-time setup, smoke checks, or manual backfill/debug runs.
- For GitHub Actions verification without live upload, run the submit workflow via `workflow_dispatch` with `submission_mode=dry_run`.
- It is normal for new submissions to be recorded before full performance metrics are visible.

## 0) Configure local secrets once (PowerShell Secret Store)

Install modules (one-time):

```powershell
Install-Module Microsoft.PowerShell.SecretManagement -Scope CurrentUser
Install-Module Microsoft.PowerShell.SecretStore -Scope CurrentUser
```

Save required credentials (one-time):

```powershell
Set-Secret -Name NUMERAI_PUBLIC_ID -Secret "YOUR_NUMERAI_PUBLIC_ID"
Set-Secret -Name NUMERAI_SECRET_KEY -Secret "YOUR_NUMERAI_SECRET_KEY"
Set-Secret -Name NUMERAI_MODEL_NAME -Secret "YOUR_NUMERAI_MODEL_NAME"
Set-Secret -Name WANDB_API_KEY -Secret "YOUR_WANDB_API_KEY"
Set-Secret -Name WANDB_ENTITY -Secret "YOUR_WANDB_ENTITY"
Set-Secret -Name WANDB_PROJECT -Secret "numerai-mlops"
Set-Secret -Name WANDB_MODEL_NAME -Secret "lgbm_numerai_v52"
```

Optional fallback only (do not commit): create `.env.inference.local` at repo root with equivalent values.

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

3. Load env from PowerShell Secret Store

```powershell
$env:PYTHONPATH = "src"
$env:NUMERAI_PUBLIC_ID = Get-Secret -Name NUMERAI_PUBLIC_ID -AsPlainText
$env:NUMERAI_SECRET_KEY = Get-Secret -Name NUMERAI_SECRET_KEY -AsPlainText
$env:NUMERAI_MODEL_NAME = Get-Secret -Name NUMERAI_MODEL_NAME -AsPlainText
$env:WANDB_API_KEY = Get-Secret -Name WANDB_API_KEY -AsPlainText
$env:WANDB_ENTITY = Get-Secret -Name WANDB_ENTITY -AsPlainText
$env:WANDB_PROJECT = Get-Secret -Name WANDB_PROJECT -AsPlainText
$env:WANDB_MODEL_NAME = Get-Secret -Name WANDB_MODEL_NAME -AsPlainText
```

Fallback loader from `.env.inference.local` if needed:

```powershell
if (Test-Path .\.env.inference.local) {
  Get-Content .\.env.inference.local |
    Where-Object { $_ -and $_ -notmatch '^\s*#' } |
    ForEach-Object {
      $key, $value = $_.Split('=', 2)
      if ($key -and $value) {
        Set-Item -Path "Env:$($key.Trim())" -Value $value.Trim()
      }
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

10. Collect/store live performance and generate manual-staking evidence recommendation

```powershell
python -m tools.performance_tracker
Get-Content .\artifacts\performance\latest_recommendation.json
```

Output artifacts:
- `artifacts/performance/raw/<timestamp>.json` (raw NumerAPI payload snapshot)
- `artifacts/performance/history.csv` (normalized score history)
- `artifacts/performance/latest_summary.json` (latest history summary)
- `artifacts/performance/latest_recommendation.json` (evidence score + recommendation)

Note:
- This workflow does not stake automatically. Use the recommendation as decision support, then perform any staking manually on the Numerai website.

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
