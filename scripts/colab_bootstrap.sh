#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/<your-org-or-user>/Numerai-Re.git}"
REPO_REF="${REPO_REF:-}"
REPO_DIR="${REPO_DIR:-/content/Numerai-Re}"

echo "[bootstrap] Expected env vars: REPO_URL (required), REPO_REF (optional), REPO_DIR (optional)."
echo "[bootstrap] Repo URL: ${REPO_URL}"
echo "[bootstrap] Repo ref: ${REPO_REF:-<default branch>}"
echo "[bootstrap] Repo dir: ${REPO_DIR}"

echo "[bootstrap] Step 1/4: Checking prerequisites (git, python)."
command -v git >/dev/null
command -v python >/dev/null

echo "[bootstrap] Step 2/4: Cloning/updating repository."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git remote set-url origin "${REPO_URL}"
git fetch --all --tags
if [[ -n "${REPO_REF}" ]]; then
  git checkout "${REPO_REF}"
fi

echo "[bootstrap] Step 3/4: Installing Python dependencies."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements-train.txt

echo "[bootstrap] Step 4/4: Loading optional runtime secrets."
python - <<'PY'
import os

if os.getenv("WANDB_API_KEY", "").strip():
    raise SystemExit(0)

try:
    from google.colab import userdata
except Exception:
    raise SystemExit(0)

api_key = userdata.get("WANDB_API_KEY")
if api_key:
    os.environ["WANDB_API_KEY"] = api_key
    print("[bootstrap] Loaded WANDB_API_KEY from Colab Secrets for this cell runtime.")
PY

echo "[bootstrap] Complete: ${REPO_DIR}"
