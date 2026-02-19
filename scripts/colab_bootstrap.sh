#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/<your-org-or-user>/Numerai-Re.git}"
REPO_REF="${REPO_REF:-}"
REPO_DIR="${REPO_DIR:-/content/Numerai-Re}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch --all --tags
if [[ -n "${REPO_REF}" ]]; then
  git checkout "${REPO_REF}"
fi

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements-train.txt

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
    print("Loaded WANDB_API_KEY from Colab Secrets for this cell runtime.")
PY

echo "Bootstrap complete at ${REPO_DIR}"
