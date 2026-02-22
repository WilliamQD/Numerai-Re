#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/WilliamQD/Numerai-Re.git"
REPO_REF="${REPO_REF:-}"
REPO_DIR="${REPO_DIR:-/content/Numerai-Re}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/content/drive/MyDrive/Numerai-Re}"
DEFAULT_NUMERAI_DATA_DIR="${DEFAULT_NUMERAI_DATA_DIR:-${PERSISTENT_ROOT}/datasets/numerai}"
ALLOW_UNPINNED_REF="${ALLOW_UNPINNED_REF:-0}"
export PERSISTENT_ROOT
export DEFAULT_NUMERAI_DATA_DIR

canon_url() {
  local url="${1%.git}"
  url="${url%/}"
  printf '%s' "${url,,}"
}

echo "[bootstrap] Expected env vars: REPO_REF (optional 40-char commit SHA), REPO_DIR (optional)."
echo "[bootstrap] Repo URL (fixed): ${REPO_URL}"
echo "[bootstrap] Repo ref: ${REPO_REF:-main/latest}"
echo "[bootstrap] Repo dir: ${REPO_DIR}"
echo "[bootstrap] Persistent root: ${PERSISTENT_ROOT}"


if [[ -n "${REPO_REF}" ]]; then
  if [[ ! "${REPO_REF}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "[bootstrap] ERROR: REPO_REF must be a full 40-character commit SHA." >&2
    exit 1
  fi
fi

echo "[bootstrap] Step 1/4: Checking prerequisites (git, python)."
command -v git >/dev/null
command -v python >/dev/null

echo "[bootstrap] Step 2/4: Cloning/updating repository."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  current_origin="$(git -C "${REPO_DIR}" remote get-url origin)"
  if [[ "$(canon_url "${current_origin}")" != "$(canon_url "${REPO_URL}")" ]]; then
    echo "[bootstrap] ERROR: Existing repo origin '${current_origin}' does not match expected '${REPO_URL}'." >&2
    exit 1
  fi
fi

cd "${REPO_DIR}"
git fetch --all --tags --prune
if [[ -n "${REPO_REF}" ]]; then
  git checkout --detach "${REPO_REF}"
else
  git checkout main
  git pull --ff-only origin main
fi

echo "[bootstrap] Step 3/4: Installing Python dependencies."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements-train.txt

echo "[bootstrap] Step 4/4: Mounting Google Drive and loading optional runtime secrets."
python - <<'PY'
import os

try:
    from google.colab import drive
except Exception:
    drive = None

if drive:
    drive.mount("/content/drive", force_remount=False)
    persistent_root = os.getenv("PERSISTENT_ROOT", "/content/drive/MyDrive/Numerai-Re")
    os.makedirs(persistent_root, exist_ok=True)
    if not os.getenv("NUMERAI_DATA_DIR", "").strip():
        default_data_dir = os.getenv("DEFAULT_NUMERAI_DATA_DIR", f"{persistent_root}/datasets/numerai")
        os.makedirs(default_data_dir, exist_ok=True)
        os.environ["NUMERAI_DATA_DIR"] = default_data_dir
        print(f"[bootstrap] Set NUMERAI_DATA_DIR={default_data_dir}")

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
