from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from numerai_re.contracts import (  # noqa: E402
    MANIFEST_FILENAME,
    POSTPROCESS_FILENAME,
    REQUIRED_POSTPROCESS_KEYS,
    load_features_by_model,
    load_manifest,
    load_union_features,
    resolve_model_files,
    validate_model_files_exist,
)


REQUIRED_ARTIFACT_FILES = (MANIFEST_FILENAME, POSTPROCESS_FILENAME)


def _safe_print(prefix: str, message: str) -> None:
    output = f"{prefix}: {message}"
    try:
        print(output)
    except UnicodeEncodeError:
        print(output.encode("ascii", errors="backslashreplace").decode("ascii"))


def _fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    _safe_print("P0 FAIL", message)


def _warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    _safe_print("P1/P2 WARN", message)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_artifact_contract(artifact_dir: Path, expected_dataset_version: str, failures: list[str], warnings: list[str]) -> None:
    for filename in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / filename).exists():
            _fail(f"Artifact missing required file: {filename}", failures)
    if failures:
        return

    try:
        manifest = load_manifest(artifact_dir, label="pipeline validation")
        model_files = resolve_model_files(manifest, label="pipeline validation")
        validate_model_files_exist(artifact_dir, model_files, label="pipeline validation")
        load_union_features(artifact_dir, manifest, label="pipeline validation")
        load_features_by_model(artifact_dir, manifest, model_files, label="pipeline validation")
    except RuntimeError as exc:
        _fail(str(exc), failures)
        return

    manifest_dataset = manifest.get("dataset_version")
    if manifest_dataset != expected_dataset_version:
        _warn(
            f"Manifest dataset_version {manifest_dataset!r} != expected runtime {expected_dataset_version!r}.",
            warnings,
        )

    post_cfg = _read_json(artifact_dir / POSTPROCESS_FILENAME)
    if not isinstance(post_cfg, dict):
        _fail(f"{POSTPROCESS_FILENAME} must be a JSON object.", failures)
        return
    missing_post = sorted(REQUIRED_POSTPROCESS_KEYS - set(post_cfg.keys()))
    if missing_post:
        _fail(f"{POSTPROCESS_FILENAME} missing required keys: {missing_post}", failures)
    if post_cfg.get("submission_transform") not in {"rank_01", "gauss_rank"}:
        _fail("postprocess submission_transform must be one of: rank_01, gauss_rank.", failures)
    for key in ("blend_alpha", "bench_neutralize_prop"):
        value = post_cfg.get(key)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            _fail(f"postprocess {key} must be numeric.", failures)
            continue
        if not 0.0 <= value_f <= 1.0:
            _fail(f"postprocess {key} must be in [0,1].", failures)
    bench_cols = post_cfg.get("bench_cols_used")
    if not isinstance(bench_cols, list) or not bench_cols:
        _fail("postprocess bench_cols_used must be a non-empty list.", failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate core NumerAI pipeline contracts and guardrails.")
    parser.add_argument("--artifact-dir", default="artifacts", help="Artifact directory to validate (default: artifacts).")
    parser.add_argument("--dataset-version", default="v5.2", help="Expected runtime dataset version.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validations without external services; expects local artifacts only.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    artifact_dir = (ROOT / args.artifact_dir).resolve()
    if not artifact_dir.exists():
        if args.dry_run:
            _warn(f"Artifact directory not found in dry-run mode, skipping artifact checks: {artifact_dir}", warnings)
        else:
            _fail(f"Artifact directory not found: {artifact_dir}", failures)
    else:
        _validate_artifact_contract(artifact_dir, args.dataset_version, failures, warnings)

    _safe_print(
        "VALIDATION_SUMMARY",
        f"failures={len(failures)} warnings={len(warnings)} dry_run={bool(args.dry_run)} artifact_dir={artifact_dir}",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
