from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REQUIRED_ARTIFACT_FILES = ("features.json", "train_manifest.json", "postprocess_config.json")
REQUIRED_POSTPROCESS_KEYS = {
    "schema_version",
    "submission_transform",
    "blend_alpha",
    "bench_neutralize_prop",
    "payout_weight_corr",
    "payout_weight_bmc",
    "bench_cols_used",
}


def _fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"P0 FAIL: {message}")


def _warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"P1/P2 WARN: {message}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_artifact_contract(artifact_dir: Path, expected_dataset_version: str, failures: list[str], warnings: list[str]) -> None:
    for filename in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / filename).exists():
            _fail(f"Artifact missing required file: {filename}", failures)
    if failures:
        return

    feature_cols = _read_json(artifact_dir / "features.json")
    if not isinstance(feature_cols, list) or not feature_cols or not all(isinstance(col, str) and col for col in feature_cols):
        _fail("features.json must be a non-empty list[str].", failures)

    manifest = _read_json(artifact_dir / "train_manifest.json")
    if not isinstance(manifest, dict):
        _fail("train_manifest.json must be a JSON object.", failures)
        return

    model_files = manifest.get("model_files")
    if not isinstance(model_files, list) or not model_files or not all(isinstance(name, str) and name.strip() for name in model_files):
        model_file = manifest.get("model_file")
        if isinstance(model_file, str) and model_file.strip():
            model_files = [model_file]
        else:
            _fail("Manifest must include non-empty model_files or model_file.", failures)
            model_files = []
    for model_file in model_files:
        if not (artifact_dir / model_file).exists():
            _fail(f"Manifest references missing model file: {model_file}", failures)

    manifest_dataset = manifest.get("dataset_version")
    if manifest_dataset != expected_dataset_version:
        _warn(
            f"Manifest dataset_version {manifest_dataset!r} != expected runtime {expected_dataset_version!r}.",
            warnings,
        )

    post_cfg = _read_json(artifact_dir / "postprocess_config.json")
    if not isinstance(post_cfg, dict):
        _fail("postprocess_config.json must be a JSON object.", failures)
        return
    missing_post = sorted(REQUIRED_POSTPROCESS_KEYS - set(post_cfg.keys()))
    if missing_post:
        _fail(f"postprocess_config.json missing required keys: {missing_post}", failures)
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


def _validate_import_consistency(failures: list[str]) -> None:
    def _imports_apply_postprocess(path: Path) -> bool:
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom) or node.module != "postprocess":
                continue
            imported = {alias.name for alias in node.names}
            if {"PostprocessConfig", "apply_postprocess"}.issubset(imported):
                return True
        return False

    if not _imports_apply_postprocess(SRC / "train_colab.py") or not _imports_apply_postprocess(SRC / "inference.py"):
        _fail("Training and inference must both import apply_postprocess from src/postprocess.py.", failures)


def _validate_guardrails(warnings: list[str]) -> None:
    infer_src = (SRC / "inference.py").read_text()
    if "exposure_sample_rows" not in infer_src or "rng.choice" not in infer_src:
        _warn("Exposure sampling guardrail not detected in inference quality gates.", warnings)
    if "read_parquet(live_path, columns=" not in infer_src:
        _warn("Inference live parquet column selection guardrail not detected.", warnings)


def _lint_sources(failures: list[str]) -> None:
    for path in (SRC / "train_colab.py", SRC / "inference.py", SRC / "postprocess.py"):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            _fail(f"Syntax error in {path.name}: {exc}", failures)


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
        _fail(f"Artifact directory not found: {artifact_dir}", failures)
    else:
        _validate_artifact_contract(artifact_dir, args.dataset_version, failures, warnings)

    _validate_import_consistency(failures)
    _validate_guardrails(warnings)
    _lint_sources(failures)
    print(
        f"VALIDATION_SUMMARY failures={len(failures)} warnings={len(warnings)} dry_run={bool(args.dry_run)} artifact_dir={artifact_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
