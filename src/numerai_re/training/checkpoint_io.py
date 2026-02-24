from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def write_signature_checkpoint(path: Path, signature: dict[str, object], payload_fields: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "signature": signature,
    }
    payload.update(payload_fields)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_signature_checkpoint(
    *,
    path: Path,
    expected_signature: dict[str, object],
    resume_mode: str,
    logger: logging.Logger,
    phase_name: str,
) -> dict[str, object] | None:
    if resume_mode == "fresh" or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid checkpoint payload at {path}: expected object.")
        if payload.get("signature") != expected_signature:
            message = f"Checkpoint signature mismatch. path={path} resume_mode={resume_mode}"
            if resume_mode == "strict":
                raise RuntimeError(message)
            logger.warning("phase=%s_checkpoint_mismatch action=restart reason=%s", phase_name, message)
            return None
        return payload
    except Exception as exc:
        if resume_mode == "strict":
            raise
        logger.warning("phase=%s_checkpoint_load_failed action=restart reason=%s", phase_name, exc)
        return None


def count_list_items(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return 0
        items = payload.get(key)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0


def payload_list(payload: dict[str, object] | None, key: str) -> list[Any]:
    if payload is None:
        return []
    items = payload.get(key)
    if not isinstance(items, list):
        return []
    return items
