from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from numerapi import NumerAPI


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_COLUMNS: tuple[str, ...] = (
    "fetched_at",
    "source",
    "model_name",
    "round_number",
    "resolved",
    "corr20",
    "mmc20",
    "bmc20",
    "fnc",
    "payout",
    "stake",
    "primary_score",
    "resolve_date",
)
METHOD_UNAVAILABLE_MESSAGE = "method not available in installed numerapi version"
UNAVAILABLE_METHOD_ERROR = METHOD_UNAVAILABLE_MESSAGE


@dataclass(frozen=True)
class TrackerConfig:
    output_dir: Path
    lookback_rounds: int
    min_resolved_rounds: int
    recommend_threshold: float
    primary_weight_corr: float
    primary_weight_secondary: float


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _resolve_model_name(cli_model_name: str | None) -> str:
    if cli_model_name and cli_model_name.strip():
        return cli_model_name.strip()
    return _required_env("NUMERAI_MODEL_NAME")


def _safe_api_call(label: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        value = fn(*args, **kwargs)
        return {"ok": True, "error": None, "value": value}
    except Exception as exc:  # noqa: BLE001
        logger.warning("phase=api_call_failed call=%s error=%s", label, exc)
        return {"ok": False, "error": str(exc), "value": None}


def _method_unavailable_result() -> dict[str, Any]:
    return {"ok": False, "error": UNAVAILABLE_METHOD_ERROR, "value": None}


def _call_with_attempts(name: str, fn: Any, attempts: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    for args, kwargs in attempts:
        result = _safe_api_call(name, fn, *args, **kwargs)
        if result["ok"]:
            return result
        errors.append(result["error"] or "unknown error")
    return {"ok": False, "error": "; ".join(errors), "value": None}


def _call_if_supported(
    napi: NumerAPI,
    method_name: str,
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> dict[str, Any]:
    method = getattr(napi, method_name, None)
    if method is None:
        return _method_unavailable_result()
    return _call_with_attempts(method_name, method, attempts)


def _build_raw_payload(napi: NumerAPI, model_name: str, model_id: str) -> dict[str, Any]:
    method_results: dict[str, Any] = {}
    method_results["get_current_round"] = _safe_api_call("get_current_round", napi.get_current_round)
    method_results["get_models"] = _safe_api_call("get_models", napi.get_models)

    by_name_or_id_attempts = [
        ((model_name,), {}),
        ((model_id,), {}),
        ((), {"model_name": model_name}),
        ((), {"model_id": model_id}),
    ]
    by_id_or_name_attempts = [
        ((model_id,), {}),
        ((model_name,), {}),
        ((), {"model_id": model_id}),
        ((), {"model_name": model_name}),
    ]

    method_results["daily_model_performances"] = _call_if_supported(
        napi,
        "daily_model_performances",
        by_name_or_id_attempts,
    )
    method_results["daily_user_performances"] = _call_if_supported(
        napi,
        "daily_user_performances",
        by_name_or_id_attempts,
    )
    method_results["round_model_performances_v2"] = _call_if_supported(
        napi,
        "round_model_performances_v2",
        by_id_or_name_attempts,
    )
    method_results["round_model_performances"] = _call_if_supported(
        napi,
        "round_model_performances",
        by_id_or_name_attempts,
    )
    method_results["intra_round_scores"] = _call_if_supported(
        napi,
        "intra_round_scores",
        by_id_or_name_attempts,
    )
    method_results["stake_get"] = _call_if_supported(
        napi,
        "stake_get",
        by_name_or_id_attempts,
    )

    return method_results


def _flatten_records(source: str, value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            out.append({"source": source, **node})
            for child in node.values():
                if isinstance(child, (dict, list)):
                    _walk(child)
            return
        if isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(value)
    return out


def _extract_intra_round_rows(source: str, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    rows: list[dict[str, Any]] = []
    for bucket in value:
        if not isinstance(bucket, dict):
            continue
        round_number = bucket.get("roundNumber")
        score_items = bucket.get("intraRoundSubmissionScores")
        if not isinstance(score_items, list):
            continue

        metric_values: dict[str, Any] = {}
        payout_pending: list[float] = []
        payout_settled: list[float] = []
        resolve_date: str | None = None

        for item in score_items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("displayName", "")).strip().lower()
            metric_values[key] = item.get("value")
            if item.get("date"):
                resolve_date = str(item.get("date"))
            pending = _to_float(item.get("payoutPending"))
            settled = _to_float(item.get("payoutSettled"))
            if pending is not None:
                payout_pending.append(pending)
            if settled is not None:
                payout_settled.append(settled)

        def _first_metric(keys: tuple[str, ...]) -> float | None:
            for k in keys:
                value_raw = metric_values.get(k)
                value_num = _to_float(value_raw)
                if value_num is not None:
                    return value_num
            return None
        corr = _first_metric(("v2_corr20", "corr", "canon_corr", "corr20", "canon_corr20"))
        corr = _first_metric(("v2_corr20", "corr", "canon_corr", "cort20", "canon_cort20"))
        bmc = _first_metric(("bmc", "canon_bmc"))
        mmc = _first_metric(("mmc", "canon_mmc"))
        fnc = _first_metric(("fnc_v3", "canon_fnc_v3"))
        payout = None
        if payout_settled:
            payout = float(sum(payout_settled))
        elif payout_pending:
            payout = float(sum(payout_pending))

        rows.append(
            {
                "source": source,
                "roundNumber": round_number,
                "corr20": corr,
                "bmc20": bmc,
                "mmc20": mmc,
                "fnc": fnc,
                "payout": payout,
                "resolve_date": resolve_date,
            }
        )

    return rows


def _pick(record: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _is_resolved(record: dict[str, Any]) -> bool:
    raw = _pick(record, ("resolved", "isresolved", "is_resolved"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "resolved"}
    if isinstance(raw, (int, float)):
        return bool(raw)

    # Fallback: if a resolve date exists and is in the past, treat as resolved.
    resolve_value = _pick(record, ("resolve", "resolvedate", "resolve_date", "resolved_at"))
    if not resolve_value:
        return False
    try:
        resolve_dt = datetime.fromisoformat(str(resolve_value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if resolve_dt.tzinfo is None:
        resolve_dt = resolve_dt.replace(tzinfo=UTC)
    return resolve_dt <= datetime.now(tz=UTC)


def _normalize_record(record: dict[str, Any], cfg: TrackerConfig, fetched_at: str, model_name: str) -> dict[str, Any]:
    round_number = _to_int(
        _pick(
            record,
            (
                "roundnumber",
                "round_number",
                "round",
                "tournamentround",
                "tournament_round",
            ),
        )
    )
    corr20 = _to_float(_pick(record, ("corr20", "corr_20", "corr")))
    mmc20 = _to_float(_pick(record, ("mmc20", "mmc_20", "mmc")))
    bmc20 = _to_float(_pick(record, ("bmc20", "bmc_20", "bmc")))

    secondary_metric = bmc20 if bmc20 is not None else mmc20
    primary_score = None
    if corr20 is not None and secondary_metric is not None:
        primary_score = cfg.primary_weight_corr * corr20 + cfg.primary_weight_secondary * secondary_metric

    return {
        "fetched_at": fetched_at,
        "source": str(record.get("source", "unknown")),
        "model_name": str(_pick(record, ("modelname", "model_name", "model")) or model_name),
        "round_number": round_number,
        "resolved": _is_resolved(record),
        "corr20": corr20,
        "mmc20": mmc20,
        "bmc20": bmc20,
        "fnc": _to_float(_pick(record, ("fnc", "fncv3", "fnc_v3"))),
        "payout": _to_float(_pick(record, ("payout", "nmrpayout", "nmr_payout"))),
        "stake": _to_float(_pick(record, ("stake", "stakevalue", "stake_value", "atrisk", "at_risk"))),
        "primary_score": primary_score,
        "resolve_date": _pick(record, ("resolve", "resolvedate", "resolve_date", "resolved_at", "date")),
    }


def _normalize_payload(raw_methods: dict[str, Any], cfg: TrackerConfig, fetched_at: str, model_name: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for source, payload in raw_methods.items():
        if not payload.get("ok"):
            continue
        source_records = []
        if source == "intra_round_scores":
            source_records.extend(_extract_intra_round_rows(source, payload.get("value")))
        source_records.extend(_flatten_records(source, payload.get("value")))
        for record in source_records:
            normalized = _normalize_record(record, cfg, fetched_at, model_name)
            if normalized["round_number"] is None:
                continue
            if normalized["primary_score"] is None and normalized["corr20"] is None and normalized["payout"] is None:
                continue
            records.append(normalized)

    if not records:
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["round_number", "fetched_at"]).drop_duplicates(
        subset=["source", "model_name", "round_number", "fetched_at"], keep="last"
    )
    return frame


def _merge_history(history_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    if history_path.exists():
        existing = pd.read_csv(history_path)
        if new_rows.empty:
            merged = existing
        elif existing.empty:
            merged = new_rows.copy()
        else:
            merged = pd.concat([existing, new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()

    if merged.empty:
        pd.DataFrame(columns=list(HISTORY_COLUMNS)).to_csv(history_path, index=False)
        return merged

    merged = merged.reindex(columns=list(HISTORY_COLUMNS))

    merged = merged.sort_values(["round_number", "fetched_at"]).drop_duplicates(
        subset=["source", "model_name", "round_number", "fetched_at"],
        keep="last",
    )
    merged.to_csv(history_path, index=False)
    return merged


def _build_summary(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {
            "history_rows": 0,
            "resolved_rows": 0,
            "round_min": None,
            "round_max": None,
            "sources": [],
        }

    return {
        "history_rows": int(len(history)),
        "resolved_rows": int(history[history["resolved"].map(_coerce_bool)].shape[0]),
        "round_min": int(history["round_number"].min()),
        "round_max": int(history["round_number"].max()),
        "sources": sorted({str(src) for src in history["source"].dropna().unique().tolist()}),
    }


def _score_evidence(history: pd.DataFrame, cfg: TrackerConfig) -> dict[str, Any]:
    resolved_mask = history["resolved"].map(_coerce_bool)
    resolved = history[resolved_mask & history["primary_score"].notna()].copy()
    resolved = resolved.sort_values("round_number", ascending=False)

    if resolved.empty:
        return {
            "evidence_score": 0.0,
            "recommendation": "NOT_ENOUGH_EVIDENCE",
            "lookback_rounds": cfg.lookback_rounds,
            "resolved_rounds_count": 0,
            "mean_primary_score": None,
            "downside_rounds": 0,
            "reasoning": "No resolved rounds with usable primary score were found.",
        }

    latest_per_round = resolved.drop_duplicates(subset=["round_number"], keep="first").head(cfg.lookback_rounds)
    scores = latest_per_round["primary_score"].astype(float)

    resolved_count = int(len(scores))
    mean_primary = float(scores.mean()) if resolved_count else 0.0
    hit_rate = float((scores > 0.0).mean()) if resolved_count else 0.0
    downside_rounds = int((scores < 0.0).sum()) if resolved_count else 0
    std_primary = float(scores.std(ddof=0)) if resolved_count else 0.0

    score = 50.0
    if resolved_count >= cfg.min_resolved_rounds:
        score += 20.0
    score += max(-15.0, min(15.0, (mean_primary / 0.02) * 15.0))
    score += max(-10.0, min(10.0, ((hit_rate - 0.5) / 0.5) * 10.0))
    score -= min(10.0, (downside_rounds / max(1, resolved_count)) * 10.0)
    if std_primary > 0.03:
        score -= 5.0
    score = max(0.0, min(100.0, score))

    recommendation = "NOT_ENOUGH_EVIDENCE"
    if resolved_count >= cfg.min_resolved_rounds and score >= cfg.recommend_threshold:
        recommendation = "EVIDENCE_SUFFICIENT_FOR_MANUAL_STAKING"

    reasons: list[str] = []
    reasons.append(f"resolved_rounds={resolved_count} (minimum={cfg.min_resolved_rounds})")
    reasons.append(f"mean_primary_score={mean_primary:.6f}")
    reasons.append(f"hit_rate={hit_rate:.2%}")
    reasons.append(f"downside_rounds={downside_rounds}")
    reasons.append(f"std_primary_score={std_primary:.6f}")
    reasons.append(f"threshold={cfg.recommend_threshold:.2f}")

    return {
        "evidence_score": round(score, 2),
        "recommendation": recommendation,
        "lookback_rounds": cfg.lookback_rounds,
        "resolved_rounds_count": resolved_count,
        "mean_primary_score": round(mean_primary, 8),
        "downside_rounds": downside_rounds,
        "reasoning": "; ".join(reasons),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_tracker(cfg: TrackerConfig, model_name: str) -> dict[str, Any]:
    public_id = _required_env("NUMERAI_PUBLIC_ID")
    secret_key = _required_env("NUMERAI_SECRET_KEY")

    fetched_at = datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    models = napi.get_models()
    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise RuntimeError(
            f"Configured model '{model_name}' was not found in NumerAI account models. Available: {available or 'none'}"
        )
    model_id = models[model_name]

    raw_methods = _build_raw_payload(napi, model_name=model_name, model_id=model_id)

    output_dir = cfg.output_dir
    raw_dir = output_dir / "raw"
    raw_path = raw_dir / f"{timestamp}.json"
    raw_payload = {
        "fetched_at": fetched_at,
        "model_name": model_name,
        "model_id": model_id,
        "methods": raw_methods,
    }
    _write_json(raw_path, raw_payload)

    normalized = _normalize_payload(raw_methods, cfg, fetched_at=fetched_at, model_name=model_name)
    history_path = output_dir / "history.csv"
    history = _merge_history(history_path, normalized)

    summary_payload = {
        "fetched_at": fetched_at,
        "model_name": model_name,
        "model_id": model_id,
        **_build_summary(history),
    }
    summary_path = output_dir / "latest_summary.json"
    _write_json(summary_path, summary_payload)

    recommendation_payload = {
        "fetched_at": fetched_at,
        "model_name": model_name,
        "model_id": model_id,
        **_score_evidence(history, cfg),
    }
    recommendation_path = output_dir / "latest_recommendation.json"
    _write_json(recommendation_path, recommendation_payload)

    return {
        "raw_path": str(raw_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "recommendation_path": str(recommendation_path),
        "history_rows": int(len(history)),
        "recommendation": recommendation_payload["recommendation"],
        "evidence_score": recommendation_payload["evidence_score"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NumerAI performance metrics and produce staking evidence report.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/performance",
        help="Directory for raw snapshots, normalized history, and recommendation outputs.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="NumerAI model name. Defaults to NUMERAI_MODEL_NAME environment variable.",
    )
    parser.add_argument("--lookback-rounds", type=int, default=int(os.getenv("PERF_LOOKBACK_ROUNDS", "20")))
    parser.add_argument("--min-resolved-rounds", type=int, default=int(os.getenv("PERF_MIN_RESOLVED_ROUNDS", "2")))
    parser.add_argument(
        "--recommend-threshold",
        type=float,
        default=float(os.getenv("PERF_RECOMMEND_THRESHOLD", "65")),
        help="Evidence score threshold for EVIDENCE_SUFFICIENT_FOR_MANUAL_STAKING.",
    )
    parser.add_argument(
        "--primary-weight-corr",
        type=float,
        default=float(os.getenv("PERF_PRIMARY_WEIGHT_CORR", "0.75")),
    )
    parser.add_argument(
        "--primary-weight-secondary",
        type=float,
        default=float(os.getenv("PERF_PRIMARY_WEIGHT_SECONDARY", "2.25")),
        help="Weight for BMC (preferred) or MMC fallback in primary score calculation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.lookback_rounds <= 0:
        raise RuntimeError("--lookback-rounds must be a positive integer")
    if args.min_resolved_rounds <= 0:
        raise RuntimeError("--min-resolved-rounds must be a positive integer")

    cfg = TrackerConfig(
        output_dir=(ROOT / args.output_dir).resolve(),
        lookback_rounds=int(args.lookback_rounds),
        min_resolved_rounds=int(args.min_resolved_rounds),
        recommend_threshold=float(args.recommend_threshold),
        primary_weight_corr=float(args.primary_weight_corr),
        primary_weight_secondary=float(args.primary_weight_secondary),
    )

    result = run_tracker(cfg, model_name=_resolve_model_name(args.model_name))
    print(
        "PERFORMANCE_TRACKER_OK "
        f"history_rows={result['history_rows']} "
        f"evidence_score={result['evidence_score']} "
        f"recommendation={result['recommendation']}"
    )
    print(f"raw_path={result['raw_path']}")
    print(f"history_path={result['history_path']}")
    print(f"summary_path={result['summary_path']}")
    print(f"recommendation_path={result['recommendation_path']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"PERFORMANCE_TRACKER_ABORT: {exc}", file=sys.stderr)
        raise SystemExit(2)
