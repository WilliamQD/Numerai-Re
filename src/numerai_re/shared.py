"""Shared utilities: era parsing, status reporting, feature sampling, and NumerAI metrics."""

from __future__ import annotations

import hashlib
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.stats import norm, rankdata
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for numerai_metrics (rankdata + norm.ppf). Install scipy.") from exc


# --- Era utilities ---

_ERA_RE = re.compile(r"(\d+)")


def era_to_int(arr) -> np.ndarray:
    """Convert era labels to integer era numbers.

    Args:
        arr: Array-like of era values, e.g. integers or strings like ``era123``.

    Returns:
        A ``np.int32`` array of parsed era numbers.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int32, copy=False)

    out = np.empty(len(a), dtype=np.int32)
    for i, value in enumerate(a):
        match = _ERA_RE.search(str(value))
        if not match:
            raise ValueError(f"Could not parse era number from: {value!r}")
        out[i] = int(match.group(1))
    return out


# --- Status reporter ---


@dataclass
class RuntimeStatusReporter:
    logger: Any
    interval_seconds: float = 60.0
    name: str = "train"

    def __post_init__(self) -> None:
        self._last_emit = 0.0
        self._last_rendered_len = 0
        self._active = False
        self._interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())

    def _format(self, phase: str, fields: dict[str, object]) -> str:
        keyvals = " ".join(f"{key}={value}" for key, value in fields.items())
        if keyvals:
            return f"[{self.name}] phase={phase} {keyvals}"
        return f"[{self.name}] phase={phase}"

    def update(self, phase: str, *, force: bool = False, **fields: object) -> None:
        now = time.monotonic()
        if not force and self._last_emit > 0 and (now - self._last_emit) < self.interval_seconds:
            return

        message = self._format(phase, fields)
        self._last_emit = now
        self._active = True

        if self._interactive:
            padded = message
            if self._last_rendered_len > len(message):
                padded += " " * (self._last_rendered_len - len(message))
            print(f"\r{padded}", end="", file=sys.stderr, flush=True)
            self._last_rendered_len = len(message)
            return

        self.logger.info("phase=status_update message=%s", message)

    def clear(self) -> None:
        if self._interactive and self._active:
            print(file=sys.stderr, flush=True)
        self._active = False
        self._last_rendered_len = 0

    def __enter__(self) -> "RuntimeStatusReporter":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.clear()


# --- Feature sampling ---


def sample_features_for_seed(
    feature_pool: list[str],
    seed: int,
    model_index: int,
    n_models: int,
    max_features_per_model: int,
    master_seed: int = 0,
    strategy: str = "sharded_shuffle",
) -> list[str]:
    if max_features_per_model <= 0 or max_features_per_model >= len(feature_pool):
        return list(feature_pool)
    if strategy != "sharded_shuffle":
        raise ValueError(f"Unsupported FEATURE_SAMPLING_STRATEGY: {strategy}")
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    if model_index < 0 or model_index >= n_models:
        raise ValueError("model_index must be in [0, n_models)")

    shuffled = list(feature_pool)
    random.Random(master_seed).shuffle(shuffled)
    shard_size = max(1, len(shuffled) // n_models)
    start = model_index * shard_size
    end = len(shuffled) if model_index == n_models - 1 else min(len(shuffled), start + shard_size)
    selected = list(shuffled[start:end])
    if len(selected) < max_features_per_model:
        remaining = [col for col in shuffled if col not in set(selected)]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: max_features_per_model - len(selected)])
    return sorted(selected[:max_features_per_model])


def features_hash(feature_cols: list[str]) -> str:
    payload = "\n".join(feature_cols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# --- NumerAI metrics ---


def _tie_rank_gauss(pred: np.ndarray) -> np.ndarray:
    r = rankdata(pred, method="average")
    u = (r - 0.5) / len(r)
    return norm.ppf(u)


def _pow_1p5_signed(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** 1.5)


def numerai_corr(pred: np.ndarray, target: np.ndarray) -> float:
    pg = _tie_rank_gauss(pred.astype(np.float64, copy=False))
    tc = target.astype(np.float64, copy=False) - float(np.mean(target))
    pg = _pow_1p5_signed(pg)
    tc = _pow_1p5_signed(tc)

    pg = pg - pg.mean()
    tc = tc - tc.mean()
    denom = np.sqrt(np.sum(pg * pg)) * np.sqrt(np.sum(tc * tc))
    if denom == 0:
        return 0.0
    return float(np.sum(pg * tc) / denom)


def mean_per_era_numerai_corr(pred: np.ndarray, target: np.ndarray, era: np.ndarray) -> float:
    eras = np.unique(era)
    corrs = []
    for e in eras:
        m = era == e
        corrs.append(numerai_corr(pred[m], target[m]))
    return float(np.mean(corrs)) if corrs else 0.0


def gauss_rank_by_era(pred: np.ndarray, era: np.ndarray) -> np.ndarray:
    out = np.empty(len(pred), dtype=np.float32)
    for e in np.unique(era):
        mask = era == e
        r = rankdata(pred[mask], method="average")
        u = (r - 0.5) / len(r)
        out[mask] = norm.ppf(u).astype(np.float32, copy=False)
    return out


def neutralize_to_matrix(y: np.ndarray, x: np.ndarray, proportion: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    y64 = y.astype(np.float64, copy=False)
    x64 = x.astype(np.float64, copy=False)
    ones = np.ones((x64.shape[0], 1), dtype=np.float64)
    z = np.concatenate([ones, x64], axis=1)
    beta, *_ = np.linalg.lstsq(z, y64, rcond=None)
    y_hat = z @ beta
    out = y64 - proportion * y_hat
    std = float(out.std())
    if std < eps:
        return out.astype(np.float32, copy=False)
    return (out / std).astype(np.float32, copy=False)


def bmc_mean_per_era(
    pred: np.ndarray,
    target: np.ndarray,
    era: np.ndarray,
    bench: np.ndarray,
    neutralize_prop: float = 1.0,
    *,
    bench_gauss: np.ndarray | None = None,
) -> float:
    if bench.ndim != 2 or bench.shape[0] != pred.shape[0]:
        raise ValueError("bench must be a 2D matrix aligned to pred rows.")

    pred_gauss = gauss_rank_by_era(pred, era)
    if bench_gauss is None:
        bench_gauss = np.empty_like(bench, dtype=np.float32)
        for idx in range(bench.shape[1]):
            bench_gauss[:, idx] = gauss_rank_by_era(bench[:, idx], era)
    elif bench_gauss.shape != bench.shape:
        raise ValueError("bench_gauss must have the same shape as bench.")

    pred_resid = neutralize_to_matrix(pred_gauss, bench_gauss, proportion=neutralize_prop)
    vals: list[float] = []
    for e in np.unique(era):
        mask = era == e
        centered_target = target[mask].astype(np.float64, copy=False)
        centered_target = centered_target - centered_target.mean()
        vals.append(float(np.mean(pred_resid[mask].astype(np.float64, copy=False) * centered_target)))
    return float(np.mean(vals)) if vals else 0.0
