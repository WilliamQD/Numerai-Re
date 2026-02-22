from __future__ import annotations

import numpy as np

try:
    from scipy.stats import norm, rankdata
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for numerai_metrics (rankdata + norm.ppf). Install scipy.") from exc


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
) -> float:
    if bench.ndim != 2 or bench.shape[0] != pred.shape[0]:
        raise ValueError("bench must be a 2D matrix aligned to pred rows.")

    pred_gauss = gauss_rank_by_era(pred, era)
    bench_gauss = np.empty_like(bench, dtype=np.float32)
    for idx in range(bench.shape[1]):
        bench_gauss[:, idx] = gauss_rank_by_era(bench[:, idx], era)

    pred_resid = neutralize_to_matrix(pred_gauss, bench_gauss, proportion=neutralize_prop)
    vals: list[float] = []
    for e in np.unique(era):
        mask = era == e
        centered_target = target[mask].astype(np.float64, copy=False)
        centered_target = centered_target - centered_target.mean()
        vals.append(float(np.mean(pred_resid[mask].astype(np.float64, copy=False) * centered_target)))
    return float(np.mean(vals)) if vals else 0.0
