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
