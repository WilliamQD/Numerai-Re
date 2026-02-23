from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

from numerai_re.metrics.numerai_metrics import neutralize_to_matrix


@dataclass(frozen=True)
class PostprocessConfig:
    schema_version: int
    submission_transform: str = "rank_01"
    blend_alpha: float = 1.0
    bench_neutralize_prop: float = 0.0
    payout_weight_corr: float = 0.75
    payout_weight_bmc: float = 2.25
    bench_cols_used: tuple[str, ...] = ()
    feature_neutralize_prop: float = 0.0
    feature_neutralize_n_features: int = 0
    feature_neutralize_seed: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "PostprocessConfig":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid postprocess_config.json: expected top-level JSON object.")
        return cls(
            schema_version=int(payload["schema_version"]),
            submission_transform=str(payload.get("submission_transform", "rank_01")),
            blend_alpha=float(payload["blend_alpha"]),
            bench_neutralize_prop=float(payload["bench_neutralize_prop"]),
            payout_weight_corr=float(payload["payout_weight_corr"]),
            payout_weight_bmc=float(payload["payout_weight_bmc"]),
            bench_cols_used=tuple(str(col) for col in payload["bench_cols_used"]),
            feature_neutralize_prop=float(payload.get("feature_neutralize_prop", 0.0)),
            feature_neutralize_n_features=int(payload.get("feature_neutralize_n_features", 0)),
            feature_neutralize_seed=int(payload.get("feature_neutralize_seed", 0)),
        )


def _gauss_rank_by_era(values: np.ndarray, era: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float32)
    for era_value in np.unique(era):
        mask = era == era_value
        ranks = rankdata(values[mask], method="average")
        out[mask] = norm.ppf((ranks - 0.5) / len(ranks)).astype(np.float32, copy=False)
    return out


def _rank_01_by_era(values: np.ndarray, era: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float32)
    for era_value in np.unique(era):
        mask = era == era_value
        ranks = rankdata(values[mask], method="average")
        out[mask] = ((ranks - 0.5) / len(ranks)).astype(np.float32, copy=False)
    return out


def apply_postprocess(
    pred_raw: np.ndarray,
    era: np.ndarray,
    cfg: PostprocessConfig,
    bench: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> np.ndarray:
    pred_gauss = _gauss_rank_by_era(pred_raw, era)
    blended = pred_gauss

    if bench is not None and bench.size:
        bench_gauss = np.empty_like(bench, dtype=np.float32)
        for idx in range(bench.shape[1]):
            bench_gauss[:, idx] = _gauss_rank_by_era(bench[:, idx], era)
        resid = neutralize_to_matrix(pred_gauss, bench_gauss, proportion=cfg.bench_neutralize_prop)
        blended = (cfg.blend_alpha * pred_gauss) + ((1.0 - cfg.blend_alpha) * resid)

    if (
        features is not None
        and features.size
        and cfg.feature_neutralize_prop > 0.0
        and cfg.feature_neutralize_n_features > 0
    ):
        out = np.empty_like(blended, dtype=np.float32)
        for era_value in np.unique(era):
            mask = era == era_value
            out[mask] = neutralize_to_matrix(
                blended[mask],
                features[mask],
                proportion=cfg.feature_neutralize_prop,
            )
        blended = out

    if cfg.submission_transform == "gauss_rank":
        return _gauss_rank_by_era(blended, era)
    if cfg.submission_transform == "rank_01":
        return _rank_01_by_era(blended, era)
    raise RuntimeError(f"Unsupported submission_transform in postprocess config: {cfg.submission_transform!r}")
