from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wandb

from numerai_re.metrics.numerai_metrics import bmc_mean_per_era, mean_per_era_numerai_corr
from numerai_re.inference.postprocess import PostprocessConfig, apply_postprocess


@dataclass(frozen=True)
class BlendTuneReport:
    best_alpha: float
    best_prop: float
    best_hit_rate: float
    best_mean_score: float
    best_mean_corr: float
    best_mean_bmc: float
    search_rows: list[dict[str, float]]
    window_rows: list[dict[str, float | int]]


def blended_predictions(
    pred_raw: np.ndarray,
    era: np.ndarray,
    bench: np.ndarray,
    alpha: float,
    neutralize_prop: float,
) -> np.ndarray:
    cfg = PostprocessConfig(
        schema_version=1,
        submission_transform="gauss_rank",
        blend_alpha=float(alpha),
        bench_neutralize_prop=float(neutralize_prop),
    )
    return apply_postprocess(pred_raw.astype(np.float32, copy=False), era, cfg, bench=bench)


def tune_blend_on_windows(
    windows: list[dict[str, np.ndarray | int]],
    alpha_grid: tuple[float, ...],
    prop_grid: tuple[float, ...],
    payout_weight_corr: float,
    payout_weight_bmc: float,
) -> BlendTuneReport:
    search_rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_metrics: tuple[float, float, float] | None = None

    for alpha in alpha_grid:
        for prop in prop_grid:
            scores: list[float] = []
            corrs: list[float] = []
            bmcs: list[float] = []
            for window in windows:
                pred_raw = window["pred_raw"]  # type: ignore[index]
                target = window["target"]  # type: ignore[index]
                era = window["era"]  # type: ignore[index]
                bench = window["bench"]  # type: ignore[index]
                blended = blended_predictions(pred_raw, era, bench, alpha=float(alpha), neutralize_prop=float(prop))
                corr = float(mean_per_era_numerai_corr(blended, target, era))
                bmc = float(bmc_mean_per_era(blended, target, era, bench, neutralize_prop=float(prop)))
                score = float((payout_weight_corr * corr) + (payout_weight_bmc * bmc))
                corrs.append(corr)
                bmcs.append(bmc)
                scores.append(score)

            score_arr = np.asarray(scores, dtype=np.float64)
            corr_arr = np.asarray(corrs, dtype=np.float64)
            bmc_arr = np.asarray(bmcs, dtype=np.float64)
            row = {
                "alpha": float(alpha),
                "prop": float(prop),
                "hit_rate": float(np.mean(score_arr > 0.0)),
                "mean_score": float(np.mean(score_arr)),
                "mean_corr": float(np.mean(corr_arr)),
                "mean_bmc": float(np.mean(bmc_arr)),
            }
            search_rows.append(row)
            metrics = (row["hit_rate"], row["mean_score"], row["mean_corr"])
            if best_row is None or metrics > best_metrics:
                best_row = row
                best_metrics = metrics

    if best_row is None:
        raise RuntimeError("Blend tuning did not produce any candidate rows.")

    best_alpha = float(best_row["alpha"])
    best_prop = float(best_row["prop"])
    window_rows: list[dict[str, float | int]] = []
    for window in windows:
        pred_raw = window["pred_raw"]  # type: ignore[index]
        target = window["target"]  # type: ignore[index]
        era = window["era"]  # type: ignore[index]
        bench = window["bench"]  # type: ignore[index]
        blended = blended_predictions(pred_raw, era, bench, alpha=best_alpha, neutralize_prop=best_prop)
        corr = float(mean_per_era_numerai_corr(blended, target, era))
        bmc = float(bmc_mean_per_era(blended, target, era, bench, neutralize_prop=best_prop))
        score = float((payout_weight_corr * corr) + (payout_weight_bmc * bmc))
        window_rows.append(
            {
                "window_id": int(window["window_id"]),  # type: ignore[arg-type]
                "alpha": best_alpha,
                "prop": best_prop,
                "corr": corr,
                "bmc": bmc,
                "score": score,
                "hit": int(score > 0.0),
            }
        )

    search_table = wandb.Table(columns=["alpha", "prop", "hit_rate", "mean_score", "mean_corr", "mean_bmc"])
    for row in search_rows:
        search_table.add_data(
            row["alpha"],
            row["prop"],
            row["hit_rate"],
            row["mean_score"],
            row["mean_corr"],
            row["mean_bmc"],
        )
    best_windows_table = wandb.Table(columns=["window_id", "alpha", "prop", "corr", "bmc", "score", "hit"])
    for row in window_rows:
        best_windows_table.add_data(
            int(row["window_id"]),
            float(row["alpha"]),
            float(row["prop"]),
            float(row["corr"]),
            float(row["bmc"]),
            float(row["score"]),
            int(row["hit"]),
        )
    wandb.log(
        {
            "blend/search": search_table,
            "blend/windows": best_windows_table,
            "blend/best_alpha": best_alpha,
            "blend/best_prop": best_prop,
            "blend/best_hit_rate": float(best_row["hit_rate"]),
            "blend/best_mean_score": float(best_row["mean_score"]),
            "blend/best_mean_corr": float(best_row["mean_corr"]),
            "blend/best_mean_bmc": float(best_row["mean_bmc"]),
        }
    )
    return BlendTuneReport(
        best_alpha=best_alpha,
        best_prop=best_prop,
        best_hit_rate=float(best_row["hit_rate"]),
        best_mean_score=float(best_row["mean_score"]),
        best_mean_corr=float(best_row["mean_corr"]),
        best_mean_bmc=float(best_row["mean_bmc"]),
        search_rows=search_rows,
        window_rows=window_rows,
    )
