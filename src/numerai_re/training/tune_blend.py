from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np
import wandb

from numerai_re.common.status_reporter import RuntimeStatusReporter
from numerai_re.metrics.numerai_metrics import bmc_mean_per_era, gauss_rank_by_era, mean_per_era_numerai_corr
from numerai_re.inference.postprocess import PostprocessConfig, apply_postprocess


logger = logging.getLogger(__name__)


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
    *,
    bench_gauss: np.ndarray | None = None,
) -> np.ndarray:
    cfg = PostprocessConfig(
        schema_version=1,
        submission_transform="gauss_rank",
        blend_alpha=float(alpha),
        bench_neutralize_prop=float(neutralize_prop),
    )
    return apply_postprocess(pred_raw.astype(np.float32, copy=False), era, cfg, bench=bench, bench_gauss=bench_gauss)


def tune_blend_on_windows(
    windows: list[dict[str, np.ndarray | int]],
    alpha_grid: tuple[float, ...],
    prop_grid: tuple[float, ...],
    payout_weight_corr: float,
    payout_weight_bmc: float,
    *,
    status: RuntimeStatusReporter | None = None,
) -> BlendTuneReport:
    search_rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_metrics: tuple[float, float, float] | None = None
    total_combos = len(alpha_grid) * len(prop_grid)
    combo_index = 0
    tune_start = time.monotonic()

    prepared_windows: list[dict[str, np.ndarray | int]] = []
    for window in windows:
        bench = window["bench"]  # type: ignore[index]
        era = window["era"]  # type: ignore[index]
        bench_gauss = np.empty_like(bench, dtype=np.float32)
        for idx in range(bench.shape[1]):
            bench_gauss[:, idx] = gauss_rank_by_era(bench[:, idx], era)
        prepared_windows.append({**window, "bench_gauss": bench_gauss})

    logger.info(
        "phase=blend_tune_started n_windows=%d total_combos=%d payout_weight_corr=%.3f payout_weight_bmc=%.3f",
        len(prepared_windows),
        total_combos,
        float(payout_weight_corr),
        float(payout_weight_bmc),
    )

    for alpha in alpha_grid:
        for prop in prop_grid:
            combo_index += 1
            combo_start = time.monotonic()
            scores: list[float] = []
            corrs: list[float] = []
            bmcs: list[float] = []
            for window_index, window in enumerate(prepared_windows, start=1):
                if status is not None:
                    status.update(
                        "blend_tune",
                        combo=f"{combo_index}/{total_combos}",
                        window=f"{window_index}/{len(windows)}",
                        alpha=f"{float(alpha):.2f}",
                        prop=f"{float(prop):.2f}",
                    )
                pred_raw = window["pred_raw"]  # type: ignore[index]
                target = window["target"]  # type: ignore[index]
                era = window["era"]  # type: ignore[index]
                bench = window["bench"]  # type: ignore[index]
                bench_gauss = window["bench_gauss"]  # type: ignore[index]
                blended = blended_predictions(
                    pred_raw,
                    era,
                    bench,
                    alpha=float(alpha),
                    neutralize_prop=float(prop),
                    bench_gauss=bench_gauss,
                )
                corr = float(mean_per_era_numerai_corr(blended, target, era))
                bmc = float(
                    bmc_mean_per_era(
                        blended,
                        target,
                        era,
                        bench,
                        neutralize_prop=float(prop),
                        bench_gauss=bench_gauss,
                    )
                )
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

            elapsed_s = max(0.0, time.monotonic() - tune_start)
            combo_elapsed_s = max(0.0, time.monotonic() - combo_start)
            eta_s = (elapsed_s / combo_index) * max(total_combos - combo_index, 0)
            logger.info(
                "phase=blend_tune_progress combo=%d total=%d alpha=%.2f prop=%.2f combo_elapsed_s=%.1f elapsed_s=%.1f eta_s=%.1f",
                combo_index,
                total_combos,
                float(alpha),
                float(prop),
                combo_elapsed_s,
                elapsed_s,
                eta_s,
            )

    if best_row is None:
        raise RuntimeError("Blend tuning did not produce any candidate rows.")

    best_alpha = float(best_row["alpha"])
    best_prop = float(best_row["prop"])
    window_rows: list[dict[str, float | int]] = []
    for window in prepared_windows:
        pred_raw = window["pred_raw"]  # type: ignore[index]
        target = window["target"]  # type: ignore[index]
        era = window["era"]  # type: ignore[index]
        bench = window["bench"]  # type: ignore[index]
        bench_gauss = window["bench_gauss"]  # type: ignore[index]
        blended = blended_predictions(
            pred_raw,
            era,
            bench,
            alpha=best_alpha,
            neutralize_prop=best_prop,
            bench_gauss=bench_gauss,
        )
        corr = float(mean_per_era_numerai_corr(blended, target, era))
        bmc = float(
            bmc_mean_per_era(
                blended,
                target,
                era,
                bench,
                neutralize_prop=best_prop,
                bench_gauss=bench_gauss,
            )
        )
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
