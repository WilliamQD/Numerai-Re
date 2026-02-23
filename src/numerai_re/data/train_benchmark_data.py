from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from numerai_re.data.bench_matrix_builder import align_bench_to_ids
from numerai_re.data.benchmarks import load_benchmark_frame


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignedBenchmarks:
    train: np.ndarray
    valid: np.ndarray
    all: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    all_mask: np.ndarray
    cols: list[str]


def load_and_align_benchmarks(
    id_train: np.ndarray,
    id_valid: np.ndarray,
    id_all: np.ndarray,
    benchmark_paths: dict[str, Path],
    id_col: str,
    *,
    drop_sparse_columns: bool = True,
    max_null_ratio_per_column: float = 0.0,
    min_benchmark_columns: int = 1,
) -> AlignedBenchmarks:
    bench_train_df = load_benchmark_frame(benchmark_paths["train"])
    bench_valid_df = load_benchmark_frame(benchmark_paths["validation"])

    logger.info(
        "phase=bench_quality split=train rows=%d cols=%d unique_ids=%d",
        bench_train_df.height,
        len(bench_train_df.columns),
        bench_train_df.select(pl.col(id_col).n_unique()).item() if id_col in bench_train_df.columns else -1,
    )
    logger.info(
        "phase=bench_quality split=validation rows=%d cols=%d unique_ids=%d",
        bench_valid_df.height,
        len(bench_valid_df.columns),
        bench_valid_df.select(pl.col(id_col).n_unique()).item() if id_col in bench_valid_df.columns else -1,
    )

    bench_train = align_bench_to_ids(
        id_train,
        bench_train_df,
        id_col,
        allow_partial_coverage=True,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    bench_valid = align_bench_to_ids(
        id_valid,
        bench_valid_df,
        id_col,
        allow_partial_coverage=False,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    if bench_valid.cols != bench_train.cols:
        raise RuntimeError(
            f"Benchmark column mismatch: train has {bench_train.cols}, validation has {bench_valid.cols}."
        )

    bench_all_df = pl.concat([bench_train_df, bench_valid_df], how="vertical")
    bench_all = align_bench_to_ids(
        id_all,
        bench_all_df,
        id_col,
        allow_partial_coverage=True,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    if bench_all.cols != bench_train.cols:
        raise RuntimeError(f"Benchmark column mismatch after concat: expected {bench_train.cols}, got {bench_all.cols}.")

    train_coverage_ratio = float(np.mean(bench_train.coverage_mask))
    valid_coverage_ratio = float(np.mean(bench_valid.coverage_mask))
    all_coverage_ratio = float(np.mean(bench_all.coverage_mask))
    if train_coverage_ratio < 1.0:
        logger.warning(
            "phase=bench_train_partial_coverage covered_rows=%d total_rows=%d coverage=%.6f policy=allow_partial",
            int(np.count_nonzero(bench_train.coverage_mask)),
            int(len(bench_train.coverage_mask)),
            train_coverage_ratio,
        )

    logger.info(
        "phase=bench_alignment_ok train_cols=%d valid_cols=%d all_cols=%d train_coverage=%.6f valid_coverage=%.6f all_coverage=%.6f",
        bench_train.matrix.shape[1],
        bench_valid.matrix.shape[1],
        bench_all.matrix.shape[1],
        train_coverage_ratio,
        valid_coverage_ratio,
        all_coverage_ratio,
    )

    del bench_train_df, bench_valid_df, bench_all_df
    gc.collect()

    return AlignedBenchmarks(
        train=bench_train.matrix,
        valid=bench_valid.matrix,
        all=bench_all.matrix,
        train_mask=bench_train.coverage_mask,
        valid_mask=bench_valid.coverage_mask,
        all_mask=bench_all.coverage_mask,
        cols=bench_train.cols,
    )
