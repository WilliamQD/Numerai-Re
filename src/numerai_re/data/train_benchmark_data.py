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

    bench_train, bench_cols = align_bench_to_ids(
        id_train,
        bench_train_df,
        id_col,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    bench_valid, bench_cols_valid = align_bench_to_ids(
        id_valid,
        bench_valid_df,
        id_col,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    if bench_cols_valid != bench_cols:
        raise RuntimeError(
            f"Benchmark column mismatch: train has {bench_cols}, validation has {bench_cols_valid}."
        )

    bench_all_df = pl.concat([bench_train_df, bench_valid_df], how="vertical")
    bench_all, bench_cols_all = align_bench_to_ids(
        id_all,
        bench_all_df,
        id_col,
        drop_sparse_columns=drop_sparse_columns,
        max_null_ratio_per_column=max_null_ratio_per_column,
        min_benchmark_columns=min_benchmark_columns,
    )
    if bench_cols_all != bench_cols:
        raise RuntimeError(f"Benchmark column mismatch after concat: expected {bench_cols}, got {bench_cols_all}.")

    logger.info(
        "phase=bench_alignment_ok train_cols=%d valid_cols=%d all_cols=%d",
        bench_train.shape[1],
        bench_valid.shape[1],
        bench_all.shape[1],
    )

    del bench_train_df, bench_valid_df, bench_all_df
    gc.collect()

    return AlignedBenchmarks(train=bench_train, valid=bench_valid, all=bench_all, cols=bench_cols)
