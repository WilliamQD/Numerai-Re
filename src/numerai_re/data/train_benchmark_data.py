from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from numerai_re.data.bench_matrix_builder import align_bench_to_ids
from numerai_re.data.benchmarks import load_benchmark_frame


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
) -> AlignedBenchmarks:
    bench_train_df = load_benchmark_frame(benchmark_paths["train"])
    bench_valid_df = load_benchmark_frame(benchmark_paths["validation"])

    bench_train, bench_cols = align_bench_to_ids(id_train, bench_train_df, id_col)
    bench_valid, bench_cols_valid = align_bench_to_ids(id_valid, bench_valid_df, id_col)
    if bench_cols_valid != bench_cols:
        raise RuntimeError(
            f"Benchmark column mismatch: train has {bench_cols}, validation has {bench_cols_valid}."
        )

    bench_all_df = pl.concat([bench_train_df, bench_valid_df], how="vertical")
    bench_all, bench_cols_all = align_bench_to_ids(id_all, bench_all_df, id_col)
    if bench_cols_all != bench_cols:
        raise RuntimeError(f"Benchmark column mismatch after concat: expected {bench_cols}, got {bench_cols_all}.")

    del bench_train_df, bench_valid_df, bench_all_df
    gc.collect()

    return AlignedBenchmarks(train=bench_train, valid=bench_valid, all=bench_all, cols=bench_cols)
