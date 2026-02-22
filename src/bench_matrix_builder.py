from __future__ import annotations

import numpy as np
import polars as pl


def bench_columns(df: pl.DataFrame, id_col: str) -> list[str]:
    return [col for col in df.columns if col != id_col]


def align_bench_to_ids(main_ids: np.ndarray, bench_df: pl.DataFrame, id_col: str) -> tuple[np.ndarray, list[str]]:
    if bench_df.height == 0:
        raise RuntimeError("Benchmark dataframe is empty.")
    source_cols = bench_columns(bench_df, id_col)
    if not source_cols:
        raise RuntimeError("No benchmark columns found in benchmark dataframe.")
    main = pl.DataFrame({id_col: main_ids})
    joined = main.join(bench_df, on=id_col, how="left")
    cols = bench_columns(joined, id_col)
    if not cols:
        raise RuntimeError("No benchmark columns found after join.")
    has_nulls = joined.select(pl.any_horizontal(*[pl.col(col).is_null() for col in cols]).any()).item()
    if bool(has_nulls):
        raise RuntimeError("Benchmark join produced nulls; id mismatch or incomplete benchmark parquet.")
    matrix = joined.select(cols).to_numpy()
    return matrix.astype(np.float32, copy=False), cols
