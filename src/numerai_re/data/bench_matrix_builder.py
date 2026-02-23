from __future__ import annotations

import numpy as np
import polars as pl


class BenchmarkAlignmentError(RuntimeError):
    pass


def bench_columns(df: pl.DataFrame, id_col: str) -> list[str]:
    return [col for col in df.columns if col != id_col]


def align_bench_to_ids(main_ids: np.ndarray, bench_df: pl.DataFrame, id_col: str) -> tuple[np.ndarray, list[str]]:
    if bench_df.height == 0:
        raise BenchmarkAlignmentError("Benchmark dataframe is empty.")
    if id_col not in bench_df.columns:
        raise BenchmarkAlignmentError(
            f"Benchmark dataframe missing id column '{id_col}'. Available columns: {bench_df.columns}."
        )

    bench_df = bench_df.with_columns(pl.col(id_col).cast(pl.Utf8, strict=False))
    main = pl.DataFrame({id_col: main_ids}).with_columns(pl.col(id_col).cast(pl.Utf8, strict=False))

    duplicate_id_count = (
        bench_df.group_by(id_col)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_id_count > 0:
        raise BenchmarkAlignmentError(
            f"Benchmark dataframe has duplicate ids in column '{id_col}': duplicate_count={duplicate_id_count}."
        )

    source_cols = bench_columns(bench_df, id_col)
    if not source_cols:
        raise BenchmarkAlignmentError("No benchmark columns found in benchmark dataframe.")
    joined = main.join(bench_df, on=id_col, how="left")
    cols = bench_columns(joined, id_col)
    if not cols:
        raise BenchmarkAlignmentError("No benchmark columns found after join.")
    null_mask = pl.any_horizontal(*[pl.col(col).is_null() for col in cols])
    has_nulls = joined.select(null_mask.any()).item()
    if bool(has_nulls):
        missing_df = joined.filter(null_mask).select(id_col)
        missing_count = missing_df.height
        sample_missing_ids = missing_df.head(10).get_column(id_col).to_list()
        raise BenchmarkAlignmentError(
            "Benchmark join produced nulls; "
            f"missing_count={missing_count} "
            f"missing_id_sample={sample_missing_ids} "
            "reason=id mismatch or incomplete benchmark parquet."
        )
    matrix = joined.select(cols).to_numpy()
    return matrix.astype(np.float32, copy=False), cols
