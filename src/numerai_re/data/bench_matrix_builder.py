from __future__ import annotations

import numpy as np
import polars as pl


class BenchmarkAlignmentError(RuntimeError):
    pass


def bench_columns(df: pl.DataFrame, id_col: str) -> list[str]:
    return [col for col in df.columns if col != id_col]


def align_bench_to_ids(
    main_ids: np.ndarray,
    bench_df: pl.DataFrame,
    id_col: str,
    *,
    drop_sparse_columns: bool = True,
    max_null_ratio_per_column: float = 0.0,
    min_benchmark_columns: int = 1,
) -> tuple[np.ndarray, list[str]]:
    if bench_df.height == 0:
        raise BenchmarkAlignmentError("Benchmark dataframe is empty.")
    if id_col not in bench_df.columns:
        raise BenchmarkAlignmentError(
            f"Benchmark dataframe missing id column '{id_col}'. Available columns: {bench_df.columns}."
        )
    if not 0.0 <= max_null_ratio_per_column <= 1.0:
        raise BenchmarkAlignmentError(
            "Invalid max_null_ratio_per_column; expected value within [0.0, 1.0], "
            f"got {max_null_ratio_per_column}."
        )
    if min_benchmark_columns <= 0:
        raise BenchmarkAlignmentError(
            f"Invalid min_benchmark_columns; expected positive integer, got {min_benchmark_columns}."
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

    bench_df = bench_df.with_columns(pl.lit(1).cast(pl.Int8).alias("__bench_row_present"))
    joined = main.join(bench_df, on=id_col, how="left")
    join_missing_mask = pl.col("__bench_row_present").is_null()
    has_join_misses = joined.select(join_missing_mask.any()).item()
    if bool(has_join_misses):
        missing_df = joined.filter(join_missing_mask).select(id_col)
        missing_count = missing_df.height
        sample_missing_ids = missing_df.head(10).get_column(id_col).to_list()
        raise BenchmarkAlignmentError(
            "Benchmark id join failed; "
            f"missing_count={missing_count} "
            f"missing_id_sample={sample_missing_ids} "
            "reason=benchmark ids do not fully cover requested ids."
        )

    joined = joined.drop("__bench_row_present")
    cols = bench_columns(joined, id_col)
    if not cols:
        raise BenchmarkAlignmentError("No benchmark columns found after join.")

    null_ratio_by_col: dict[str, float] = {}
    for col in cols:
        null_ratio_by_col[col] = float(joined.select(pl.col(col).is_null().mean()).item())

    dropped_cols: list[str] = []
    if drop_sparse_columns:
        dropped_cols = [col for col in cols if null_ratio_by_col[col] > max_null_ratio_per_column]
        cols = [col for col in cols if col not in dropped_cols]

    if len(cols) < min_benchmark_columns:
        raise BenchmarkAlignmentError(
            "Benchmark sparse-column filtering removed too many columns; "
            f"kept_columns={len(cols)} min_benchmark_columns={min_benchmark_columns} "
            f"dropped_columns={len(dropped_cols)} total_columns={len(source_cols)}."
        )

    remaining_null_mask = pl.any_horizontal(*[pl.col(col).is_null() for col in cols])
    has_remaining_nulls = joined.select(remaining_null_mask.any()).item()
    if bool(has_remaining_nulls):
        sparse_df = joined.filter(remaining_null_mask).select(id_col)
        sparse_count = sparse_df.height
        sample_sparse_ids = sparse_df.head(10).get_column(id_col).to_list()
        raise BenchmarkAlignmentError(
            "Benchmark columns contain nulls after sparse-column filtering; "
            f"sparse_count={sparse_count} "
            f"sparse_id_sample={sample_sparse_ids} "
            f"max_null_ratio_per_column={max_null_ratio_per_column}."
        )

    matrix = joined.select(cols).to_numpy()
    return matrix.astype(np.float32, copy=False), cols
