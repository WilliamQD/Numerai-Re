"""Benchmark data: download, alignment, and loading for NumerAI benchmark model predictions."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numerapi import NumerAPI

from numerai_re.data.numerapi_datasets import pick_benchmark_models_parquet


logger = logging.getLogger(__name__)


# --- Alignment ---


class BenchmarkAlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkAlignmentResult:
    matrix: np.ndarray
    cols: list[str]
    coverage_mask: np.ndarray


def bench_columns(df: pl.DataFrame, id_col: str) -> list[str]:
    return [col for col in df.columns if col != id_col]


def align_bench_to_ids(
    main_ids: np.ndarray,
    bench_df: pl.DataFrame,
    id_col: str,
    *,
    allow_partial_coverage: bool = False,
    drop_sparse_columns: bool = True,
    max_null_ratio_per_column: float = 0.0,
    min_benchmark_columns: int = 1,
) -> BenchmarkAlignmentResult:
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
    if bool(has_join_misses) and not allow_partial_coverage:
        missing_df = joined.filter(join_missing_mask).select(id_col)
        missing_count = missing_df.height
        sample_missing_ids = missing_df.head(10).get_column(id_col).to_list()
        raise BenchmarkAlignmentError(
            "Benchmark id join failed; "
            f"missing_count={missing_count} "
            f"missing_id_sample={sample_missing_ids} "
            "reason=benchmark ids do not fully cover requested ids."
        )

    coverage_mask = (~joined.get_column("__bench_row_present").is_null()).to_numpy()
    covered_rows = int(np.count_nonzero(coverage_mask))
    if covered_rows == 0:
        raise BenchmarkAlignmentError("Benchmark alignment has zero covered rows after id join.")

    joined_covered = joined.filter(~join_missing_mask)
    joined = joined.drop("__bench_row_present")
    joined_covered = joined_covered.drop("__bench_row_present")
    cols = bench_columns(joined, id_col)
    if not cols:
        raise BenchmarkAlignmentError("No benchmark columns found after join.")

    null_ratio_by_col: dict[str, float] = {}
    for col in cols:
        null_ratio_by_col[col] = float(joined_covered.select(pl.col(col).is_null().mean()).item())

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
    has_remaining_nulls = joined_covered.select(remaining_null_mask.any()).item()
    if bool(has_remaining_nulls):
        sparse_df = joined_covered.filter(remaining_null_mask).select(id_col)
        sparse_count = sparse_df.height
        sample_sparse_ids = sparse_df.head(10).get_column(id_col).to_list()
        raise BenchmarkAlignmentError(
            "Benchmark columns contain nulls after sparse-column filtering; "
            f"sparse_count={sparse_count} "
            f"sparse_id_sample={sample_sparse_ids} "
            f"max_null_ratio_per_column={max_null_ratio_per_column}."
        )

    covered_matrix = joined_covered.select(cols).to_numpy().astype(np.float32, copy=False)
    matrix = np.zeros((len(main_ids), len(cols)), dtype=np.float32)
    matrix[coverage_mask] = covered_matrix
    return BenchmarkAlignmentResult(matrix=matrix, cols=cols, coverage_mask=coverage_mask)


# --- Download ---


def _pick_benchmark_dataset(datasets: list[str], dataset_version: str, split: str) -> str:
    return pick_benchmark_models_parquet(datasets, dataset_version, split)


def download_benchmark_parquets(
    napi: NumerAPI,
    dataset_version: str,
    out_dir: Path,
    *,
    force_redownload: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = napi.list_datasets()

    mapping = {
        "train": _pick_benchmark_dataset(datasets, dataset_version, "train"),
        "validation": _pick_benchmark_dataset(datasets, dataset_version, "validation"),
    }
    try:
        mapping["live"] = _pick_benchmark_dataset(datasets, dataset_version, "live")
    except RuntimeError:
        pass

    out: dict[str, Path] = {}
    for split, dataset_path in mapping.items():
        local_path = out_dir / Path(dataset_path).name
        if local_path.exists() and not force_redownload:
            logger.info("phase=bench_reused split=%s path=%s", split, local_path)
        else:
            if force_redownload and local_path.exists():
                logger.info("phase=bench_redownload_forced split=%s path=%s", split, local_path)
                local_path.unlink(missing_ok=True)
            logger.info("phase=bench_downloading split=%s dataset=%s path=%s", split, dataset_path, local_path)
            napi.download_dataset(dataset_path, str(local_path))
        out[split] = local_path
    return out


def load_benchmark_frame(path: Path) -> pl.DataFrame:
    return pl.read_parquet(str(path))


# --- Load and align ---


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
