from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from numerapi import NumerAPI

from numerapi_datasets import pick_benchmark_models_parquet


logger = logging.getLogger(__name__)


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
