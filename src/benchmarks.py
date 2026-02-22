from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from numerapi import NumerAPI


logger = logging.getLogger(__name__)


def download_benchmark_parquets(napi: NumerAPI, dataset_version: str, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = napi.list_datasets()

    def pick(name_contains: tuple[str, ...]) -> str:
        prefix = dataset_version.lower() + "/"
        for dataset in datasets:
            lowered = dataset.lower()
            if lowered.startswith(prefix) and all(token in lowered for token in name_contains):
                return dataset
        raise RuntimeError(f"Could not find dataset containing {name_contains} under {dataset_version}/")

    mapping = {
        "train": pick(("train", "benchmark", "models")),
        "validation": pick(("validation", "benchmark", "models")),
    }
    try:
        mapping["live"] = pick(("live", "benchmark", "models"))
    except RuntimeError:
        pass

    out: dict[str, Path] = {}
    for split, dataset_path in mapping.items():
        local_path = out_dir / Path(dataset_path).name
        if local_path.exists():
            logger.info("phase=bench_reused split=%s path=%s", split, local_path)
        else:
            logger.info("phase=bench_downloading split=%s dataset=%s path=%s", split, dataset_path, local_path)
            napi.download_dataset(dataset_path, str(local_path))
        out[split] = local_path
    return out


def load_benchmark_frame(path: Path) -> pl.DataFrame:
    return pl.read_parquet(str(path))
