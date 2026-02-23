from __future__ import annotations

from pathlib import Path


def resolve_split_parquet(
    datasets: list[str],
    dataset_version: str,
    split_tokens: tuple[str, ...],
    *,
    use_int8: bool,
) -> str:
    prefix = dataset_version.lower() + "/"
    token_set = tuple(token.lower() for token in split_tokens)
    parquet_matches = [
        ds
        for ds in datasets
        if ds.lower().startswith(prefix)
        and ds.lower().endswith(".parquet")
        and all(token in ds.lower() for token in token_set)
    ]
    if not use_int8:
        default_match = next((ds for ds in parquet_matches if "int8" not in ds.lower()), None)
        return default_match or f"{dataset_version}/{'_'.join(split_tokens)}.parquet"

    int8_match = next((ds for ds in parquet_matches if "int8" in ds.lower()), None)
    if int8_match:
        return int8_match

    default_match = next((ds for ds in parquet_matches if "int8" not in ds.lower()), None)
    return default_match or f"{dataset_version}/{'_'.join(split_tokens)}.parquet"


def pick_benchmark_models_parquet(datasets: list[str], dataset_version: str, split: str) -> str:
    prefix = dataset_version.lower() + "/"
    split_token = split.lower()
    candidates = sorted(
        ds
        for ds in datasets
        if ds.lower().startswith(prefix)
        and ds.lower().endswith(".parquet")
        and split_token in ds.lower()
        and "benchmark" in ds.lower()
        and "models" in ds.lower()
    )
    if not candidates:
        raise RuntimeError(f"Could not find {split} benchmark dataset under {dataset_version}/")
    if len(candidates) == 1:
        return candidates[0]

    expected_name = f"{split}_benchmark_models.parquet"
    exact = [ds for ds in candidates if Path(ds).name.lower() == expected_name]
    if len(exact) == 1:
        return exact[0]

    raise RuntimeError(
        f"Ambiguous benchmark dataset for split={split} dataset_version={dataset_version}. "
        f"Candidates={candidates}."
    )
