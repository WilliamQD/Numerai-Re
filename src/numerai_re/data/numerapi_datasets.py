from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitParquetResolution:
    selected: str
    candidates: tuple[str, ...]
    int8_candidates: tuple[str, ...]
    non_int8_candidates: tuple[str, ...]
    use_int8_requested: bool
    dataset_version: str
    split_tokens: tuple[str, ...]

    @property
    def selected_is_int8(self) -> bool:
        return "int8" in self.selected.lower()


def resolve_split_parquet_with_report(
    datasets: list[str],
    dataset_version: str,
    split_tokens: tuple[str, ...],
    *,
    use_int8: bool,
) -> SplitParquetResolution:
    prefix = dataset_version.lower() + "/"
    token_set = tuple(token.lower() for token in split_tokens)
    parquet_matches = tuple(
        ds
        for ds in datasets
        if ds.lower().startswith(prefix)
        and ds.lower().endswith(".parquet")
        and all(token in ds.lower() for token in token_set)
    )
    int8_matches = tuple(ds for ds in parquet_matches if "int8" in ds.lower())
    non_int8_matches = tuple(ds for ds in parquet_matches if "int8" not in ds.lower())

    if not use_int8:
        selected = non_int8_matches[0] if non_int8_matches else f"{dataset_version}/{'_'.join(split_tokens)}.parquet"
    elif int8_matches:
        selected = int8_matches[0]
    elif non_int8_matches:
        selected = non_int8_matches[0]
    else:
        selected = f"{dataset_version}/{'_'.join(split_tokens)}.parquet"

    return SplitParquetResolution(
        selected=selected,
        candidates=parquet_matches,
        int8_candidates=int8_matches,
        non_int8_candidates=non_int8_matches,
        use_int8_requested=bool(use_int8),
        dataset_version=dataset_version,
        split_tokens=token_set,
    )


def resolve_split_parquet(
    datasets: list[str],
    dataset_version: str,
    split_tokens: tuple[str, ...],
    *,
    use_int8: bool,
) -> str:
    return resolve_split_parquet_with_report(
        datasets,
        dataset_version,
        split_tokens,
        use_int8=use_int8,
    ).selected


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
