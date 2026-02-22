from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WFWindow:
    window_id: int
    train_start: int
    train_end: int
    purge_start: int
    purge_end: int
    val_start: int
    val_end: int


def build_windows(
    era_numbers: np.ndarray,
    chunk_size: int = 156,
    purge_eras: int = 8,
    min_chunk_index: int = 2,
) -> list[WFWindow]:
    eras = np.unique(era_numbers.astype(np.int32, copy=False))
    eras.sort()
    if eras.size == 0:
        return []

    max_era = int(eras[-1])
    max_chunk_index = max_era // chunk_size
    windows: list[WFWindow] = []
    window_id = 0

    for chunk_idx in range(min_chunk_index, max_chunk_index + 1):
        val_start = (chunk_idx - 1) * chunk_size + 1
        val_end = chunk_idx * chunk_size
        train_end = val_start - purge_eras - 1
        purge_start = train_end + 1
        purge_end = val_start - 1

        if train_end < int(eras[0]):
            continue

        has_val = np.any((eras >= val_start) & (eras <= val_end))
        has_train = np.any((eras >= int(eras[0])) & (eras <= train_end))
        if not (has_train and has_val):
            continue

        window_id += 1
        windows.append(
            WFWindow(
                window_id=window_id,
                train_start=int(eras[0]),
                train_end=int(train_end),
                purge_start=int(purge_start),
                purge_end=int(purge_end),
                val_start=int(val_start),
                val_end=int(val_end),
            )
        )

    return windows
