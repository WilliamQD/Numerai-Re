from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl


def load_split_numpy(
    path: Path,
    feature_cols: list[str],
    id_col: str,
    era_col: str,
    target_col: str,
    downcast: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_cols = [*feature_cols, target_col, era_col, id_col]
    frame = pl.scan_parquet(str(path)).select(selected_cols).collect(streaming=True)
    frame = frame.filter(pl.col(target_col).is_not_null())
    if downcast:
        frame = frame.with_columns(pl.col(feature_cols).cast(pl.Float32, strict=False))
    x = frame.select(feature_cols).to_numpy()
    y = frame.get_column(target_col).to_numpy().astype(np.float32, copy=False)
    era = frame.get_column(era_col).to_numpy()
    row_id = frame.get_column(id_col).to_numpy()
    del frame
    gc.collect()
    return x, y, era, row_id
