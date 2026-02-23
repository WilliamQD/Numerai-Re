from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Maximum allowed difference between parquet mtime and cached mtime (seconds).
_CACHE_MTIME_TOLERANCE_S: float = 1.0


def _feature_cols_hash(feature_cols: list[str]) -> str:
    """Return a short deterministic hash of a feature-column list."""
    return hashlib.sha256("|".join(feature_cols).encode()).hexdigest()[:16]


def _polars_dtype_for(feature_dtype: type) -> pl.DataType:
    """Return the Polars dtype to use when casting feature columns."""
    if np.dtype(feature_dtype) == np.dtype(np.int8):
        return pl.Int8
    return pl.Float32


def _cache_manifest_path(path: Path, feature_cols: list[str], feature_dtype: type) -> Path:
    fhash = _feature_cols_hash(feature_cols)
    dtype_tag = np.dtype(feature_dtype).name  # e.g. 'float32' or 'int8'
    return path.parent / "_np_cache" / f"{path.stem}_{fhash}_{dtype_tag}.json"


def _cache_valid(parquet_path: Path, manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        return abs(manifest.get("parquet_mtime", 0) - parquet_path.stat().st_mtime) < _CACHE_MTIME_TOLERANCE_S
    except Exception:
        return False


def _cache_array_path(manifest_path: Path, tag: str) -> Path:
    return manifest_path.parent / f"{manifest_path.stem}_{tag}.npy"


def _load_from_cache(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.load(str(_cache_array_path(manifest_path, "x")), allow_pickle=False)
    y = np.load(str(_cache_array_path(manifest_path, "y")), allow_pickle=False)
    era = np.load(str(_cache_array_path(manifest_path, "era")), allow_pickle=True)
    row_id = np.load(str(_cache_array_path(manifest_path, "id")), allow_pickle=True)
    return x, y, era, row_id


def _save_to_cache(
    manifest_path: Path,
    parquet_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    era: np.ndarray,
    row_id: np.ndarray,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(_cache_array_path(manifest_path, "x")), x)
    np.save(str(_cache_array_path(manifest_path, "y")), y)
    np.save(str(_cache_array_path(manifest_path, "era")), era)
    np.save(str(_cache_array_path(manifest_path, "id")), row_id)
    manifest_path.write_text(json.dumps({"parquet_mtime": parquet_path.stat().st_mtime}))


def load_split_numpy(
    path: Path,
    feature_cols: list[str],
    id_col: str,
    era_col: str,
    target_col: str,
    downcast: bool = True,
    feature_dtype: type = np.float32,
    use_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a parquet split into numpy arrays with column projection.

    Parameters
    ----------
    feature_dtype:
        Numpy dtype to cast feature columns to.  Use ``np.int8`` when loading
        Numerai int8 parquet files to keep memory footprint 4× smaller than
        the default ``np.float32``.
    use_cache:
        When *True*, save/reload the processed numpy arrays next to the parquet
        file under a ``_np_cache/`` sub-directory.  Speeds up session restarts
        at the cost of extra disk space.
    """
    if use_cache:
        manifest_path = _cache_manifest_path(path, feature_cols, feature_dtype)
        if _cache_valid(path, manifest_path):
            return _load_from_cache(manifest_path)

    selected_cols = [*feature_cols, target_col, era_col, id_col]
    frame = pl.scan_parquet(str(path)).select(selected_cols).collect(streaming=True)
    frame = frame.filter(pl.col(target_col).is_not_null())
    if downcast:
        frame = frame.with_columns(pl.col(feature_cols).cast(_polars_dtype_for(feature_dtype), strict=False))
    x = frame.select(feature_cols).to_numpy()
    y = frame.get_column(target_col).to_numpy().astype(np.float32, copy=False)
    era = frame.get_column(era_col).to_numpy()
    row_id = frame.get_column(id_col).to_numpy()
    del frame
    gc.collect()

    if use_cache:
        try:
            _save_to_cache(manifest_path, path, x, y, era, row_id)
        except Exception:
            logger.debug("phase=cache_write_failed path=%s", manifest_path, exc_info=True)

    return x, y, era, row_id
