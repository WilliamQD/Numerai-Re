from __future__ import annotations

import re

import numpy as np


_ERA_RE = re.compile(r"(\d+)")


def era_to_int(arr) -> np.ndarray:
    """Convert era labels to integer era numbers."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int32, copy=False)

    out = np.empty(len(a), dtype=np.int32)
    for i, value in enumerate(a):
        match = _ERA_RE.search(str(value))
        if not match:
            raise ValueError(f"Could not parse era number from: {value!r}")
        out[i] = int(match.group(1))
    return out
