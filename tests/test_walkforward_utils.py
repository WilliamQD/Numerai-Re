from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era_utils import era_to_int
from walkforward import build_windows


class WalkforwardUtilsTests(unittest.TestCase):
    def test_era_to_int_parses_prefixed_eras(self) -> None:
        parsed = era_to_int(np.array(["era1", "era20", "era003"]))
        np.testing.assert_array_equal(parsed, np.array([1, 20, 3], dtype=np.int32))

    def test_build_windows_matches_expected_ranges(self) -> None:
        eras = np.arange(1, 625, dtype=np.int32)
        windows = build_windows(eras, chunk_size=156, purge_eras=8, min_chunk_index=2)
        self.assertEqual(len(windows), 3)
        self.assertEqual((windows[0].train_start, windows[0].train_end, windows[0].purge_start, windows[0].purge_end), (1, 148, 149, 156))
        self.assertEqual((windows[0].val_start, windows[0].val_end), (157, 312))
        self.assertEqual((windows[1].train_end, windows[1].val_start, windows[1].val_end), (304, 313, 468))
        self.assertEqual((windows[2].train_end, windows[2].val_start, windows[2].val_end), (460, 469, 624))


if __name__ == "__main__":
    unittest.main()
