from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from tools.performance_tracker import TrackerConfig, _score_evidence


class PerformanceTrackerAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TrackerConfig(
            output_dir=Path("."),
            lookback_rounds=20,
            min_resolved_rounds=2,
            recommend_threshold=65.0,
            primary_weight_corr=0.75,
            primary_weight_secondary=2.25,
        )

    def test_not_enough_evidence_when_no_resolved_scores(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "round_number": 1220,
                    "resolved": False,
                    "primary_score": None,
                }
            ]
        )
        result = _score_evidence(history, self.cfg)
        self.assertEqual(result["recommendation"], "NOT_ENOUGH_EVIDENCE")
        self.assertEqual(result["resolved_rounds_count"], 0)

    def test_sufficient_evidence_for_strong_recent_scores(self) -> None:
        history = pd.DataFrame(
            [
                {"round_number": 1225, "resolved": True, "primary_score": 0.030},
                {"round_number": 1224, "resolved": True, "primary_score": 0.022},
                {"round_number": 1223, "resolved": True, "primary_score": 0.018},
            ]
        )
        result = _score_evidence(history, self.cfg)
        self.assertEqual(result["recommendation"], "EVIDENCE_SUFFICIENT_FOR_MANUAL_STAKING")
        self.assertGreaterEqual(result["evidence_score"], 65.0)

    def test_not_enough_evidence_for_consistent_negative_scores(self) -> None:
        history = pd.DataFrame(
            [
                {"round_number": 1225, "resolved": True, "primary_score": -0.030},
                {"round_number": 1224, "resolved": True, "primary_score": -0.010},
                {"round_number": 1223, "resolved": True, "primary_score": -0.015},
            ]
        )
        result = _score_evidence(history, self.cfg)
        self.assertEqual(result["recommendation"], "NOT_ENOUGH_EVIDENCE")
        self.assertLess(result["evidence_score"], 65.0)

    def test_resolved_string_values_are_handled(self) -> None:
        history = pd.DataFrame(
            [
                {"round_number": 1225, "resolved": "True", "primary_score": 0.020},
                {"round_number": 1224, "resolved": "true", "primary_score": 0.015},
                {"round_number": 1223, "resolved": "False", "primary_score": 0.050},
            ]
        )
        result = _score_evidence(history, self.cfg)
        self.assertEqual(result["resolved_rounds_count"], 2)
        self.assertEqual(result["recommendation"], "EVIDENCE_SUFFICIENT_FOR_MANUAL_STAKING")


if __name__ == "__main__":
    unittest.main()
