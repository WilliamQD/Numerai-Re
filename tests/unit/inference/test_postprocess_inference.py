from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerai_re.runtime.config import InferenceRuntimeConfig
from numerai_re.cli.inference import (
    RANK_01_EPSILON,
    DriftGuardError,
    _download_live_benchmark_dataset,
    _download_live_dataset,
    apply_quality_gates,
)
from numerai_re.inference.postprocess import PostprocessConfig, apply_postprocess
from numerai_re.inference.inference_runtime import _align_live_benchmarks


class _FakeNumerAPI:
    def __init__(self, datasets: list[str]) -> None:
        self._datasets = datasets
        self.downloaded: tuple[str, str] | None = None

    def list_datasets(self) -> list[str]:
        return self._datasets

    def download_dataset(self, dataset: str, out_path: str) -> None:
        self.downloaded = (dataset, out_path)


class PostprocessInferenceTests(unittest.TestCase):
    def test_postprocess_config_loads_defaults(self) -> None:
        payload = {
            "schema_version": 1,
            "blend_alpha": 0.6,
            "bench_neutralize_prop": 0.5,
            "payout_weight_corr": 0.75,
            "payout_weight_bmc": 2.25,
            "bench_cols_used": ["benchmark_1"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "postprocess_config.json"
            cfg_path.write_text(json.dumps(payload))
            cfg = PostprocessConfig.from_json(cfg_path)
        self.assertEqual(cfg.submission_transform, "rank_01")
        self.assertEqual(cfg.feature_neutralize_n_features, 0)
        self.assertEqual(cfg.bench_cols_used, ("benchmark_1",))

    def test_apply_postprocess_rank_01_bounds(self) -> None:
        pred_raw = np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float32)
        era = np.array(["era1", "era1", "era2", "era2"])
        bench = np.array([[0.4], [0.1], [0.2], [0.3]], dtype=np.float32)
        cfg = PostprocessConfig(
            schema_version=1,
            submission_transform="rank_01",
            blend_alpha=0.4,
            bench_neutralize_prop=0.5,
            payout_weight_corr=0.75,
            payout_weight_bmc=2.25,
            bench_cols_used=("benchmark_1",),
        )
        pred_final = apply_postprocess(pred_raw, era, cfg, bench=bench)
        self.assertEqual(pred_final.shape, pred_raw.shape)
        self.assertTrue((pred_final > 0.0).all())
        self.assertTrue((pred_final < 1.0).all())

    def test_quality_gate_rejects_rank_01_out_of_bounds(self) -> None:
        cfg = InferenceRuntimeConfig(
            numerai_public_id="a",
            numerai_secret_key="b",
            numerai_model_name="m",
            wandb_entity="e",
            wandb_project="p",
            max_abs_exposure=1.0,
            exposure_sample_rows=2,
        )
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [2.0, 1.0, 4.0]})
        with self.assertRaises(DriftGuardError):
            apply_quality_gates(features, np.array([0.0, 0.3, 0.7], dtype=np.float32), cfg, "rank_01")
        with self.assertRaises(DriftGuardError):
            apply_quality_gates(features, np.array([0.2, 0.3, 1.0], dtype=np.float32), cfg, "rank_01")
        apply_quality_gates(
            features,
            np.array([RANK_01_EPSILON, 0.3, 1.0 - RANK_01_EPSILON], dtype=np.float32),
            cfg,
            "rank_01",
        )

    def test_download_live_benchmark_dataset_discovers_expected_name(self) -> None:
        napi = _FakeNumerAPI(["v5.2/live.parquet", "v5.2/live_benchmark_models.parquet"])
        out_path = Path("live_benchmark_models.parquet")
        resolved = _download_live_benchmark_dataset(napi, "v5.2", out_path)
        self.assertEqual(resolved, out_path)
        self.assertEqual(napi.downloaded, ("v5.2/live_benchmark_models.parquet", str(out_path)))

    def test_download_live_dataset_prefers_int8_when_available(self) -> None:
        napi = _FakeNumerAPI(["v5.2/live.parquet", "v5.2/live_int8.parquet"])
        out_path = Path("live.parquet")
        resolved = _download_live_dataset(napi, "v5.2", out_path, use_int8_parquet=True)
        self.assertEqual(resolved, out_path)
        self.assertEqual(napi.downloaded, ("v5.2/live_int8.parquet", str(out_path)))

    def test_align_live_benchmarks_accepts_id_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "live_benchmark_models.parquet"
            pd.DataFrame(
                {
                    "id": ["id_1", "id_2"],
                    "benchmark_1": [0.1, 0.2],
                    "benchmark_2": [0.3, 0.4],
                }
            ).to_parquet(path, index=False)

            aligned = _align_live_benchmarks(path, np.array(["id_2", "id_1"]), ["benchmark_1", "benchmark_2"])

        self.assertEqual(aligned.shape, (2, 2))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(aligned[1, 1]), 0.3, places=6)

    def test_align_live_benchmarks_accepts_id_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "live_benchmark_models.parquet"
            df = pd.DataFrame(
                {
                    "benchmark_1": [0.1, 0.2],
                    "benchmark_2": [0.3, 0.4],
                },
                index=pd.Index(["id_1", "id_2"], name="id"),
            )
            df.to_parquet(path)

            aligned = _align_live_benchmarks(path, np.array(["id_2", "id_1"]), ["benchmark_1", "benchmark_2"])

        self.assertEqual(aligned.shape, (2, 2))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(aligned[1, 1]), 0.3, places=6)

    def test_align_live_benchmarks_drops_non_numeric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "live_benchmark_models.parquet"
            pd.DataFrame(
                {
                    "id": ["id_1", "id_2"],
                    "era": ["X", "Y"],
                    "benchmark_1": [0.1, 0.2],
                }
            ).to_parquet(path, index=False)

            aligned = _align_live_benchmarks(path, np.array(["id_2", "id_1"]), ["era", "benchmark_1"])

        self.assertEqual(aligned.shape, (2, 1))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(aligned[1, 0]), 0.1, places=6)

    def test_align_live_benchmarks_raises_when_all_non_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "live_benchmark_models.parquet"
            pd.DataFrame(
                {
                    "id": ["id_1", "id_2"],
                    "era": ["X", "Y"],
                }
            ).to_parquet(path, index=False)

            with self.assertRaises(DriftGuardError):
                _align_live_benchmarks(path, np.array(["id_2", "id_1"]), ["era"])


if __name__ == "__main__":
    unittest.main()
