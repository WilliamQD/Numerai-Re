from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if "polars" not in sys.modules:
    fake_polars = types.ModuleType("polars")
    fake_polars.DataFrame = object
    fake_polars.read_parquet = lambda *_args, **_kwargs: None
    sys.modules["polars"] = fake_polars
if "numerapi" not in sys.modules:
    fake_numerapi = types.ModuleType("numerapi")
    fake_numerapi.NumerAPI = object
    sys.modules["numerapi"] = fake_numerapi

from numerai_re.data.benchmarks import _pick_benchmark_dataset, download_benchmark_parquets


class _FakeNumerAPI:
    def __init__(self, datasets: list[str]) -> None:
        self._datasets = datasets
        self.download_calls: list[tuple[str, str]] = []

    def list_datasets(self) -> list[str]:
        return self._datasets

    def download_dataset(self, dataset_path: str, out_path: str) -> None:
        self.download_calls.append((dataset_path, out_path))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("stub")


class BenchmarksTests(unittest.TestCase):
    def test_pick_benchmark_dataset_prefers_exact_default_name(self) -> None:
        datasets = [
            "v5.2/train_benchmark_models.parquet",
            "v5.2/train_benchmark_models_alt.parquet",
        ]
        chosen = _pick_benchmark_dataset(datasets, "v5.2", "train")
        self.assertEqual(chosen, "v5.2/train_benchmark_models.parquet")

    def test_pick_benchmark_dataset_raises_on_true_ambiguity(self) -> None:
        datasets = [
            "v5.2/train_benchmark_models_alt_a.parquet",
            "v5.2/train_benchmark_models_alt_b.parquet",
        ]
        with self.assertRaises(RuntimeError):
            _pick_benchmark_dataset(datasets, "v5.2", "train")

    def test_force_redownload_downloads_even_when_file_exists(self) -> None:
        datasets = [
            "v5.2/train_benchmark_models.parquet",
            "v5.2/validation_benchmark_models.parquet",
        ]
        api = _FakeNumerAPI(datasets)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing = out_dir / "train_benchmark_models.parquet"
            existing.write_text("old")
            download_benchmark_parquets(api, "v5.2", out_dir, force_redownload=True)
        self.assertEqual(len(api.download_calls), 2)


if __name__ == "__main__":
    unittest.main()
