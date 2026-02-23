from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PipelineDryRunTests(unittest.TestCase):
    def test_train_and_infer_dry_runs(self) -> None:
        train = subprocess.run(
            [sys.executable, "-m", "numerai_re.cli.train_colab"],
            cwd=ROOT,
            env={**os.environ, "TRAIN_DRY_RUN": "true", "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(train.returncode, 0, msg=train.stderr + train.stdout)
        self.assertIn("TRAIN_DRY_RUN_OK", train.stdout)

        infer = subprocess.run(
            [sys.executable, "-m", "numerai_re.cli.inference"],
            cwd=ROOT,
            env={**os.environ, "INFER_DRY_RUN": "true", "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(infer.returncode, 0, msg=infer.stderr + infer.stdout)
        self.assertIn("INFER_DRY_RUN_OK", infer.stdout)

    def test_validate_pipeline_dry_run(self) -> None:
        validate = subprocess.run(
            [sys.executable, "-m", "tools.validate_pipeline", "--dry-run"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(validate.returncode, 0, msg=validate.stderr + validate.stdout)
        self.assertIn("VALIDATION_SUMMARY", validate.stdout)


if __name__ == "__main__":
    unittest.main()
