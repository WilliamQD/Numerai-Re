from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artifact_contract import (
    MANIFEST_FILENAME,
    load_features_by_model,
    load_manifest,
    load_union_features,
    resolve_model_files,
    validate_model_files_exist,
)


class ArtifactContractTests(unittest.TestCase):
    def test_load_manifest_and_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "dataset_version": "v5.2",
                        "feature_set": "medium",
                        "artifact_schema_version": 4,
                        "model_files": ["m1.txt"],
                        "features_union_file": "features_union.json",
                        "features_by_model_file": "features_by_model.json",
                    }
                )
            )
            manifest = load_manifest(root, label="test")
            model_files = resolve_model_files(manifest, label="test")
        self.assertEqual(model_files, ["m1.txt"])

    def test_load_features_mapping_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "features_by_model_file": "missing.json",
            }
            model_files = ["m1.txt"]
            with self.assertRaises(RuntimeError):
                load_features_by_model(
                    root,
                    manifest,
                    model_files,
                    label="test",
                )

    def test_load_union_features_requires_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"features_union_file": "features_union.json"}
            (root / "features_union.json").write_text(json.dumps(["feature_1"]))
            features = load_union_features(root, manifest, label="test")
        self.assertEqual(features, ["feature_1"])

    def test_validate_model_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m1.txt").write_text("stub")
            validate_model_files_exist(root, ["m1.txt"], label="test")


if __name__ == "__main__":
    unittest.main()
