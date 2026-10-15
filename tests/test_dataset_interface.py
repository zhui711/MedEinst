"""Focused checks for public rows and fixed support identity."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.medeinst_adapter import load_pairs, resolve_support_manifest


ROOT = Path(__file__).resolve().parents[1]


class DatasetInterfaceTests(unittest.TestCase):
    def test_public_rows_and_paired_files_preserve_clinical_inputs(self):
        rows = [
            {"case_id": "case_1", "case_type": "control", "narrative": "Original control text.", "ground_truth": "Pneumonia"},
            {"case_id": "case_1", "case_type": "trap", "narrative": "Original trap text.", "ground_truth": "Bronchitis"},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)
            public = path / "train.jsonl"
            public.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pair = load_pairs(public)[0]
            self.assertEqual((pair.name, pair.control.narrative, pair.trap.narrative),
                             ("case_1", "Original control text.", "Original trap text."))
            self.assertEqual((pair.control.ground_truth_id, pair.trap.ground_truth_id), ("45", "23"))

            legacy = path / "paired" / "case_1" / "final_benchmark_pair.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(json.dumps({
                "control_case": {"narrative": rows[0]["narrative"], "ground_truth": rows[0]["ground_truth"]},
                "trap_case": {"narrative": rows[1]["narrative"], "ground_truth": rows[1]["ground_truth"]},
            }), encoding="utf-8")
            self.assertEqual(load_pairs(path / "paired")[0], pair)

            manifest = path / "support.json"
            manifest.write_text(json.dumps([{"case_id": "case_1", "type": "trap", "ground_truth_id": "23",
                                             "ground_truth": "Bronchitis"}]), encoding="utf-8")
            resolved = resolve_support_manifest(manifest, [pair])
            self.assertEqual(pair.get(resolved[0]["type"]).narrative, rows[1]["narrative"])
            manifest.write_text(manifest.read_text().replace("case_1", "case_missing"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case_missing"):
                resolve_support_manifest(manifest, [pair])

            public.write_text(public.read_text() + json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate dataset row"):
                load_pairs(public)

if __name__ == "__main__":
    unittest.main()
