#!/usr/bin/env python3
"""Optionally resample individual Reference Set records with the historical CGME rule.

The fixed paper-support selection is data/medeinst_support_853.json. This command
writes a separate candidate manifest and compares it with that fixed selection.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.medeinst_adapter import load_pairs
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP

OFFICIAL_MANIFEST = PROJECT_ROOT / "data" / "medeinst_support_853.json"


def sample_support(pairs, cap: int = 20, seed: int = 42) -> list[dict]:
    """Match the historical record-level capped shuffle and final ordering."""
    if cap < 1:
        raise ValueError("--cap must be positive")
    distribution = defaultdict(list)
    for pair in sorted(pairs, key=lambda item: item.case_id):
        for record in (pair.control, pair.trap):
            distribution[record.ground_truth_id].append({
                "case_id": record.case_id,
                "type": record.case_type.removesuffix("_case"),
                "ground_truth_id": record.ground_truth_id,
                "ground_truth": DIAGNOSIS_ID_MAP[record.ground_truth_id],
            })
    rng = random.Random(seed)
    selected = []
    # Python's stable sort preserves first-seen disease order for count ties.
    for _, records in sorted(distribution.items(), key=lambda item: len(item[1])):
        shuffled = records.copy()
        rng.shuffle(shuffled)
        selected.extend(shuffled[:cap])
    return sorted(selected, key=lambda row: (row["ground_truth_id"], row["case_id"]))

def main() -> None:
    parser = argparse.ArgumentParser(description="Resample a candidate CGME support manifest from the MedEinst Reference Set")
    parser.add_argument("--reference-set", required=True, help="Public train.jsonl or a paired-case directory")
    parser.add_argument("--output", required=True, help="Candidate manifest path; cannot replace the fixed official manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=20, help="Maximum individual case records per pathology")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output == OFFICIAL_MANIFEST.resolve():
        parser.error("The fixed official manifest cannot be overwritten by resampling")
    sampled = sample_support(load_pairs(args.reference_set), cap=args.cap, seed=args.seed)
    official = json.loads(OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sampled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    expected = {(r["case_id"], r["type"], r["ground_truth_id"]) for r in official}
    actual = {(r["case_id"], r["type"], r["ground_truth_id"]) for r in sampled}
    print(f"Wrote {len(sampled)} candidate records to {output}")
    print(f"Fixed official selection: {len(official)} records; identical order/content: {sampled == official}")
    if actual != expected:
        print("Only in fixed selection:", sorted(expected - actual))
        print("Only in candidate selection:", sorted(actual - expected))

if __name__ == "__main__":
    main()
