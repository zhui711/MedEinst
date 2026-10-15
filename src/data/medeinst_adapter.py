"""Read public MedEinst JSONL rows or the historical paired-JSON layout.

Only dataset input is adapted here. The agent receives the original narrative,
case ID, case variant, and diagnosis label without rewriting clinical text.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.utils.prompt_utils import DIAGNOSIS_ID_MAP, normalize_diagnosis_id


_NAME_TO_ID = {name.casefold(): disease_id for disease_id, name in DIAGNOSIS_ID_MAP.items()}
_SYNONYMS = json.loads(
    (Path(__file__).resolve().parents[2] / "config" / "disease_synonyms.json").read_text(encoding="utf-8")
)
for _disease_id, _info in _SYNONYMS.items():
    for _name in _info.get("synonyms", []):
        _NAME_TO_ID.setdefault(_name.casefold(), _disease_id)

def diagnosis_id(value: object) -> Optional[str]:
    """Map a diagnosis ID or published disease name to the original 01–49 IDs."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.casefold() in _NAME_TO_ID:
        return _NAME_TO_ID[text.casefold()]
    normalized = normalize_diagnosis_id(text)
    return normalized if normalized in DIAGNOSIS_ID_MAP else None

def case_type(value: str) -> str:
    """Return the original agent/output variant name."""
    if value in ("control", "control_case"):
        return "control_case"
    if value in ("trap", "trap_case"):
        return "trap_case"
    raise ValueError(f"Unknown case_type: {value!r}")

@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    case_type: str
    narrative: str
    ground_truth: str
    ground_truth_id: str

@dataclass(frozen=True)
class CasePair:
    case_id: str
    control: CaseRecord
    trap: CaseRecord

    @property
    def name(self) -> str:
        """Keep the existing runner's case-directory identity interface."""
        return self.case_id

    def get(self, variant: str) -> CaseRecord:
        return self.control if case_type(variant) == "control_case" else self.trap

def _record(row: dict, context: str) -> CaseRecord:
    if not isinstance(row, dict):
        raise ValueError(f"{context}: expected an object")
    case_id = row.get("case_id")
    narrative = row.get("narrative")
    label = row.get("ground_truth")
    if not isinstance(case_id, str) or not case_id or not isinstance(narrative, str) or not narrative:
        raise ValueError(f"{context}: missing case_id or narrative")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{context}: missing ground_truth")
    variant = case_type(row.get("case_type"))
    disease_id = diagnosis_id(label)
    if disease_id is None:
        raise ValueError(f"{context}: unmapped ground_truth {label!r} for {case_id}/{variant}")
    return CaseRecord(case_id, variant, narrative, label, disease_id)

def _public_rows(path: Path) -> Iterable[Tuple[dict, str]]:
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected a public .jsonl file or paired-case directory: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield json.loads(line), f"{path}:{line_number}"

def _paired_rows(path: Path) -> Iterable[Tuple[dict, str]]:
    for case_dir in sorted(path.glob("case_*")):
        if not case_dir.is_dir():
            continue
        pair_file = case_dir / "final_benchmark_pair.json"
        if not pair_file.is_file():
            raise ValueError(f"Missing paired file: {pair_file}")
        pair_data = json.loads(pair_file.read_text(encoding="utf-8"))
        for variant in ("control_case", "trap_case"):
            if variant not in pair_data:
                raise ValueError(f"Missing {variant} in {pair_file}")
            yield {
                "case_id": case_dir.name,
                "case_type": variant,
                "narrative": pair_data[variant].get("narrative"),
                "ground_truth": pair_data[variant].get("ground_truth"),
            }, f"{pair_file}:{variant}"

def load_pairs(source: str | Path) -> List[CasePair]:
    """Load complete control/trap pairs; reject duplicate or incomplete IDs."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MedEinst input not found: {path}")
    rows = _paired_rows(path) if path.is_dir() else _public_rows(path)
    by_id: Dict[str, Dict[str, CaseRecord]] = {}
    for row, context in rows:
        record = _record(row, context)
        variants = by_id.setdefault(record.case_id, {})
        if record.case_type in variants:
            raise ValueError(f"Duplicate dataset row: {record.case_id}/{record.case_type} at {context}")
        variants[record.case_type] = record
    if not by_id:
        raise ValueError(f"No MedEinst pairs found in {path}")
    incomplete = [case_id for case_id, variants in by_id.items() if set(variants) != {"control_case", "trap_case"}]
    if incomplete:
        raise ValueError(f"Incomplete counterfactual pairs: {', '.join(sorted(incomplete))}")
    return [CasePair(case_id, variants["control_case"], variants["trap_case"])
            for case_id, variants in sorted(by_id.items())]

def resolve_support_manifest(manifest_path: str | Path, pairs: List[CasePair]) -> List[dict]:
    """Resolve the fixed record-level selection without resampling or reordering."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Support manifest must be a JSON list")
    by_id = {pair.case_id: pair for pair in pairs}
    resolved, errors, seen = [], [], set()
    for entry in manifest:
        case_id = entry.get("case_id")
        variant = case_type(entry.get("type"))
        wanted_id = diagnosis_id(entry.get("ground_truth_id"))
        key = (case_id, variant, wanted_id)
        if key in seen:
            errors.append(f"duplicate manifest key {key}")
            continue
        seen.add(key)
        pair = by_id.get(case_id)
        if pair is None:
            errors.append(f"missing {key}")
            continue
        record = pair.get(variant)
        if wanted_id is None or record.ground_truth_id != wanted_id:
            errors.append(f"label mismatch {key}: reference_set={record.ground_truth_id} ({record.ground_truth})")
            continue
        if entry.get("ground_truth") != DIAGNOSIS_ID_MAP[wanted_id]:
            errors.append(f"manifest name mismatch {key}: {entry.get('ground_truth')!r}")
            continue
        resolved.append({
            "case_id": case_id,
            "type": variant.removesuffix("_case"),
            "ground_truth": entry["ground_truth"],
            "ground_truth_id": wanted_id,
            "path": case_id,  # existing trainer's narrative callback uses this as an in-memory key
        })
    if errors:
        raise ValueError("Support manifest does not resolve exactly:\n" + "\n".join(errors))
    return resolved
