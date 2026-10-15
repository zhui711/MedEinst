"""Shared state fields for the diagnostic pipeline."""
from typing import TypedDict, Optional, Dict, Any, Literal, List


class AgentState(TypedDict):
    """Fields exchanged across diagnostic phases."""

    case_id: str
    input_case: Dict[str, Any]

    phase1_result: Optional[Dict[str, Any]]

    # {
    #   "structured_analysis": {...},
    #   "differential_reasoning": "...",
    #   "top_candidates": ["25", "45", ...],
    #   "final_diagnosis_id": "25",
    #   "calculated_confidence": 0.0,
    #   "error": null
    # }

    graph_json: Optional[Dict[str, Any]]

    # {
    #   "case_metadata": {...},
    #   "phase1_context": {...},
    #   "graph": {
    #     "nodes": {"p_nodes": [], "k_nodes": [], "d_nodes": []},
    #     "edges": {"p_k_links": [], "k_d_links": []}
    #   }
    # }

    graph_summary: Optional[str]

    naive_scores: Optional[Dict[str, float]]

    final_output: Optional[Dict[str, Any]]

    # {
    #   "final_diagnosis_id": "25",
    #   "final_diagnosis_name": "Acute COPD exacerbation / infection",
    #   "status": "Confirm" | "Overturn" | "Fallback",
    #   "reasoning_path": "...",
    #   "audit_log": [...]
    # }

    status: Literal["pending", "processing", "success", "failed"]

    error_log: Optional[str]

    global_hint: Optional[str]

    retry_count: int

    memory_context: Optional[str]

    memory_records: Optional[List[Dict[str, Any]]]
