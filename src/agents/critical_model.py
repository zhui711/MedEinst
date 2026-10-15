"""Critic feedback for unsuccessful offline diagnostic attempts."""

from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.graph_tools import summarize_graph_for_critique
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from config.prompt_critical_model import (
    CRITICAL_MODEL_SYSTEM_PROMPT,
    build_critique_prompt,
    parse_critique_response,
    ERROR_TYPE_GT_NOT_IN_TOP5,
    ERROR_TYPE_FINAL_WRONG
)


class CriticalModelAgent:
    """Provide corrective feedback after an unsuccessful CGME attempt."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-5"
    ):

        self.llm_client = llm_client
        self.model_name = model_name

        self._critique_history: List[str] = []

    def generate_critique(
        self,
        narrative: str,
        ground_truth_id: str,
        phase1_result: Dict[str, Any],
        final_output: Dict[str, Any],
        graph_json: Dict[str, Any],
        previous_critiques: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        top_candidates = phase1_result.get("top_candidates", [])
        initial_diagnosis_id = phase1_result.get("final_diagnosis_id", "")
        final_diagnosis_id = final_output.get("final_diagnosis_id", "")

        gt_id_variants = [ground_truth_id, f"d_{ground_truth_id}", ground_truth_id.lstrip("d_")]
        gt_in_top5 = any(
            gt_var in top_candidates or gt_var.lstrip("d_") in [c.lstrip("d_") for c in top_candidates]
            for gt_var in gt_id_variants
        )

        if not gt_in_top5:
            error_type = ERROR_TYPE_GT_NOT_IN_TOP5
        else:
            error_type = ERROR_TYPE_FINAL_WRONG

        ground_truth_name = DIAGNOSIS_ID_MAP.get(
            ground_truth_id.lstrip("d_"),
            "Unknown Diagnosis"
        )
        initial_diagnosis_name = DIAGNOSIS_ID_MAP.get(
            initial_diagnosis_id.lstrip("d_"),
            "Unknown"
        )
        final_diagnosis_name = final_output.get(
            "final_diagnosis_name",
            DIAGNOSIS_ID_MAP.get(final_diagnosis_id.lstrip("d_"), "Unknown")
        )

        graph_summary = summarize_graph_for_critique(graph_json, ground_truth_id)

        previous_critiques_text = ""
        if previous_critiques:
            for i, critique in enumerate(previous_critiques, 1):
                previous_critiques_text += f"\n--- Attempt {i} Critique ---\n{critique}\n"

        user_prompt = build_critique_prompt(
            narrative=narrative,
            ground_truth_id=ground_truth_id,
            ground_truth_name=ground_truth_name,
            error_type=error_type,
            top_candidates=top_candidates,
            initial_diagnosis=initial_diagnosis_name,
            final_diagnosis=final_diagnosis_name,
            graph_summary=graph_summary,
            previous_critiques=previous_critiques_text
        )

        messages = [
            {"role": "system", "content": CRITICAL_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                temperature=0.3,
                max_tokens=2048
            )

            if result.get("error"):
                return {
                    "error": f"LLM call failed: {result['error']}",
                    "critique_for_agent": "Please re-examine all clinical features carefully."
                }

            critique = parse_critique_response(result["content"])

            print(f"[CriticalModel] Generated critique for {ground_truth_name}")
            print(f"    Error Type: {error_type}")
            print(f"    Missed Features: {critique.get('missed_features', [])}")

            return critique

        except Exception as e:
            print(f"[CriticalModel] Error generating critique: {e}")
            return {
                "error": str(e),
                "critique_for_agent": "Please re-examine all clinical features carefully."
            }

    def get_injectable_critique(self, critique: Dict[str, Any]) -> str:

        if "critique_for_agent" in critique:
            return critique["critique_for_agent"]

        parts = []

        if critique.get("missed_features"):
            features = ", ".join(critique["missed_features"][:5])
            parts.append(f"Consider these potentially missed features: {features}")

        if critique.get("reasoning_guidance"):
            parts.append(critique["reasoning_guidance"][:300])

        if not parts:
            return "Please re-examine all clinical features carefully."

        return " ".join(parts)

    def reset_history(self) -> None:

        self._critique_history = []

    def add_to_history(self, critique: str) -> None:

        self._critique_history.append(critique)

    def get_history(self) -> List[str]:

        return self._critique_history.copy()

if __name__ == "__main__":

    print("CriticalModelAgent module loaded successfully.")
    print(f"Error types defined:")
    print(f"  - {ERROR_TYPE_GT_NOT_IN_TOP5}")
    print(f"  - {ERROR_TYPE_FINAL_WRONG}")

