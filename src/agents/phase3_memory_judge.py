"""Online evidence audit with Exemplar Base retrieval."""

import json
from typing import Dict, Any, Optional, List

from .phase3_debate import JudgeAgent
from src.utils.api_client import LLMClient
from src.memory.memory_bank import MemoryBankManager
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.json_utils import parse_json_from_text
from config.prompt_phase3_judge import PHASE3_JUDGE_SYSTEM_PROMPT


class MemoryAugmentedJudge(JudgeAgent):
    """Audit Online evidence with retrieved case exemplars."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "qwen3-32b",
        memory_bank: Optional[MemoryBankManager] = None,
        memory_bank_dir: str = "memory_bank"
    ):

        super().__init__(llm_client, model_name)

        if memory_bank is not None:
            self.memory_bank = memory_bank
        else:
            self.memory_bank = MemoryBankManager(output_dir=memory_bank_dir)
            try:
                self.memory_bank.load()
                print(f"[MemoryAugmentedJudge] Loaded Memory Bank: "
                      f"{self.memory_bank.get_statistics()}")
            except Exception as e:
                print(f"[MemoryAugmentedJudge] Warning: Failed to load Memory Bank: {e}")

        self._stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "avg_similarity_overturn": 0.0,
            "avg_similarity_confirm": 0.0
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:

        try:

            graph_json = state.get("graph_json")
            if not graph_json:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Graph JSON is missing"
                }

            phase1_result = state.get("phase1_result", {})
            input_case = state.get("input_case", {})
            raw_text = input_case.get("narrative", "")

            graph_summary = state.get("graph_summary", "")
            naive_scores = state.get("naive_scores", {})

            # ========== 2. JIT Memory Retrieval ==========

            memory_records = state.get("memory_records") or []

            if not memory_records:

                memory_records = self._retrieve_from_graph_p_nodes(graph_json)

                state["memory_records"] = memory_records

            few_shot_context = self._format_few_shot_from_records(memory_records)

            user_prompt = self._construct_unified_prompt(
                graph_json=graph_json,
                phase1_result=phase1_result,
                raw_text=raw_text,
                graph_summary=graph_summary,
                naive_scores=naive_scores,
                few_shot_context=few_shot_context
            )

            messages = [
                {"role": "system", "content": PHASE3_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            result = self.llm_client.generate_json(
                messages=messages,
                model=self.model_name,
                logprobs=False,
                temperature=0.2,
                max_tokens=4096
            )

            if result["error"]:
                print(f"[MemoryAugmentedJudge] LLM error: {result['error']}")
                return {
                    **state,
                    "status": "failed",
                    "error_log": f"LLM error: {result['error']}"
                }

            response_content = result["content"]
            parsed_json = parse_json_from_text(response_content, verbose=True)

            if not parsed_json:
                print(f"[MemoryAugmentedJudge] Failed to parse JSON")
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Failed to parse JSON from LLM response"
                }

            final_output = self._validate_output_unified(parsed_json, phase1_result)

            final_output["memory_retrieval_used"] = bool(memory_records)

            print(f"[MemoryAugmentedJudge] Final diagnosis: {final_output.get('final_diagnosis_id')} - "
                  f"{final_output.get('final_diagnosis_name')} ({final_output.get('status')})")

            return {
                **state,
                "final_output": final_output,
                "memory_records": memory_records,
                "status": "success"
            }

        except Exception as e:
            import traceback
            error_msg = f"Phase 3 Memory-Augmented error: {str(e)}\n{traceback.format_exc()}"
            print(f"[MemoryAugmentedJudge] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }

    def _retrieve_from_graph_p_nodes(
        self,
        graph_json: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        self._stats["total_retrievals"] += 1

        try:
            if self.memory_bank is None:
                print("[MemoryAugmentedJudge] Memory Bank not initialized")
                return []

            p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])

            if not p_nodes:
                print("[MemoryAugmentedJudge] No P-Nodes in graph for retrieval")
                return []

            # Retrieve both corrected and confirmed cases for the final evidence audit.
            similar_cases = self.memory_bank.retrieve_similar(
                query_p_nodes=p_nodes,
                n_overturn=2,
                n_confirm=2
            )

            overturn_cases = similar_cases.get("overturn", [])
            confirm_cases = similar_cases.get("confirm", [])

            all_cases = overturn_cases + confirm_cases

            if not all_cases:
                print("[MemoryAugmentedJudge] No similar cases found")
                return []

            self._stats["successful_retrievals"] += 1

            if overturn_cases:
                avg_sim = sum(c.get("similarity_score", 0) for c in overturn_cases) / len(overturn_cases)
                self._stats["avg_similarity_overturn"] = avg_sim
            if confirm_cases:
                avg_sim = sum(c.get("similarity_score", 0) for c in confirm_cases) / len(confirm_cases)
                self._stats["avg_similarity_confirm"] = avg_sim

            print(f"[MemoryAugmentedJudge] Retrieved {len(overturn_cases)} Overturn + {len(confirm_cases)} Confirm cases")
            return all_cases

        except Exception as e:
            print(f"[MemoryAugmentedJudge] Retrieval error: {e}")
            return []

    def _format_few_shot_from_records(
        self,
        memory_records: List[Dict[str, Any]]
    ) -> str:

        if not memory_records:
            return ""

        lines = ["### 🧠 SIMILAR HISTORICAL CASES"]
        lines.append("*Reference cases from training data. Use their reasoning patterns as guidance.*\n")

        for idx, case in enumerate(memory_records, 1):

            initial = case.get("initial_diagnosis_name", "Unknown")
            final = case.get("final_diagnosis_name", "Unknown")
            ground_truth = case.get("ground_truth_name", "Unknown")
            outcome = case.get("outcome", "Unknown")
            p_summary = case.get("p_nodes_summary", "N/A")
            similarity = case.get("similarity_score", 0.0)

            is_correct = (final == ground_truth)
            correctness = "✅ Correct" if is_correct else "❌ Incorrect"

            lines.append(f"**[Case {idx}]** (Similarity: {similarity:.2f})")
            lines.append(f"  - Patient Summary: {p_summary}")
            lines.append(f"  - Initial Diagnosis: {initial}")
            lines.append(f"  - Final Diagnosis: {final} ({correctness})")
            lines.append(f"  - Outcome: {outcome}")

            # Key Insight
            if outcome == "Overturn":
                lines.append(f"  - 💡 Insight: Changed from {initial} to {final}")
            else:
                lines.append(f"  - 💡 Insight: Confirmed initial diagnosis {initial}")
            lines.append("")

        return "\n".join(lines)

    def _retrieve_and_format_context(
        self,
        phase1_result: Dict[str, Any]
    ) -> str:

        track_b_output = phase1_result.get("track_b_output", {})
        p_nodes = track_b_output.get("p_nodes", [])

        if not p_nodes or self.memory_bank is None:
            return ""

        try:
            similar_cases = self.memory_bank.retrieve_similar(
                query_p_nodes=p_nodes,
                n_overturn=2,
                n_confirm=2
            )
            all_cases = similar_cases.get("overturn", []) + similar_cases.get("confirm", [])
            return self._format_few_shot_from_records(all_cases)
        except Exception as e:
            print(f"[MemoryAugmentedJudge] Deprecated retrieval error: {e}")
            return ""

    def _format_hidden_label_context(
        self,
        overturn_cases: List[Dict[str, Any]],
        confirm_cases: List[Dict[str, Any]]
    ) -> str:

        all_cases = overturn_cases + confirm_cases
        return self._format_few_shot_from_records(all_cases)

    def _construct_unified_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        graph_summary: str,
        naive_scores: Dict[str, float],
        few_shot_context: str
    ) -> str:

        initial_candidates = phase1_result.get("top_candidates", [])
        initial_candidates_list = ", ".join([
            f"{DIAGNOSIS_ID_MAP.get(cid, 'Unknown')} (ID: {cid})"
            for cid in initial_candidates
        ])
        initial_reasoning = phase1_result.get("differential_reasoning", "")

        naive_scores_section = self._format_naive_scores_unified(
            naive_scores or {},
            initial_candidates
        )

        return f"""
## PATIENT CASE

{raw_text}

## PHASE 1 INITIAL ASSESSMENT

**Initial Candidates:** {initial_candidates_list}

**Differential Reasoning:**
{initial_reasoning}

## 📊 NAIVE SCORES (Reference Baseline)

{naive_scores_section}

## ⚖️ RICH EVIDENCE SUMMARY

{graph_summary}

{few_shot_context}

## YOUR TASK

Based on the above evidence, determine the **final diagnosis**.

### Decision Framework:

1. **Fatal Conflict Check (Tier 1):**
   - If a candidate has a **Conflict** on an **Essential/Pathognomonic** feature (P-Node status = "Absent"), it is **disqualified**.
   - ⚠️ **Shadow ≠ Conflict**: Missing symptoms (Shadow) are NOT disqualifying factors.

2. **Pivot Competition (Tier 2):**
   - Candidates with **matched Pivot Features** are superior to those without.
   - If a lower-ranked candidate has Pivot support but Top-1 does not, consider **Overturn**.

3. **Evidence Coverage (Tier 3):**
   - Use **Naive Scores** as a quantitative baseline.
   - Higher score = more supporting evidence.

### ⚠️ CRITICAL INSTRUCTIONS ON SHADOW NODES:

1. **Do NOT penalize candidates based on the *count* or *number* of Shadow nodes.**
2. **Only penalize if CRITICAL Pathognomonic evidence is missing:**
   - Example: Missing "D-dimer elevation" for Pulmonary Embolism = significant concern.
   - Example: Missing "Fatigue" = ignore (not a deal-breaker).
3. **Naive Score Context:** The score formula uses Shadow×0.1 penalty. If a candidate's score appears low due to many irrelevant Shadows, use your **medical judgment to override** the score.

### Output Format (JSON):

```json
{{
    "final_diagnosis_id": "XX",
    "final_diagnosis_name": "Disease Name",
    "status": "Confirm|Overturn|Fallback",
    "reasoning_path": "Step-by-step reasoning explaining your decision...",
    "audit_log": ["Key decision points..."]
}}
```
"""

    def _format_naive_scores_unified(
        self,
        naive_scores: Dict[str, float],
        top_candidates: List[str]
    ) -> str:

        if not naive_scores:
            return "*No Naive Scores available*"

        lines = ["**Formula:** `Score = (Match × 1.0) - (Conflict × 1.5) - (Shadow × 0.1)`"]
        lines.append("")
        lines.append("*Note: Shadow penalty (0.1) is minimal. Do NOT over-penalize missing evidence.*")
        lines.append("")

        max_score = max(naive_scores.values()) if naive_scores else 0
        top1_score = naive_scores.get(f"d_{top_candidates[0]}", 0) if top_candidates else 0

        for rank, cand_id in enumerate(top_candidates, 1):
            d_id = f"d_{cand_id}"
            score = naive_scores.get(d_id, 0.0)

            signal = ""
            if rank > 1 and score > top1_score:
                signal = " ⬆️ **Higher than Top-1 (Potential Overturn Signal)**"
            elif score == max_score and score > 0:
                signal = " 🏆 **Highest**"

            disease_name = DIAGNOSIS_ID_MAP.get(cand_id, "Unknown")
            score_str = f"+{score:.1f}" if score >= 0 else f"{score:.1f}"
            lines.append(f"- **{d_id}** ({disease_name}) [Rank {rank}]: Score = **{score_str}**{signal}")

        return "\n".join(lines)

    def _validate_output_unified(
        self,
        parsed_json: Dict[str, Any],
        phase1_result: Dict[str, Any]
    ) -> Dict[str, Any]:

        final_diagnosis_id = parsed_json.get("final_diagnosis_id")

        if not final_diagnosis_id or final_diagnosis_id not in DIAGNOSIS_ID_MAP:

            top_candidates = phase1_result.get("top_candidates", [])

            if top_candidates and top_candidates[0] in DIAGNOSIS_ID_MAP:
                fallback_id = top_candidates[0]
                print(f"[MemoryAugmentedJudge] Invalid ID '{final_diagnosis_id}', using fallback: {fallback_id}")
                parsed_json["final_diagnosis_id"] = fallback_id
                parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[fallback_id]
                parsed_json["status"] = "Fallback"
            elif "01" in DIAGNOSIS_ID_MAP:
                parsed_json["final_diagnosis_id"] = "01"
                parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP["01"]
                parsed_json["status"] = "Emergency_Fallback"
        else:

            parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP.get(
                final_diagnosis_id,
                parsed_json.get("final_diagnosis_name", "Unknown")
            )

            phase1_final_id = phase1_result.get("final_diagnosis_id", "")
            if final_diagnosis_id == phase1_final_id:
                parsed_json["status"] = "Confirm"
            else:
                parsed_json["status"] = "Overturn"

        if "reasoning_path" not in parsed_json:
            parsed_json["reasoning_path"] = "Diagnosis selected based on graph analysis."
        if "audit_log" not in parsed_json:
            parsed_json["audit_log"] = []

        return parsed_json

    def _construct_memory_augmented_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        audit_results: Dict[str, Any],
        graph_summary: str,
        naive_scores: Dict[str, float],
        few_shot_context: str
    ) -> str:

        return self._construct_unified_prompt(
            graph_json=graph_json,
            phase1_result=phase1_result,
            raw_text=raw_text,
            graph_summary=graph_summary,
            naive_scores=naive_scores,
            few_shot_context=few_shot_context
        )

    def get_statistics(self) -> Dict[str, Any]:

        return self._stats.copy()

def create_judge_agent(
    llm_client: LLMClient,
    model_name: str = "qwen3-32b",
    memory_bank_dir: str = "memory_bank",
    memory_bank: Optional[MemoryBankManager] = None
) -> MemoryAugmentedJudge:

    return MemoryAugmentedJudge(
        llm_client=llm_client,
        model_name=model_name,
        memory_bank=memory_bank,
        memory_bank_dir=memory_bank_dir
    )

if __name__ == "__main__":
    print("MemoryAugmentedJudge module loaded successfully")
    print("Use create_judge_agent() to instantiate")

