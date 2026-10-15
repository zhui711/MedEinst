"""Offline evidence audit and diagnostic judgment."""
import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.json_utils import parse_json_from_text
from config.prompt_phase3_judge import PHASE3_JUDGE_SYSTEM_PROMPT, PHASE3_JUDGE_USER_PROMPT_TEMPLATE


class JudgeAgent:
    """Produce the Offline diagnostic judgment from graph evidence."""

    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):

        self.llm_client = llm_client
        self.model_name = model_name
        self.risk_map = self._load_risk_map()

        self.shadow_threshold = 0.5

    def _load_risk_map(self) -> Dict[str, Dict[str, str]]:

        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "disease_metadata.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

                risk_map = {}
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    match = re.match(r'"(\d+)":\s*\{\s*"name":\s*"([^"]+)",\s*"risk":\s*"([^"]+)"\s*\}', line)
                    if match:
                        disease_id, name, risk = match.groups()
                        risk_map[disease_id] = {"name": name, "risk": risk}

                return risk_map
        except Exception as e:
            print(f"[JudgeAgent] Error loading risk map: {e}")
            return {}

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

            audit_results = self._pre_audit(graph_json, phase1_result)

            user_prompt = self._construct_judge_prompt(
                graph_json,
                phase1_result,
                raw_text,
                audit_results,
                graph_summary=graph_summary,
                naive_scores=naive_scores
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
                print(f"[JudgeAgent] LLM error: {result['error']}")
                return {
                    **state,
                    "status": "failed",
                    "error_log": f"LLM error: {result['error']}"
                }

            response_content = result["content"]
            print(f"[JudgeAgent] LLM response length: {len(response_content)} chars")

            parsed_json = parse_json_from_text(response_content, verbose=True)

            if not parsed_json:
                print(f"[JudgeAgent] Failed to parse JSON")
                print(f"[JudgeAgent] Raw response preview: {response_content[:500]}...")
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Failed to parse JSON from LLM response"
                }

            final_output = self._validate_and_fix_output(parsed_json, phase1_result, audit_results)

            print(f"[JudgeAgent] Final diagnosis: {final_output.get('final_diagnosis_id')} - "
                  f"{final_output.get('final_diagnosis_name')} ({final_output.get('status')})")

            return {
                **state,
                "final_output": final_output,
                "status": "success"
            }

        except Exception as e:
            import traceback
            error_msg = f"Phase 3 error: {str(e)}\n{traceback.format_exc()}"
            print(f"[JudgeAgent] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }

    def _pre_audit(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any]
    ) -> Dict[str, Any]:

        audit_results = {
            "disqualified_candidates": [],
            "safeguarded_candidates": [],
            "shadow_penalties": {},
            "coverage_scores": {}
        }

        d_nodes = graph_json.get("graph", {}).get("nodes", {}).get("d_nodes", [])
        k_nodes = graph_json.get("graph", {}).get("nodes", {}).get("k_nodes", [])
        p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])
        p_k_links = graph_json.get("graph", {}).get("edges", {}).get("p_k_links", [])
        k_d_links = graph_json.get("graph", {}).get("edges", {}).get("k_d_links", [])

        k_node_map = {k["id"]: k for k in k_nodes}
        p_node_map = {p["id"]: p for p in p_nodes}

        # Round 1: Safety Sentinel (Line 42-47)
        for d_node in d_nodes:
            d_id = d_node["id"]
            original_id = d_node.get("original_id", "")
            risk_info = self.risk_map.get(original_id, {})
            risk_level = risk_info.get("risk", "Low")

            has_fatal_conflict = self._check_fatal_conflict(
                d_id, k_d_links, p_k_links, k_node_map, p_node_map
            )

            if has_fatal_conflict:
                audit_results["disqualified_candidates"].append({
                    "d_id": d_id,
                    "reason": "Fatal Conflict (Essential symptom absent)"
                })
            elif risk_level == "High":

                audit_results["safeguarded_candidates"].append({
                    "d_id": d_id,
                    "reason": "High Risk disease without Fatal Conflict"
                })

        # Round 2: Shadow Interrogation (Line 48-52)
        initial_d_id = f"d_{phase1_result.get('final_diagnosis_id', phase1_result.get('top_candidates', [''])[0])}"

        for d_node in d_nodes:
            d_id = d_node["id"]

            shadow_count, total_count = self._count_shadows(
                d_id, k_d_links, p_k_links
            )

            if total_count > 0:
                shadow_ratio = shadow_count / total_count
                audit_results["shadow_penalties"][d_id] = {
                    "shadow_count": shadow_count,
                    "total_count": total_count,
                    "ratio": shadow_ratio
                }

                if d_id == initial_d_id and shadow_ratio > self.shadow_threshold:
                    audit_results["shadow_penalties"][d_id]["penalty_applied"] = True

            coverage = self._calculate_coverage(d_id, k_d_links, p_k_links)
            audit_results["coverage_scores"][d_id] = coverage

        return audit_results

    def _check_fatal_conflict(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict],
        k_node_map: Dict,
        p_node_map: Dict
    ) -> bool:

        related_k_ids = [
            link["source"] for link in k_d_links
            if link["target"] == d_id
        ]

        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            importance = k_node.get("importance", "Common")

            if importance not in ["Essential", "Pathognomonic"]:
                continue

            for pk_link in p_k_links:
                if pk_link["target"] == k_id and pk_link["relation"] == "Conflict":
                    return True

        return False

    def _count_shadows(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict]
    ) -> tuple:

        related_k_ids = [
            link["source"] for link in k_d_links
            if link["target"] == d_id
        ]

        shadow_count = 0
        total_count = len(related_k_ids)

        for k_id in related_k_ids:

            for pk_link in p_k_links:
                if pk_link["target"] == k_id and pk_link["relation"] == "Void":
                    shadow_count += 1
                    break

        return shadow_count, total_count

    def _calculate_coverage(
        self,
        d_id: str,
        k_d_links: List[Dict],
        p_k_links: List[Dict]
    ) -> float:

        related_k_ids = [
            link["source"] for link in k_d_links
            if link["target"] == d_id
        ]

        match_count = 0
        void_count = 0
        conflict_count = 0

        for k_id in related_k_ids:
            for pk_link in p_k_links:
                if pk_link["target"] == k_id:
                    relation = pk_link["relation"]
                    if relation == "Match":
                        match_count += 1
                    elif relation == "Void":
                        void_count += 1
                    elif relation == "Conflict":
                        conflict_count += 1

        coverage = match_count - (void_count * 0.1) - (conflict_count * 1.5)
        return coverage

    def _construct_judge_prompt(
        self,
        graph_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        raw_text: str,
        audit_results: Dict[str, Any],
        graph_summary: str = "",
        naive_scores: Dict[str, float] = None
    ) -> str:

        p_nodes = graph_json.get("graph", {}).get("nodes", {}).get("p_nodes", [])
        p_nodes_json = json.dumps(p_nodes, ensure_ascii=False, indent=2)

        d_nodes = graph_json.get("graph", {}).get("nodes", {}).get("d_nodes", [])

        low_evidence_diseases = graph_json.get("low_evidence_diseases", [])
        knowledge_sources = graph_json.get("knowledge_sources", {})

        for d_node in d_nodes:
            original_id = d_node.get("original_id", "")
            risk_info = self.risk_map.get(original_id, {})
            d_node["risk_level"] = risk_info.get("risk", "Low")

            d_id = d_node.get("id", "")
            d_node["knowledge_source"] = knowledge_sources.get(d_id, "Unknown")

            disease_name = d_node.get("name", "")
            if disease_name in low_evidence_diseases:
                d_node["low_evidence"] = True
            else:
                d_node["low_evidence"] = False

        pre_audit_info = {
            "disqualified_candidates": audit_results.get("disqualified_candidates", []),
            "safeguarded_candidates": audit_results.get("safeguarded_candidates", []),
                "coverage_scores": audit_results.get("coverage_scores", {})
        }
        pre_audit_json = json.dumps(pre_audit_info, ensure_ascii=False, indent=2)

        low_evidence_warning = ""
        if low_evidence_diseases:
            low_evidence_warning = (
                "### ⚠️ LOW EVIDENCE WARNING\n"
                f"The following diseases have limited knowledge (search returned no results): {', '.join(low_evidence_diseases)}\n"
                "**Do NOT automatically disqualify these diseases!** Consider them based on:\n"
                "1. Existing Match edges (if any)\n"
                "2. Absence of Fatal Conflicts\n"
                "3. Clinical reasoning from the patient narrative\n"
            )

        initial_candidates = phase1_result.get("top_candidates", [])
        initial_candidates_list = ", ".join([
            f"{DIAGNOSIS_ID_MAP.get(cid, 'Unknown')} (ID: {cid})"
            for cid in initial_candidates
        ])

        initial_reasoning = phase1_result.get("differential_reasoning", "")

        if graph_summary:

            graph_evidence_section = f"""
### GRAPH EVIDENCE SUMMARY (Structured Natural Language)

{graph_summary}

### PRE-AUDIT RESULTS (System Computed)

```json
{pre_audit_json}
```
"""
        else:

            k_nodes = graph_json.get("graph", {}).get("nodes", {}).get("k_nodes", [])
            edges = graph_json.get("graph", {}).get("edges", {})

            graph_edges_info = {
                "p_k_links": edges.get("p_k_links", []),
                "k_d_links": edges.get("k_d_links", []),
                "k_nodes": k_nodes,
                "d_nodes": d_nodes,
                "pre_audit": pre_audit_info
            }
            graph_edges_json = json.dumps(graph_edges_info, ensure_ascii=False, indent=2)
            graph_evidence_section = f"""
### GRAPH EVIDENCE (JSON Format - Fallback)

```json
{graph_edges_json}
```
"""

        naive_scores_section = self._format_naive_scores(
            naive_scores or {},
            phase1_result.get("top_candidates", [])
        )

        user_prompt = self._format_judge_prompt_v2(
            raw_text=raw_text,
            initial_candidates_list=initial_candidates_list,
            initial_reasoning=initial_reasoning,
            p_nodes_json=p_nodes_json,
            graph_evidence_section=graph_evidence_section,
            low_evidence_warning=low_evidence_warning,
            naive_scores_section=naive_scores_section
        )

        return user_prompt

    def _format_naive_scores(
        self,
        naive_scores: Dict[str, float],
        top_candidates: List[str]
    ) -> str:

        if not naive_scores:
            return ""

        lines = ["### 📊 NAIVE SCORES (Evidence Strength Reference)\n"]
        lines.append("*Formula: Score = (Match × 1.0) - (Conflict × 1.5) - (Shadow × 0.1)*\n")

        max_score = max(naive_scores.values()) if naive_scores else 0
        top1_score = naive_scores.get(f"d_{top_candidates[0]}", 0) if top_candidates else 0

        for rank, cand_id in enumerate(top_candidates, 1):
            d_id = f"d_{cand_id}"
            score = naive_scores.get(d_id, 0.0)

            signal = ""
            if rank > 1 and score > top1_score:
                signal = " ⬆️ **Higher than Top-1**"
            elif score == max_score and score > 0:
                signal = " 🏆"

            disease_name = DIAGNOSIS_ID_MAP.get(cand_id, "Unknown")
            lines.append(f"- **{d_id}** ({disease_name}) [Phase1 Rank: {rank}]: Score = **{score:.1f}**{signal}")

        lines.append("")
        return "\n".join(lines)

    def _format_judge_prompt_v2(
        self,
        raw_text: str,
        initial_candidates_list: str,
        initial_reasoning: str,
        p_nodes_json: str,
        graph_evidence_section: str,
        low_evidence_warning: str,
        naive_scores_section: str = ""
    ) -> str:

        return f"""
## PATIENT CASE

{raw_text}

## PHASE 1 INITIAL ASSESSMENT

**Initial Candidates:** {initial_candidates_list}

**Differential Reasoning:**
{initial_reasoning}

## PATIENT FINDINGS (P-Nodes)

```json
{p_nodes_json}
```

{graph_evidence_section}

{naive_scores_section}

{low_evidence_warning}

## YOUR TASK

Based on the above evidence, determine the final diagnosis. Apply the Tiered Logic:

1. **Safety Sentinel (Tier 1):** Check for Fatal Conflicts on Essential/Pathognomonic features (P-Node status must be "Absent", not just missing)
2. **Pivot Competition (Tier 2):** Compare Pivot support - candidates with matched Pivot Features are superior
3. **Coverage Audit (Tier 3):** Use Naive Scores as tie-breaker; select highest coverage

⚠️ **IMPORTANT:** A "Missing" symptom (Shadow) is NOT a Conflict. Only "Absent" (patient denied) causes Fatal Conflict.

Output your decision in JSON format:
```json
{{
    "final_diagnosis_id": "XX",
    "final_diagnosis_name": "Disease Name",
    "status": "Confirm|Overturn|Fallback",
    "reasoning_path": "Step-by-step reasoning...",
    "audit_log": ["Key decision points..."]
}}
```
"""

    def _validate_and_fix_output(
        self,
        parsed_json: Dict[str, Any],
        phase1_result: Dict[str, Any],
        audit_results: Dict[str, Any]
    ) -> Dict[str, Any]:

        final_diagnosis_id = parsed_json.get("final_diagnosis_id")

        if not final_diagnosis_id or final_diagnosis_id not in DIAGNOSIS_ID_MAP:

            top_candidates = phase1_result.get("top_candidates", [])

            disqualified_ids = [
                d["d_id"].replace("d_", "")
                for d in audit_results.get("disqualified_candidates", [])
            ]

            for candidate_id in top_candidates:
                if candidate_id in DIAGNOSIS_ID_MAP and candidate_id not in disqualified_ids:
                    print(f"[JudgeAgent] Invalid ID '{final_diagnosis_id}', using fallback: {candidate_id}")
                    parsed_json["final_diagnosis_id"] = candidate_id
                    parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[candidate_id]
                    parsed_json["status"] = "Fallback"
                    break
            else:

                if top_candidates and top_candidates[0] in DIAGNOSIS_ID_MAP:
                    parsed_json["final_diagnosis_id"] = top_candidates[0]
                    parsed_json["final_diagnosis_name"] = DIAGNOSIS_ID_MAP[top_candidates[0]]
                    parsed_json["status"] = "Emergency_Fallback"
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
