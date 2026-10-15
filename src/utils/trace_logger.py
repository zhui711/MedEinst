"""Record diagnostic reasoning traces."""
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional


class TraceLogger:
    """Persist detailed case reasoning traces."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._case_id: str = ""
        self._logs: Dict[str, Any] = {}
        self._reset_logs()

    @classmethod
    def get_instance(cls) -> "TraceLogger":

        return cls()

    def _reset_logs(self):

        self._logs = {
            "case_id": "",
            "timestamp": "",
            "phase1": {
                "top_candidates": [],
                "initial_diagnosis": "",
                "p_nodes_count": 0
            },
            "phase2": {
                "kgen_searches": [],
                "kpivot_searches": [],
                "extractions": [],
                "reasoning": [],
                "batch_reasoning_raw": ""
            },
            "phase3": {
                "final_diagnosis": "",
                "status": "",
                "reasoning_path": ""
            },
            "naive_scores": {},
            "graph_summary": ""
        }

    def start_case(self, case_id: str):

        self._reset_logs()
        self._case_id = case_id
        self._logs["case_id"] = case_id
        self._logs["timestamp"] = datetime.now().isoformat()
        print(f"[TraceLogger] Started tracing case: {case_id}")

    def log_phase1_result(
        self,
        top_candidates: List[str],
        initial_diagnosis: str,
        p_nodes_count: int
    ):

        self._logs["phase1"]["top_candidates"] = top_candidates
        self._logs["phase1"]["initial_diagnosis"] = initial_diagnosis
        self._logs["phase1"]["p_nodes_count"] = p_nodes_count

    def log_kgen_search(
        self,
        disease_name: str,
        disease_id: str,
        query_type: str,
        source: str,
        raw_snippet: str,
        snippet_length: int
    ):

        search_record = {
            "disease_name": disease_name,
            "disease_id": disease_id,
            "query_type": query_type,
            "source": source,
            "raw_snippet": raw_snippet[:500] if raw_snippet else "",
            "full_length": snippet_length,
            "timestamp": datetime.now().isoformat()
        }
        self._logs["phase2"]["kgen_searches"].append(search_record)

    def log_kpivot_search(
        self,
        candidate_a: str,
        candidate_b: str,
        query: str,
        raw_snippets: List[str],
        total_length: int
    ):

        truncated_snippets = [
            s[:300] + "..." if len(s) > 300 else s
            for s in raw_snippets[:3]
        ]

        search_record = {
            "comparison": f"{candidate_a} vs {candidate_b}",
            "query": query,
            "raw_snippets": truncated_snippets,
            "snippet_count": len(raw_snippets),
            "total_length": total_length,
            "timestamp": datetime.now().isoformat()
        }
        self._logs["phase2"]["kpivot_searches"].append(search_record)

    def log_extraction(
        self,
        source_type: str,
        disease_id: str,
        k_nodes_extracted: List[Dict[str, str]],
        edges_created: int
    ):

        extraction_record = {
            "source_type": source_type,
            "disease_id": disease_id,
            "k_nodes": [
                {
                    "content": k.get("content", "")[:100],
                    "importance": k.get("importance", "Common")
                }
                for k in k_nodes_extracted[:10]
            ],
            "k_nodes_count": len(k_nodes_extracted),
            "edges_created": edges_created
        }
        self._logs["phase2"]["extractions"].append(extraction_record)

    def log_reasoning(
        self,
        k_node_content: str,
        p_node_content: str,
        relation: str,
        reason: str
    ):

        reasoning_record = {
            "k_node": k_node_content[:100] if k_node_content else "",
            "p_node": p_node_content[:100] if p_node_content else "",
            "relation": relation,
            "reason": reason[:200] if reason else ""
        }
        self._logs["phase2"]["reasoning"].append(reasoning_record)

    def log_batch_reasoning_raw(self, raw_response: str, parse_success: bool = False):

        if parse_success:

            return

        self._logs["phase2"]["batch_reasoning_raw"] = raw_response[:1500] if raw_response else "(empty response)"
        print(f"[TraceLogger] ⚠️ Recorded failed Batch Reasoning response ({len(raw_response)} chars)")

    def log_phase3_result(
        self,
        final_diagnosis: str,
        status: str,
        reasoning_path: str
    ):

        self._logs["phase3"]["final_diagnosis"] = final_diagnosis
        self._logs["phase3"]["status"] = status
        self._logs["phase3"]["reasoning_path"] = reasoning_path[:500] if reasoning_path else ""

    def log_naive_scores(self, scores: Dict[str, float]):

        self._logs["naive_scores"] = scores

    def log_graph_summary(self, summary: str):

        self._logs["graph_summary"] = summary[:2000] if summary else ""

    def export_to_markdown(self, output_dir: str = "output/debug_traces") -> str:

        os.makedirs(output_dir, exist_ok=True)

        filename = f"{self._case_id}_debug.md"
        filepath = os.path.join(output_dir, filename)

        md_content = self._generate_markdown()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"[TraceLogger] Exported debug trace to: {filepath}")
        return filepath

    def _generate_markdown(self) -> str:

        lines = []

        lines.append(f"# Debug Trace: {self._logs['case_id']}")
        lines.append(f"\n**Generated:** {self._logs['timestamp']}\n")
        lines.append("---\n")

        lines.append("## 📋 Phase 1: Initial Diagnosis\n")
        p1 = self._logs["phase1"]
        lines.append(f"- **Top-5 Candidates:** {', '.join(p1['top_candidates'])}")
        lines.append(f"- **Initial Diagnosis:** {p1['initial_diagnosis']}")
        lines.append(f"- **P-Nodes Count:** {p1['p_nodes_count']}\n")

        lines.append("## 🔍 Phase 2: Knowledge Search Details\n")

        if self._logs["phase2"]["kgen_searches"]:
            lines.append("### K-Gen Searches (General Knowledge)\n")
            lines.append("| Disease | Source | Snippet Length | Raw Snippet Preview |")
            lines.append("|---------|--------|----------------|---------------------|")

            for search in self._logs["phase2"]["kgen_searches"]:
                snippet_preview = search["raw_snippet"][:150].replace("\n", " ").replace("|", "\\|")
                if len(search["raw_snippet"]) > 150:
                    snippet_preview += "..."
                lines.append(
                    f"| **{search['disease_name']}** | {search['source']} | "
                    f"{search['full_length']} chars | {snippet_preview} |"
                )
            lines.append("")

        if self._logs["phase2"]["kpivot_searches"]:
            lines.append("### 🎯 K-Pivot Searches (Differential Diagnosis) - CRITICAL\n")

            for i, search in enumerate(self._logs["phase2"]["kpivot_searches"], 1):
                lines.append(f"#### Comparison {i}: {search['comparison']}\n")
                lines.append(f"**Query:** `{search['query']}`\n")
                lines.append(f"**Results:** {search['snippet_count']} abstracts, {search['total_length']} total chars\n")

                if search["raw_snippets"]:
                    lines.append("**Raw Snippets Preview:**\n")
                    for j, snippet in enumerate(search["raw_snippets"], 1):
                        lines.append(f"```text\n[Snippet {j}]\n{snippet}\n```\n")
                lines.append("")

        if self._logs["phase2"]["extractions"]:
            lines.append("### 📦 K-Node Extractions\n")
            lines.append("| Source Type | Disease ID | K-Nodes | Edges |")
            lines.append("|-------------|------------|---------|-------|")

            for ext in self._logs["phase2"]["extractions"]:
                lines.append(
                    f"| {ext['source_type']} | {ext['disease_id']} | "
                    f"{ext['k_nodes_count']} | {ext['edges_created']} |"
                )
            lines.append("")

            lines.append("<details>\n<summary>Click to expand K-Node details</summary>\n")
            for ext in self._logs["phase2"]["extractions"]:
                lines.append(f"\n**{ext['disease_id']}:**")
                for k in ext["k_nodes"]:
                    lines.append(f"- [{k['importance']}] {k['content']}")
            lines.append("\n</details>\n")

        if self._logs["phase2"]["reasoning"]:
            lines.append("### 🧠 Match/Conflict Reasoning\n")
            lines.append("| K-Node | P-Node | Relation | Reason |")
            lines.append("|--------|--------|----------|--------|")

            for r in self._logs["phase2"]["reasoning"][:20]:
                k_content = r["k_node"][:50].replace("|", "\\|")
                p_content = r["p_node"][:50].replace("|", "\\|")
                reason = r["reason"][:100].replace("|", "\\|")
                relation_emoji = {"Match": "✅", "Conflict": "❌", "Void": "❓"}.get(r["relation"], "")
                lines.append(f"| {k_content} | {p_content} | {relation_emoji} {r['relation']} | {reason} |")
            lines.append("")

        batch_raw = self._logs["phase2"].get("batch_reasoning_raw", "")
        if batch_raw:
            lines.append("### ⚠️ Batch Reasoning Parse Failed - Raw LLM Response\n")
            lines.append("<details>")
            lines.append("<summary>Click to expand raw response (diagnostic info)</summary>\n")
            lines.append("```json")

            safe_raw = batch_raw.replace("```", "'''")
            lines.append(safe_raw)
            lines.append("```")
            lines.append("</details>\n")

        # Naive Scores
        if self._logs["naive_scores"]:
            lines.append("## 📊 Naive Scores (Deterministic)\n")
            lines.append("| Candidate | Score |")
            lines.append("|-----------|-------|")

            sorted_scores = sorted(
                self._logs["naive_scores"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for d_id, score in sorted_scores:
                lines.append(f"| {d_id} | {score:.2f} |")
            lines.append("")

        # Graph Summary
        if self._logs["graph_summary"]:
            lines.append("## 📝 Graph Summary (Input to Phase 3)\n")
            lines.append("```text")
            lines.append(self._logs["graph_summary"])
            lines.append("```\n")

        lines.append("## ⚖️ Phase 3: Final Verdict\n")
        p3 = self._logs["phase3"]
        status_emoji = {"Confirm": "✅", "Overturn": "🔄", "Fallback": "⚠️"}.get(p3["status"], "")
        lines.append(f"- **Final Diagnosis:** {p3['final_diagnosis']}")
        lines.append(f"- **Status:** {status_emoji} {p3['status']}")
        lines.append(f"\n**Reasoning Path:**\n```\n{p3['reasoning_path']}\n```\n")

        lines.append("---")
        lines.append(f"*Generated by TraceLogger at {datetime.now().isoformat()}*")

        return "\n".join(lines)

    def get_logs(self) -> Dict[str, Any]:

        return self._logs.copy()

