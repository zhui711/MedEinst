"""Online graph reasoning with prior Illness Graphs and live evidence."""

import json
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass

import yaml

from src.graph.schema import MedicalGraph
from src.graph.canonical_graph import CanonicalGraph, CanonicalKNode
from src.graph.golden_graph_loader import GoldenGraphLoader
from src.agents.phase2_investigation import Phase2GraphReasoning
from src.utils.api_client import LLMClient
from src.utils.json_utils import robust_parse_k_gen_response
from src.utils.text_utils import SemanticMatcher, MatchResult
from config.prompt_phase2_KPivot import (
    PHASE2_BATCH_SYSTEM_PROMPT,
    build_batch_prompt
)


@dataclass
class PruningStats:
    """Counts for illness-graph node matching and pruning."""
    gg_p_nodes_total: int = 0
    gg_p_nodes_matched: int = 0
    gg_p_nodes_pruned: int = 0
    gg_k_nodes_total: int = 0
    gg_k_nodes_kept_match: int = 0
    gg_k_nodes_kept_pathognomonic: int = 0
    gg_k_nodes_dropped: int = 0
    pathognomonic_shadows_created: int = 0

    def log_summary(self, disease_name: str) -> None:

        print(f"[Phase 2] Pruning Stats for '{disease_name}':")
        print(f"  - P-Nodes: {self.gg_p_nodes_matched} matched, "
              f"{self.gg_p_nodes_pruned} pruned (of {self.gg_p_nodes_total} total)")
        print(f"  - K-Nodes: Kept {self.gg_k_nodes_kept_match} (Match) + "
              f"{self.gg_k_nodes_kept_pathognomonic} (Pathognomonic Shadow), "
              f"Dropped {self.gg_k_nodes_dropped} (of {self.gg_k_nodes_total} total)")

class Phase2HybridEngine:
    """Initialize Online reasoning from Illness Graphs and add live evidence."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-4o",
        golden_graph_dir: str = "golden_graphs",
        refined_graph_dir: str = "golden_graphs_refined",
        similarity_threshold: float = 0.90,
        fuzzy_threshold: float = 0.80,
        embedding_client=None,
        config_path: Optional[str] = None
    ):

        self.llm_client = llm_client
        self.model_name = model_name
        self.config_path = config_path

        self._load_thresholds_from_config()

        if similarity_threshold != 0.90:
            self.similarity_threshold = similarity_threshold
        if fuzzy_threshold != 0.80:
            self.fuzzy_threshold = fuzzy_threshold

        self.golden_loader = GoldenGraphLoader(
            golden_graph_dir=golden_graph_dir,
            refined_graph_dir=refined_graph_dir
        )

        self.live_engine = Phase2GraphReasoning(llm_client, model_name)

        self.semantic_matcher = SemanticMatcher(
            similarity_threshold=self.similarity_threshold,
            fuzzy_threshold=self.fuzzy_threshold,
            embedding_client=embedding_client
        )

        self._stats = {
            "hybrid_processed": 0,
            "fallback_processed": 0,
            "pruned_p_nodes": 0,
            "pruned_k_nodes": 0,
            "kept_k_nodes_match": 0,
            "kept_k_nodes_pathognomonic": 0,
            "golden_k_nodes_inherited": 0,
            "live_k_nodes_added": 0,
            "pathognomonic_shadows": 0
        }

        print(f"[Phase2 Hybrid] Initialized with:")
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"  - Fuzzy threshold: {self.fuzzy_threshold}")

    def _load_thresholds_from_config(self) -> None:

        self.similarity_threshold = 0.90
        self.fuzzy_threshold = 0.80

        try:
            config_path = self.config_path or os.path.join(
                os.path.dirname(__file__),
                "..", "..", "config", "settings.yaml"
            )

            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                embedding_config = config.get("embedding", {})
                self.similarity_threshold = embedding_config.get(
                    "similarity_threshold", 0.90
                )
                self.fuzzy_threshold = embedding_config.get(
                    "fuzzy_threshold", 0.80
                )
        except Exception as e:
            print(f"[Phase2 Hybrid] Warning: Failed to load config: {e}")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:

        self._reset_stats()

        try:

            phase1_result = state.get("phase1_result")
            if not phase1_result:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Phase 1 result is missing"
                }

            input_case = state.get("input_case", {})
            raw_text = input_case.get("narrative", "")
            if not raw_text:
                return {
                    **state,
                    "status": "failed",
                    "error_log": "Raw text (narrative) is missing"
                }

            print("[Phase2 Hybrid] Step 0: Initializing graph from Phase 1 result...")
            graph = MedicalGraph.from_phase1(phase1_result, raw_text)

            top_candidates = phase1_result.get("top_candidates", [])
            track_b_output = phase1_result.get("track_b_output", {})
            patient_p_nodes = track_b_output.get("p_nodes", [])

            print(f"[Phase2 Hybrid] Patient P-Nodes count: {len(patient_p_nodes)}")

            print("[Phase2 Hybrid] Step 1: Loading Golden Graphs...")
            # Refined graphs take precedence; missing diseases still use live reasoning.
            golden_graphs = self.golden_loader.load_for_candidates(top_candidates)

            d_nodes = graph.get_d_nodes()

            all_candidate_k_nodes: List[Dict[str, Any]] = []

            print("[Phase2 Hybrid] Step 2: Strict Pruning for each candidate...")

            for candidate_id in top_candidates:
                d_id = f"d_{candidate_id}" if not candidate_id.startswith("d_") else candidate_id
                golden = golden_graphs.get(candidate_id)

                if golden is None:

                    print(f"[Phase2 Hybrid] {d_id}: No Golden Graph, will use Live Search only")
                    self._stats["fallback_processed"] += 1
                else:

                    print(f"[Phase2 Hybrid] {d_id}: Processing Golden Graph...")
                    self._stats["hybrid_processed"] += 1

                    filtered_k_nodes, stats = self._instantiate_golden_graph_strict(
                        golden_graph=golden,
                        patient_p_nodes=patient_p_nodes,
                        graph=graph,
                        target_d_id=d_id
                    )

                    self._stats["pruned_p_nodes"] += stats.gg_p_nodes_pruned
                    self._stats["pruned_k_nodes"] += stats.gg_k_nodes_dropped
                    self._stats["kept_k_nodes_match"] += stats.gg_k_nodes_kept_match
                    self._stats["kept_k_nodes_pathognomonic"] += stats.gg_k_nodes_kept_pathognomonic
                    self._stats["pathognomonic_shadows"] += stats.pathognomonic_shadows_created

                    for k in filtered_k_nodes:
                        k["origin"] = "GoldenGraph"
                        k["target_d_id"] = d_id
                        all_candidate_k_nodes.append(k)

                    stats.log_summary(golden.pathology_name)

            # Step 3: Live Search Integration
            print("[Phase2 Hybrid] Step 3: Live Search Integration...")
            live_k_nodes = self._execute_live_search(graph, patient_p_nodes, d_nodes)

            for k in live_k_nodes:
                k["origin"] = "LiveSearch"
                all_candidate_k_nodes.append(k)

            self._stats["live_k_nodes_added"] = len(live_k_nodes)

            print(f"[Phase2 Hybrid] Total candidate K-Nodes: {len(all_candidate_k_nodes)} "
                  f"(GG: {len(all_candidate_k_nodes) - len(live_k_nodes)}, Live: {len(live_k_nodes)})")

            # Step 4: Unified Connection
            print("[Phase2 Hybrid] Step 4: Unified Connection...")
            self._unified_connection(graph, all_candidate_k_nodes, patient_p_nodes, d_nodes)

            print("[Phase2 Hybrid] Step 5: Recall Verification...")
            self.live_engine._step3_recall(graph, raw_text)

            graph_json = graph.to_dict()

            graph_json["hybrid_engine_stats"] = self._stats.copy()

            print(f"[Phase2 Hybrid] Complete. "
                  f"P:{len(graph.get_p_nodes())}, "
                  f"K:{len(graph.get_k_nodes())}, "
                  f"D:{len(graph.get_d_nodes())}")

            return {
                **state,
                "graph_json": graph_json,
                "status": "processing"
            }

        except Exception as e:
            import traceback
            error_msg = f"Phase 2 Hybrid error: {str(e)}\n{traceback.format_exc()}"
            print(f"[Phase2 Hybrid] {error_msg}")
            return {
                **state,
                "status": "failed",
                "error_log": error_msg
            }

    def _instantiate_golden_graph_strict(
        self,
        golden_graph: CanonicalGraph,
        patient_p_nodes: List[Dict[str, Any]],
        graph: MedicalGraph,
        target_d_id: str
    ) -> Tuple[List[Dict[str, Any]], PruningStats]:

        stats = PruningStats()

        gg_p_nodes = [
            {"content": p.content, "status": p.typical_status}
            for p in golden_graph.p_nodes.values()
        ]

        gg_k_nodes = list(golden_graph.k_nodes.values())

        stats.gg_p_nodes_total = len(gg_p_nodes)
        stats.gg_k_nodes_total = len(gg_k_nodes)

        print(f"[Phase2 Hybrid] Matching {len(gg_p_nodes)} GG P-Nodes "
              f"against {len(patient_p_nodes)} Patient P-Nodes...")

        match_results, used_embedding = self.semantic_matcher.match_p_nodes(
            gg_p_nodes, patient_p_nodes
        )

        matched_gg_p_contents: Set[str] = set()
        p_match_map: Dict[str, MatchResult] = {}  # gg_content -> MatchResult

        for result in match_results:
            p_match_map[result.gg_content.lower().strip()] = result
            if result.matched:
                matched_gg_p_contents.add(result.gg_content.lower().strip())
                stats.gg_p_nodes_matched += 1
            else:
                stats.gg_p_nodes_pruned += 1

        k_to_p_edges: Dict[str, List[str]] = {}  # k_content -> [p_content, ...]

        for edge in golden_graph.get_edges_by_type("P-K"):
            p_content_key = edge.source_content.lower().strip()
            k_content_key = edge.target_content.lower().strip()

            if k_content_key not in k_to_p_edges:
                k_to_p_edges[k_content_key] = []
            k_to_p_edges[k_content_key].append(p_content_key)

        filtered_k_nodes: List[Dict[str, Any]] = []

        for k_node in gg_k_nodes:
            k_content = k_node.content
            k_content_key = k_content.lower().strip()
            k_importance = k_node.importance
            is_pathognomonic = (k_importance == "Pathognomonic")

            connected_p_contents = k_to_p_edges.get(k_content_key, [])
            has_matched_p = any(
                p_key in matched_gg_p_contents
                for p_key in connected_p_contents
            )

            if has_matched_p:

                stats.gg_k_nodes_kept_match += 1

                for p_key in connected_p_contents:
                    if p_key in matched_gg_p_contents:
                        match_result = p_match_map.get(p_key)
                        if match_result and match_result.matched:

                            k_id = graph.add_k_node(
                                content=k_content,
                                k_type=k_node.k_type,
                                source="GoldenGraph",
                                importance=k_importance,
                                source_type="GoldenGraph"
                            )

                            p_id = self._find_or_create_p_node(
                                graph,
                                match_result.patient_content,
                                match_result.patient_status
                            )

                            if match_result.patient_status == "Present":
                                relation = "Match"
                            elif match_result.patient_status == "Absent":
                                relation = "Conflict"
                            else:
                                relation = "Void"

                            graph.add_pk_edge(p_id, k_id, relation=relation, provenance="GoldenGraph")

                            graph.add_kd_edge(
                                k_id, target_d_id,
                                relation="Support",
                                strength=k_importance,
                                provenance="GoldenGraph"
                            )

                            filtered_k_nodes.append({
                                "content": k_content,
                                "k_type": k_node.k_type,
                                "importance": k_importance,
                                "k_id": k_id,
                                "matched_p_content": match_result.patient_content,
                                "matched_p_status": match_result.patient_status
                            })

                            break

            elif is_pathognomonic:

                # Keep unmatched critical evidence as a shadow for backward verification.
                stats.gg_k_nodes_kept_pathognomonic += 1
                stats.pathognomonic_shadows_created += 1

                k_id = graph.add_k_node(
                    content=k_content,
                    k_type=k_node.k_type,
                    source="GoldenGraph",
                    importance=k_importance,
                    source_type="GoldenGraph"
                )

                shadow_p_id = graph.create_shadow_p_node(
                    k_content,
                    source_type="GoldenGraph"
                )
                graph.add_pk_edge(shadow_p_id, k_id, relation="Void", provenance="GoldenGraph")

                graph.add_kd_edge(
                    k_id, target_d_id,
                    relation="Support",
                    strength=k_importance,
                    provenance="GoldenGraph"
                )

                filtered_k_nodes.append({
                    "content": k_content,
                    "k_type": k_node.k_type,
                    "importance": k_importance,
                    "k_id": k_id,
                    "is_shadow": True
                })

                print(f"[Phase 2] Pruning: Kept Pathognomonic Shadow K-Node: '{k_content[:50]}...'")

            else:

                stats.gg_k_nodes_dropped += 1

        return filtered_k_nodes, stats

    def _find_or_create_p_node(
        self,
        graph: MedicalGraph,
        content: str,
        status: str
    ) -> str:

        existing = graph.find_p_node_by_content(content)
        if existing:
            return existing["id"]

        return graph.add_p_node(
            content=content,
            status=status,
            source="Phase2_GoldenInstantiation",
            source_type="GoldenGraph"
        )

    def _execute_live_search(
        self,
        graph: MedicalGraph,
        patient_p_nodes: List[Dict[str, Any]],
        d_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        from src.utils.knowledge_utils import (
            get_all_search_terms_for_disease,
            PubMedQueryBuilder,
            ContextBudgetManager
        )
        from src.tools.search_tools import OpenTargetsTool, PubMedTool
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        live_k_nodes = []

        open_targets_results: Dict[str, str] = {}
        pubmed_results: Dict[str, str] = {}

        open_targets_tool = OpenTargetsTool()

        def retrieve_knowledge(d_node: Dict) -> Tuple[str, str, str]:
            d_id = d_node["id"]
            d_name = d_node["name"]

            ot_text = ""
            pm_text = ""

            try:
                search_terms = get_all_search_terms_for_disease(d_id, d_name)
                ot_result, _ = open_targets_tool.get_disease_description_with_raw(
                    d_name, search_terms=search_terms
                )
                if ot_result and len(ot_result) >= 100:
                    ot_text = ot_result

                time.sleep(0.1)
                pm_abstracts, _ = self.live_engine._search_pubmed_waterfall(d_name, d_id)
                if pm_abstracts:
                    pm_text = "\n\n---\n\n".join(pm_abstracts)

            except Exception as e:
                print(f"[Phase2 Hybrid] Live Search error for {d_name}: {e}")

            return d_id, ot_text, pm_text

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(retrieve_knowledge, d): d for d in d_nodes}

            for future in as_completed(futures):
                try:
                    d_id, ot_text, pm_text = future.result()
                    open_targets_results[d_id] = ot_text
                    pubmed_results[d_id] = pm_text
                except Exception as e:
                    print(f"[Phase2 Hybrid] Live Search task error: {e}")

        context_manager = ContextBudgetManager(max_total_tokens=30000)
        knowledge_map = context_manager.allocate_budget(
            candidates=d_nodes,
            open_targets_texts=open_targets_results,
            pubmed_texts=pubmed_results
        )

        # Batch Reasoning
        user_prompt = build_batch_prompt(
            candidates=d_nodes,
            p_nodes=patient_p_nodes,
            knowledge_map=knowledge_map
        )

        messages = [
            {"role": "system", "content": PHASE2_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        result = self.llm_client.generate_json(
            messages=messages,
            model=self.model_name,
            logprobs=False,
            temperature=0.0,
            max_tokens=8192
        )

        if result["error"]:
            print(f"[Phase2 Hybrid] Live Search LLM error: {result['error']}")
            return live_k_nodes

        parsed_json = robust_parse_k_gen_response(result["content"], verbose=False)

        for k_item in parsed_json.get("k_nodes", []):
            live_k_nodes.append({
                "content": k_item.get("content", ""),
                "k_type": k_item.get("type", "General"),
                "importance": k_item.get("importance", "Common"),
                "supported_candidates": k_item.get("supported_candidates", []),
                "ruled_out_candidates": k_item.get("ruled_out_candidates", [])
            })

        print(f"[Phase2 Hybrid] Live Search extracted {len(live_k_nodes)} K-Nodes")

        return live_k_nodes

    def _unified_connection(
        self,
        graph: MedicalGraph,
        candidate_k_nodes: List[Dict[str, Any]],
        patient_p_nodes: List[Dict[str, Any]],
        d_nodes: List[Dict[str, Any]]
    ) -> None:

        valid_d_ids = {d["id"] for d in d_nodes}
        d_name_to_id = {d["name"].lower(): d["id"] for d in d_nodes}

        existing_k_contents = {
            k.get("content", "").lower().strip()
            for k in graph.get_k_nodes()
        }

        for k_item in candidate_k_nodes:
            k_content = k_item.get("content", "")
            k_content_key = k_content.lower().strip()
            origin = k_item.get("origin", "LiveSearch")

            if not k_content:
                continue

            if origin == "GoldenGraph" and k_content_key in existing_k_contents:
                continue

            if origin == "LiveSearch":
                k_id = graph.add_k_node(
                    content=k_content,
                    k_type=k_item.get("k_type", "General"),
                    source="LiveSearch",
                    importance=k_item.get("importance", "Common"),
                    source_type="LiveSearch"
                )

                matched_p = graph.find_p_node_by_content(k_content)

                if matched_p:
                    p_status = matched_p.get("status", "Present")
                    relation = "Match" if p_status == "Present" else "Conflict"
                    graph.add_pk_edge(matched_p["id"], k_id, relation=relation, provenance="LiveInference")
                else:

                    shadow_p_id = graph.create_shadow_p_node(k_content, source_type="LiveSearch")
                    graph.add_pk_edge(shadow_p_id, k_id, relation="Void", provenance="LiveInference")

                for target_id in k_item.get("supported_candidates", []):
                    resolved_id = self._resolve_d_node_id(target_id, valid_d_ids, d_name_to_id)
                    if resolved_id:
                        graph.add_kd_edge(
                            k_id, resolved_id,
                            relation="Support",
                            strength=k_item.get("importance", "Common"),
                            provenance="LiveInference"
                        )

                for target_id in k_item.get("ruled_out_candidates", []):
                    resolved_id = self._resolve_d_node_id(target_id, valid_d_ids, d_name_to_id)
                    if resolved_id:
                        graph.add_kd_edge(
                            k_id, resolved_id,
                            relation="Rule_Out",
                            strength=k_item.get("importance", "Common"),
                            provenance="LiveInference"
                        )

    def _resolve_d_node_id(
        self,
        target: str,
        valid_d_ids: Set[str],
        d_name_to_id: Dict[str, str]
    ) -> Optional[str]:

        if not target:
            return None

        if target in valid_d_ids:
            return target

        if f"d_{target}" in valid_d_ids:
            return f"d_{target}"

        target_lower = target.lower()
        for name, d_id in d_name_to_id.items():
            if name in target_lower or target_lower in name:
                return d_id

        return None

    def _reset_stats(self) -> None:

        for key in self._stats:
            self._stats[key] = 0

    def get_statistics(self) -> Dict[str, Any]:

        return self._stats.copy()

    def reset_statistics(self) -> None:

        self._reset_stats()

if __name__ == "__main__":
    print("Phase2HybridEngine module loaded successfully (v2.0 - Strict Pruning)")
    print("Use Phase2HybridEngine.process(state) for Online inference")
