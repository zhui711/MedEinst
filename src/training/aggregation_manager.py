"""Merge successful case graphs into disease Illness Graphs."""

import re
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from src.graph.canonical_graph import CanonicalGraph


class AggregationManager:
    """Merge validated case evidence into Illness Graphs."""

    def __init__(
        self,
        aggregation_step: int = 5,
        output_dir: str = "golden_graphs"
    ):

        self.aggregation_step = aggregation_step
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.graphs: Dict[str, CanonicalGraph] = {}

        self.success_counters: Dict[str, int] = defaultdict(int)

        self.total_aggregated = 0
        self.total_saved = 0

    def record_success(
        self,
        pathology_id: str,
        pathology_name: str,
        case_graph: Dict[str, Any],
        case_id: str
    ) -> None:

        pathology_id = pathology_id.replace("d_", "")

        if pathology_id not in self.graphs:

            existing_graph = self._try_load_existing(pathology_id)
            if existing_graph:
                self.graphs[pathology_id] = existing_graph
                print(f"[Aggregation] Loaded existing Golden Graph for {pathology_name}")
            else:
                self.graphs[pathology_id] = CanonicalGraph(
                    pathology_id=pathology_id,
                    pathology_name=pathology_name
                )
                print(f"[Aggregation] Created new Golden Graph for {pathology_name}")

        graph = self.graphs[pathology_id]
        graph.aggregate_from_medical_graph(case_graph, case_id)

        self.total_aggregated += 1
        self.success_counters[pathology_id] += 1

        print(f"[Aggregation] Aggregated {case_id} to {pathology_name} "
              f"(count: {self.success_counters[pathology_id]}/{self.aggregation_step})")

        if self.success_counters[pathology_id] >= self.aggregation_step:
            self._save_graph(pathology_id)
            self.success_counters[pathology_id] = 0

    def _try_load_existing(self, pathology_id: str) -> Optional[CanonicalGraph]:

        pattern = f"{pathology_id}_*_GG.json"
        matches = list(self.output_dir.glob(pattern))

        if matches:
            try:
                return CanonicalGraph.load(matches[0])
            except Exception as e:
                print(f"[Aggregation] Failed to load existing graph: {e}")

        return None

    def _save_graph(self, pathology_id: str) -> None:

        if pathology_id not in self.graphs:
            return

        graph = self.graphs[pathology_id]

        filename = f"{pathology_id}_{self._sanitize_name(graph.pathology_name)}_GG.json"
        filepath = self.output_dir / filename

        graph.save(filepath)
        self.total_saved += 1

        stats = graph.get_statistics()
        print(f"[Aggregation] ✓ Saved {filename}")
        print(f"    P-Nodes: {stats['total_p_nodes']}, "
              f"K-Nodes: {stats['total_k_nodes']}, "
              f"Edges: {stats['total_edges']}, "
              f"Cases: {stats['source_cases_count']}")

    def finalize(self) -> Dict[str, Any]:

        print(f"\n[Aggregation] Finalizing... ({len(self.graphs)} graphs to save)")

        for pathology_id in self.graphs:
            self._save_graph(pathology_id)

        self.success_counters = defaultdict(int)

        stats = self.get_statistics()
        print(f"[Aggregation] Finalization complete. {stats}")

        return stats

    def get_statistics(self) -> Dict[str, Any]:

        graph_stats = {}
        for pid, graph in self.graphs.items():
            graph_stats[pid] = graph.get_statistics()

        return {
            "total_aggregated": self.total_aggregated,
            "total_saved": self.total_saved,
            "unique_pathologies": len(self.graphs),
            "output_dir": str(self.output_dir),
            "graphs": graph_stats
        }

    @staticmethod
    def _sanitize_name(name: str) -> str:

        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', safe_name)
        return safe_name

if __name__ == "__main__":

    manager = AggregationManager(aggregation_step=2, output_dir="golden_graphs_test")

    mock_graph = {
        "graph": {
            "nodes": {
                "p_nodes": [
                    {"id": "p_1", "content": "Fever", "status": "Present"},
                    {"id": "p_2", "content": "Cough", "status": "Present"}
                ],
                "k_nodes": [
                    {"id": "k_1", "content": "High temperature indicates infection",
                     "k_type": "General", "importance": "Essential", "source": "Test"}
                ],
                "d_nodes": [
                    {"id": "d_45", "name": "Pneumonia"}
                ]
            },
            "edges": {
                "p_k_links": [
                    {"source": "p_1", "target": "k_1", "relation": "Match"}
                ],
                "k_d_links": [
                    {"source": "k_1", "target": "d_45", "relation": "Support", "strength": "Essential"}
                ]
            }
        }
    }

    manager.record_success("45", "Pneumonia", mock_graph, "test_case_001")
    manager.record_success("45", "Pneumonia", mock_graph, "test_case_002")
    manager.record_success("23", "Bronchitis", mock_graph, "test_case_003")

    stats = manager.finalize()
    print("\nFinal Statistics:", stats)

