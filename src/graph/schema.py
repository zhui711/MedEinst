"""Case-level causal reasoning graph and node schema."""
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Literal
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP


SourceType = Literal["GoldenGraph", "LiveSearch"]


EdgeProvenance = Literal["GoldenGraph", "LiveInference"]


class MedicalGraph:
    """Directed case-level graph of observations, evidence, and diagnoses."""

    def __init__(self):

        self.graph = nx.DiGraph()
        self.meta = {}

        self._p_counter = 0
        self._k_counter = 0
        self._d_counter = 0

    # ==================== Node Operations ====================

    def add_p_node(
        self,
        node_id: Optional[str] = None,
        content: str = "",
        original_text: str = "",
        status: str = "Present",  # Present, Absent, Missing
        source: str = "Phase1",
        source_type: SourceType = "LiveSearch"
    ) -> str:

        # Patient observations retain Present, Absent, and Missing states.
        if node_id is None:
            self._p_counter += 1
            node_id = f"p_{self._p_counter}"

        self.graph.add_node(
            node_id,
            type="P",
            content=content,
            original_text=original_text,
            status=status,
            source=source,
            source_type=source_type
        )
        return node_id

    def add_k_node(
        self,
        node_id: Optional[str] = None,
        content: str = "",
        k_type: str = "General",  # General, Pivot
        source: str = "Wiki_API",  # Wiki_API, PubMed_API
        importance: str = "Common",  # Essential, Pathognomonic, Common
        source_type: SourceType = "LiveSearch"
    ) -> str:

        # General nodes support a hypothesis; Pivot nodes distinguish candidates.
        if node_id is None:
            self._k_counter += 1
            node_id = f"k_{self._k_counter}"

        self.graph.add_node(
            node_id,
            type="K",
            content=content,
            k_type=k_type,
            source=source,
            importance=importance,
            source_type=source_type
        )
        return node_id

    def add_d_node(
        self,
        node_id: Optional[str] = None,
        name: str = "",
        original_id: str = "",
        initial_rank: int = 1,
        risk_level: Optional[str] = None,
        state: str = "Active"  # Active, Pruned
    ) -> str:

        if node_id is None:
            if original_id:
                node_id = f"d_{original_id}"
            else:
                self._d_counter += 1
                node_id = f"d_{self._d_counter}"

        self.graph.add_node(
            node_id,
            type="D",
            name=name,
            original_id=original_id,
            initial_rank=initial_rank,
            risk_level=risk_level,
            state=state
        )
        return node_id

    # ==================== Edge Operations ====================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        provenance: EdgeProvenance = "LiveInference",
        **kwargs
    ):

        edge_attrs = {
            "relation": relation,
            "provenance": provenance
        }
        edge_attrs.update(kwargs)
        self.graph.add_edge(source_id, target_id, **edge_attrs)

    def add_pk_edge(
        self,
        p_id: str,
        k_id: str,
        relation: str,  # Match, Conflict, Void
        provenance: EdgeProvenance = "LiveInference"
    ):

        self.add_edge(p_id, k_id, relation=relation, provenance=provenance)

    def add_kd_edge(
        self,
        k_id: str,
        d_id: str,
        relation: str,  # Support, Rule_Out
        strength: str = "Weak",  # Pathognomonic, Essential, Strong, Weak
        provenance: EdgeProvenance = "LiveInference"
    ):

        self.add_edge(k_id, d_id, relation=relation, strength=strength, provenance=provenance)

    def update_edge(
        self,
        source_id: str,
        target_id: str,
        **kwargs
    ):

        if self.graph.has_edge(source_id, target_id):
            for key, value in kwargs.items():
                self.graph[source_id][target_id][key] = value

    def remove_edge(self, source_id: str, target_id: str):

        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)

    # ==================== Node State Operations ====================

    def update_p_node_status(self, p_id: str, new_status: str, source: str = None):

        if self.graph.has_node(p_id):
            self.graph.nodes[p_id]["status"] = new_status
            if source:
                self.graph.nodes[p_id]["source"] = source

    def prune_d_node(self, d_id: str):

        if self.graph.has_node(d_id):
            self.graph.nodes[d_id]["state"] = "Pruned"

    def set_d_node_safeguard(self, d_id: str, safeguard: bool = True):

        if self.graph.has_node(d_id):
            self.graph.nodes[d_id]["safeguard"] = safeguard

    # ==================== Query Operations ====================

    def get_p_nodes(self) -> List[Dict[str, Any]]:

        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "P"
        ]

    def get_k_nodes(self) -> List[Dict[str, Any]]:

        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "K"
        ]

    def get_d_nodes(self) -> List[Dict[str, Any]]:

        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "D"
        ]

    def get_active_d_nodes(self) -> List[Dict[str, Any]]:

        return [
            {**data, "id": node_id}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "D" and data.get("state") != "Pruned"
        ]

    def get_p_node_by_id(self, p_id: str) -> Optional[Dict[str, Any]]:

        if self.graph.has_node(p_id):
            data = self.graph.nodes[p_id]
            if data.get("type") == "P":
                return {**data, "id": p_id}
        return None

    def get_k_node_by_id(self, k_id: str) -> Optional[Dict[str, Any]]:

        if self.graph.has_node(k_id):
            data = self.graph.nodes[k_id]
            if data.get("type") == "K":
                return {**data, "id": k_id}
        return None

    def get_d_node_by_id(self, d_id: str) -> Optional[Dict[str, Any]]:

        if self.graph.has_node(d_id):
            data = self.graph.nodes[d_id]
            if data.get("type") == "D":
                return {**data, "id": d_id}
        return None

    def find_p_node_by_content(self, content: str) -> Optional[Dict[str, Any]]:

        content_lower = content.lower().strip()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "P":
                p_content = data.get("content", "").lower().strip()

                if content_lower == p_content or content_lower in p_content or p_content in content_lower:
                    return {**data, "id": node_id}
        return None

    def find_k_node_by_content(self, content: str) -> Optional[Dict[str, Any]]:

        content_lower = content.lower().strip()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "K":
                k_content = data.get("content", "").lower().strip()

                if content_lower == k_content or content_lower in k_content or k_content in content_lower:
                    return {**data, "id": node_id}
        return None

    def get_void_k_nodes(self) -> List[Dict[str, Any]]:

        void_k_nodes = []

        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") == "Void":
                source_data = self.graph.nodes.get(source, {})
                target_data = self.graph.nodes.get(target, {})

                if source_data.get("type") == "P" and target_data.get("type") == "K":
                    k_node = {**target_data, "id": target}

                    k_node["shadow_p_id"] = source
                    if k_node not in void_k_nodes:
                        void_k_nodes.append(k_node)

        return void_k_nodes

    def get_k_nodes_for_d(self, d_id: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:

        results = []
        for source, target, data in self.graph.edges(data=True):
            if target == d_id:
                source_data = self.graph.nodes.get(source, {})
                if source_data.get("type") == "K":
                    k_node = {**source_data, "id": source}
                    edge_info = {**data}
                    results.append((k_node, edge_info))
        return results

    def get_p_nodes_for_k(self, k_id: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:

        results = []
        for source, target, data in self.graph.edges(data=True):
            if target == k_id:
                source_data = self.graph.nodes.get(source, {})
                if source_data.get("type") == "P":
                    p_node = {**source_data, "id": source}
                    edge_info = {**data}
                    results.append((p_node, edge_info))
        return results

    def has_fatal_conflict(self, d_id: str) -> bool:

        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)

        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            importance = k_node.get("importance", "Common")

            if importance in ["Essential", "Pathognomonic"]:

                p_nodes_with_edges = self.get_p_nodes_for_k(k_id)
                for p_node, pk_edge in p_nodes_with_edges:
                    if pk_edge.get("relation") == "Conflict":
                        return True

        return False

    def count_shadow_nodes(self, d_id: str) -> int:

        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)
        shadow_count = 0

        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            p_nodes_with_edges = self.get_p_nodes_for_k(k_id)
            for p_node, pk_edge in p_nodes_with_edges:
                if pk_edge.get("relation") == "Void":
                    shadow_count += 1

        return shadow_count

    def count_total_evidence(self, d_id: str) -> int:

        return len(self.get_k_nodes_for_d(d_id))

    def calculate_explanatory_coverage(self, d_id: str) -> float:

        k_nodes_with_edges = self.get_k_nodes_for_d(d_id)

        match_count = 0
        void_count = 0
        conflict_count = 0

        for k_node, kd_edge in k_nodes_with_edges:
            k_id = k_node["id"]
            p_nodes_with_edges = self.get_p_nodes_for_k(k_id)

            for p_node, pk_edge in p_nodes_with_edges:
                relation = pk_edge.get("relation", "")
                if relation == "Match":
                    match_count += 1
                elif relation == "Void":
                    void_count += 1
                elif relation == "Conflict":
                    conflict_count += 1

        coverage = match_count - (void_count * 0.5) - conflict_count
        return coverage

    # ==================== Serialization ====================

    def to_dict(self) -> Dict[str, Any]:

        p_nodes = []
        for node in self.get_p_nodes():

            if "source_type" not in node:
                node["source_type"] = "LiveSearch"
            p_nodes.append(node)

        k_nodes = []
        for node in self.get_k_nodes():

            if "source_type" not in node:
                node["source_type"] = "LiveSearch"
            k_nodes.append(node)

        d_nodes = self.get_d_nodes()

        p_k_links = []
        k_d_links = []

        for source, target, data in self.graph.edges(data=True):
            source_data = self.graph.nodes.get(source, {})
            target_data = self.graph.nodes.get(target, {})

            edge_info = {
                "source": source,
                "target": target,
                "relation": data.get("relation", "")
            }

            edge_info["provenance"] = data.get("provenance", "LiveInference")

            if source_data.get("type") == "P" and target_data.get("type") == "K":
                p_k_links.append(edge_info)
            elif source_data.get("type") == "K" and target_data.get("type") == "D":
                edge_info["strength"] = data.get("strength", "Weak")
                k_d_links.append(edge_info)

        return {
            "case_metadata": {
                "raw_text": self.meta.get("raw_text", ""),
                "case_id": self.meta.get("case_id", "")
            },
            "phase1_context": {
                "initial_candidates": self.meta.get("initial_candidates", []),
                "initial_reasoning": self.meta.get("initial_reasoning", "")
            },
            "graph": {
                "nodes": {
                    "p_nodes": p_nodes,
                    "k_nodes": k_nodes,
                    "d_nodes": d_nodes
                },
                "edges": {
                    "p_k_links": p_k_links,
                    "k_d_links": k_d_links
                }
            }
        }

    # ==================== Factory Method ====================

    @classmethod
    def from_phase1(cls, phase1_result: Dict[str, Any], raw_text: str) -> "MedicalGraph":

        graph = cls()

        graph.meta["raw_text"] = raw_text
        graph.meta["initial_candidates"] = phase1_result.get("top_candidates", [])
        graph.meta["initial_reasoning"] = phase1_result.get("differential_reasoning", "")

        top_candidates = phase1_result.get("top_candidates", [])
        for rank, d_id in enumerate(top_candidates):
            d_name = DIAGNOSIS_ID_MAP.get(d_id, "Unknown")
            graph.add_d_node(
                node_id=f"d_{d_id}",
                name=d_name,
                original_id=d_id,
                initial_rank=rank + 1
            )

        track_b_output = phase1_result.get("track_b_output", {})
        p_nodes_data = track_b_output.get("p_nodes", [])

        if p_nodes_data:
            print(f"[MedicalGraph] Initializing {len(p_nodes_data)} P-Nodes from Track B")
            for p_data in p_nodes_data:
                if not isinstance(p_data, dict):
                    continue

                p_id = p_data.get("id", None)
                content = p_data.get("content", "")
                original_text = p_data.get("original_text", "")
                status = p_data.get("status", "Present")

                if not content:
                    continue

                graph.add_p_node(
                    node_id=p_id,
                    content=content,
                    original_text=original_text,
                    status=status,
                    source="Phase1_TrackB"
                )
        else:

            print("[MedicalGraph] WARNING: No P-Nodes found in Track B output, trying structured_analysis fallback...")
            structured_analysis = phase1_result.get("structured_analysis", {})

            if structured_analysis:

                positive_findings = structured_analysis.get("positive_findings", [])
                for finding in positive_findings:
                    if isinstance(finding, dict):
                        content = finding.get("finding", "")
                        status = finding.get("status", "Present")
                    elif isinstance(finding, str):
                        content = finding.strip()
                        status = "Present"
                    else:
                        continue

                    if content:
                        graph.add_p_node(
                            content=content,
                            status=status,
                            source="Phase1_Fallback"
                        )

                negative_findings = structured_analysis.get("negative_findings", [])
                for finding in negative_findings:
                    if isinstance(finding, dict):
                        content = finding.get("finding", "")
                        status = "Absent"
                    elif isinstance(finding, str):
                        content = finding.strip()
                        if content.lower().startswith("no "):
                            content = content[3:].strip()
                        status = "Absent"
                    else:
                        continue

                    if content:
                        graph.add_p_node(
                            content=content,
                            status=status,
                            source="Phase1_Fallback"
                        )

        print(f"[MedicalGraph] Initialized with {len(graph.get_d_nodes())} D-Nodes, {len(graph.get_p_nodes())} P-Nodes")

        return graph

    def create_shadow_p_node(
        self,
        k_content: str,
        source_type: SourceType = "LiveSearch"
    ) -> str:

        return self.add_p_node(
            content=k_content,
            status="Missing",
            source="Phase2_Shadow",
            source_type=source_type
        )

    # ==================== Cascade Operations (Phase 2 Strict Pruning) ====================

    def cascade_remove_node(self, node_id: str) -> int:

        if not self.graph.has_node(node_id):
            return 0

        edges_removed = 0

        in_edges = list(self.graph.in_edges(node_id))
        out_edges = list(self.graph.out_edges(node_id))

        edges_removed = len(in_edges) + len(out_edges)

        self.graph.remove_node(node_id)

        return edges_removed

    def get_k_nodes_connected_to_p(self, p_id: str) -> List[str]:

        k_ids = []

        for source, target, data in self.graph.out_edges(p_id, data=True):
            target_data = self.graph.nodes.get(target, {})
            if target_data.get("type") == "K":
                k_ids.append(target)

        return k_ids

    def get_p_nodes_connected_to_k(self, k_id: str) -> List[str]:

        p_ids = []

        for source, target, data in self.graph.in_edges(k_id, data=True):
            source_data = self.graph.nodes.get(source, {})
            if source_data.get("type") == "P":
                p_ids.append(source)

        return p_ids

    def is_k_node_orphan(self, k_id: str) -> bool:

        return len(self.get_p_nodes_connected_to_k(k_id)) == 0

    def get_k_node_importance(self, k_id: str) -> str:

        if not self.graph.has_node(k_id):
            return "Common"

        return self.graph.nodes[k_id].get("importance", "Common")

    def batch_remove_nodes(self, node_ids: List[str]) -> int:

        total_edges = 0

        for node_id in node_ids:
            edges = self.cascade_remove_node(node_id)
            total_edges += edges

        return total_edges
