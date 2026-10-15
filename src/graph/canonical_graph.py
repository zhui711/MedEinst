"""Disease-specific Illness Graph representation and serialization."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CanonicalPNode:
    """Patient observation in a disease Illness Graph."""
    content: str
    occurrence_count: int = 1
    typical_status: str = "Present"
    example_original_texts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "content": self.content,
            "occurrence_count": self.occurrence_count,
            "typical_status": self.typical_status,
            "example_original_texts": self.example_original_texts[:3]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalPNode":

        return cls(
            content=data.get("content", ""),
            occurrence_count=data.get("occurrence_count", 1),
            typical_status=data.get("typical_status", "Present"),
            example_original_texts=data.get("example_original_texts", [])
        )

@dataclass
class CanonicalKNode:
    """Disease knowledge in an Illness Graph."""
    content: str
    k_type: str = "General"
    importance: str = "Common"
    occurrence_count: int = 1
    original_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "content": self.content,
            "k_type": self.k_type,
            "importance": self.importance,
            "occurrence_count": self.occurrence_count,
            "original_sources": list(set(self.original_sources))[:5]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalKNode":

        return cls(
            content=data.get("content", ""),
            k_type=data.get("k_type", "General"),
            importance=data.get("importance", "Common"),
            occurrence_count=data.get("occurrence_count", 1),
            original_sources=data.get("original_sources", [])
        )

@dataclass
class CanonicalEdge:
    """Relation in an Illness Graph."""
    source_content: str
    target_content: str
    edge_type: str  # "P-K" or "K-D"
    relation: str
    strength: Optional[str] = None
    occurrence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:

        result = {
            "source_content": self.source_content,
            "target_content": self.target_content,
            "edge_type": self.edge_type,
            "relation": self.relation,
            "occurrence_count": self.occurrence_count
        }
        if self.strength:
            result["strength"] = self.strength
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEdge":

        return cls(
            source_content=data.get("source_content", ""),
            target_content=data.get("target_content", ""),
            edge_type=data.get("edge_type", "P-K"),
            relation=data.get("relation", ""),
            strength=data.get("strength"),
            occurrence_count=data.get("occurrence_count", 1)
        )

class CanonicalGraph:
    """Aggregated disease-specific Illness Graph."""

    def __init__(
        self,
        pathology_id: str,
        pathology_name: str
    ):

        self.pathology_id = pathology_id
        self.pathology_name = pathology_name

        self.p_nodes: Dict[str, CanonicalPNode] = {}
        self.k_nodes: Dict[str, CanonicalKNode] = {}

        self._edges: Dict[Tuple[str, str, str], CanonicalEdge] = {}

        self.source_cases: List[str] = []
        self.version: str = "1.0"

    def add_p_node(
        self,
        content: str,
        status: str = "Present",
        original_text: str = ""
    ) -> None:

        content_key = content.lower().strip()

        if content_key in self.p_nodes:

            existing = self.p_nodes[content_key]
            existing.occurrence_count += 1
            if original_text and original_text not in existing.example_original_texts:
                existing.example_original_texts.append(original_text)
        else:

            self.p_nodes[content_key] = CanonicalPNode(
                content=content,
                occurrence_count=1,
                typical_status=status,
                example_original_texts=[original_text] if original_text else []
            )

    def add_k_node(
        self,
        content: str,
        k_type: str = "General",
        importance: str = "Common",
        source: str = ""
    ) -> None:

        content_key = content.lower().strip()

        importance_order = {
            "Pathognomonic": 4,
            "Essential": 3,
            "Strong": 2,
            "Common": 1,
            "Weak": 0
        }

        if content_key in self.k_nodes:

            existing = self.k_nodes[content_key]
            existing.occurrence_count += 1

            if importance_order.get(importance, 0) > importance_order.get(existing.importance, 0):
                existing.importance = importance

            if source and source not in existing.original_sources:
                existing.original_sources.append(source)
        else:

            self.k_nodes[content_key] = CanonicalKNode(
                content=content,
                k_type=k_type,
                importance=importance,
                occurrence_count=1,
                original_sources=[source] if source else []
            )

    def add_edge(
        self,
        source_content: str,
        target_content: str,
        edge_type: str,
        relation: str,
        strength: Optional[str] = None
    ) -> None:

        edge_key = (
            source_content.lower().strip(),
            target_content.lower().strip(),
            relation
        )

        if edge_key in self._edges:

            self._edges[edge_key].occurrence_count += 1
        else:

            self._edges[edge_key] = CanonicalEdge(
                source_content=source_content,
                target_content=target_content,
                edge_type=edge_type,
                relation=relation,
                strength=strength,
                occurrence_count=1
            )

    def add_source_case(self, case_id: str) -> None:

        if case_id not in self.source_cases:
            self.source_cases.append(case_id)

    def aggregate_from_medical_graph(
        self,
        graph_dict: Dict[str, Any],
        case_id: str = ""
    ) -> None:

        if case_id:
            self.add_source_case(case_id)

        graph_data = graph_dict.get("graph", {})
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", {})

        for p_node in nodes.get("p_nodes", []):
            self.add_p_node(
                content=p_node.get("content", ""),
                status=p_node.get("status", "Present"),
                original_text=p_node.get("original_text", "")
            )

        for k_node in nodes.get("k_nodes", []):
            self.add_k_node(
                content=k_node.get("content", ""),
                k_type=k_node.get("k_type", "General"),
                importance=k_node.get("importance", "Common"),
                source=k_node.get("source", "")
            )

        id_to_content: Dict[str, str] = {}

        for p_node in nodes.get("p_nodes", []):
            node_id = p_node.get("id", "")
            if node_id:
                id_to_content[node_id] = p_node.get("content", "")

        for k_node in nodes.get("k_nodes", []):
            node_id = k_node.get("id", "")
            if node_id:
                id_to_content[node_id] = k_node.get("content", "")

        for d_node in nodes.get("d_nodes", []):
            node_id = d_node.get("id", "")
            if node_id:
                id_to_content[node_id] = d_node.get("name", "")

        for edge in edges.get("p_k_links", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            relation = edge.get("relation", "")

            source_content = id_to_content.get(source_id, source_id)
            target_content = id_to_content.get(target_id, target_id)

            if source_content and target_content:
                self.add_edge(
                    source_content=source_content,
                    target_content=target_content,
                    edge_type="P-K",
                    relation=relation
                )

        for edge in edges.get("k_d_links", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            relation = edge.get("relation", "")
            strength = edge.get("strength", "Weak")

            source_content = id_to_content.get(source_id, source_id)
            target_content = id_to_content.get(target_id, target_id)

            target_disease_id = target_id.replace("d_", "") if target_id.startswith("d_") else ""

            if target_disease_id == self.pathology_id:
                if source_content and target_content:
                    self.add_edge(
                        source_content=source_content,
                        target_content=target_content,
                        edge_type="K-D",
                        relation=relation,
                        strength=strength
                    )

    def get_essential_k_nodes(self) -> List[CanonicalKNode]:

        return [
            node for node in self.k_nodes.values()
            if node.importance in ["Essential", "Pathognomonic"]
        ]

    def get_pivot_k_nodes(self) -> List[CanonicalKNode]:

        return [
            node for node in self.k_nodes.values()
            if node.k_type == "Pivot"
        ]

    def get_common_p_nodes(self, min_occurrence: int = 2) -> List[CanonicalPNode]:

        return [
            node for node in self.p_nodes.values()
            if node.occurrence_count >= min_occurrence
        ]

    def get_edges_by_type(self, edge_type: str) -> List[CanonicalEdge]:

        return [
            edge for edge in self._edges.values()
            if edge.edge_type == edge_type
        ]

    @property
    def edges(self) -> List[CanonicalEdge]:

        return list(self._edges.values())

    def get_statistics(self) -> Dict[str, Any]:

        return {
            "pathology_id": self.pathology_id,
            "pathology_name": self.pathology_name,
            "total_p_nodes": len(self.p_nodes),
            "total_k_nodes": len(self.k_nodes),
            "total_edges": len(self._edges),
            "source_cases_count": len(self.source_cases),
            "essential_k_nodes_count": len(self.get_essential_k_nodes()),
            "pivot_k_nodes_count": len(self.get_pivot_k_nodes())
        }

    def to_dict(self) -> Dict[str, Any]:

        return {
            "metadata": {
                "pathology_id": self.pathology_id,
                "pathology_name": self.pathology_name,
                "version": self.version,
                "source_cases_count": len(self.source_cases),
                "source_cases": self.source_cases
            },
            "statistics": self.get_statistics(),
            "graph": {
                "p_nodes": [node.to_dict() for node in self.p_nodes.values()],
                "k_nodes": [node.to_dict() for node in self.k_nodes.values()],
                "edges": [edge.to_dict() for edge in self._edges.values()]
            }
        }

    def save(self, filepath: Path) -> None:

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[CanonicalGraph] Saved {self.pathology_name} to {filepath}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalGraph":

        metadata = data.get("metadata", {})

        graph = cls(
            pathology_id=metadata.get("pathology_id", ""),
            pathology_name=metadata.get("pathology_name", "")
        )
        graph.version = metadata.get("version", "1.0")
        graph.source_cases = metadata.get("source_cases", [])

        graph_data = data.get("graph", {})

        for p_data in graph_data.get("p_nodes", []):
            node = CanonicalPNode.from_dict(p_data)
            graph.p_nodes[node.content.lower().strip()] = node

        for k_data in graph_data.get("k_nodes", []):
            node = CanonicalKNode.from_dict(k_data)
            graph.k_nodes[node.content.lower().strip()] = node

        for e_data in graph_data.get("edges", []):
            edge = CanonicalEdge.from_dict(e_data)
            edge_key = (
                edge.source_content.lower().strip(),
                edge.target_content.lower().strip(),
                edge.relation
            )
            graph._edges[edge_key] = edge

        return graph

    @classmethod
    def load(cls, filepath: Path) -> "CanonicalGraph":

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @staticmethod
    def get_canonical_filename(pathology_name: str) -> str:

        safe_name = pathology_name.replace(" ", "_").replace("/", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        return f"{safe_name}_GG"

    def __repr__(self) -> str:

        return (
            f"CanonicalGraph("
            f"id={self.pathology_id}, "
            f"name='{self.pathology_name}', "
            f"p_nodes={len(self.p_nodes)}, "
            f"k_nodes={len(self.k_nodes)}, "
            f"edges={len(self._edges)}, "
            f"cases={len(self.source_cases)}"
            f")"
        )

