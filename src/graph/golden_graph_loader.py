"""Load raw and refined Illness Graphs by disease ID."""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .canonical_graph import CanonicalGraph


class GoldenGraphLoader:
    """Find refined or raw Illness Graphs for candidate diseases."""

    def __init__(
        self,
        golden_graph_dir: str = "golden_graphs",
        refined_graph_dir: str = "golden_graphs_refined"
    ):

        self.golden_graph_dir = Path(golden_graph_dir)
        self.refined_graph_dir = Path(refined_graph_dir)
        self._cache: Dict[str, Optional[CanonicalGraph]] = {}
        self._lock = threading.Lock()

        self._available_ids = self._scan_available_graphs()

        refined_count = sum(1 for info in self._available_ids.values() if info[1])
        raw_count = sum(1 for info in self._available_ids.values() if not info[1])

        print(f"[GoldenGraphLoader] Initialized with {len(self._available_ids)} available graphs")
        print(f"  → Refined: {refined_count}, Raw: {raw_count}")

    def _scan_available_graphs(self) -> Dict[str, Tuple[Path, bool]]:

        available = {}

        if self.golden_graph_dir.exists():
            for filepath in self.golden_graph_dir.glob("*_GG.json"):
                filename = filepath.stem
                parts = filename.split("_")

                if len(parts) >= 2:
                    disease_id = parts[0]
                    if disease_id.isdigit():

                        available[disease_id] = (filepath, False)
        else:
            print(f"[GoldenGraphLoader] Warning: Raw directory not found: {self.golden_graph_dir}")

        if self.refined_graph_dir.exists():
            for filepath in self.refined_graph_dir.glob("*_Refined_GG.json"):
                filename = filepath.stem
                parts = filename.split("_")

                if len(parts) >= 2:
                    disease_id = parts[0]
                    if disease_id.isdigit():

                        available[disease_id] = (filepath, True)
        else:
            print(f"[GoldenGraphLoader] Info: Refined directory not found: {self.refined_graph_dir}")
            print(f"[GoldenGraphLoader] Will load raw Golden Graphs only.")

        return available

    def load_for_candidates(
        self,
        candidate_ids: List[str]
    ) -> Dict[str, Optional[CanonicalGraph]]:

        result = {}

        for disease_id in candidate_ids:

            clean_id = disease_id.replace("d_", "")

            with self._lock:
                if clean_id in self._cache:
                    result[disease_id] = self._cache[clean_id]
                    continue

            graph = self._load_single(clean_id)

            with self._lock:
                self._cache[clean_id] = graph

            result[disease_id] = graph

        loaded = sum(1 for g in result.values() if g is not None)
        print(f"[GoldenGraphLoader] Loaded {loaded}/{len(candidate_ids)} Golden Graphs")

        return result

    def _load_single(self, disease_id: str) -> Optional[CanonicalGraph]:

        if disease_id not in self._available_ids:
            print(f"[GoldenGraphLoader] No Golden Graph found for disease {disease_id}")
            return None

        filepath, is_refined = self._available_ids[disease_id]
        version_tag = "Refined" if is_refined else "Raw"

        try:
            graph = CanonicalGraph.load(filepath)

            print(f"[GoldenGraphLoader] ✓ Loaded [{version_tag}] {filepath.name} "
                  f"(P:{len(graph.p_nodes)}, K:{len(graph.k_nodes)})")

            return graph

        except Exception as e:
            print(f"[GoldenGraphLoader] ✗ Error loading {filepath}: {e}")
            return None

    def get_available_diseases(self) -> List[str]:

        return list(self._available_ids.keys())

    def is_available(self, disease_id: str) -> bool:

        clean_id = disease_id.replace("d_", "")
        return clean_id in self._available_ids

    def is_refined(self, disease_id: str) -> bool:

        clean_id = disease_id.replace("d_", "")
        if clean_id not in self._available_ids:
            return False
        return self._available_ids[clean_id][1]

    def get_graph_info(self, disease_id: str) -> Optional[Dict[str, any]]:

        clean_id = disease_id.replace("d_", "")
        if clean_id not in self._available_ids:
            return None

        filepath, is_refined = self._available_ids[clean_id]
        return {
            "disease_id": clean_id,
            "filepath": str(filepath),
            "is_refined": is_refined,
            "version": "Refined" if is_refined else "Raw"
        }

    def clear_cache(self) -> None:

        with self._lock:
            self._cache.clear()
        print("[GoldenGraphLoader] Cache cleared")

    def refresh(self) -> None:

        with self._lock:
            self._cache.clear()

        self._available_ids = self._scan_available_graphs()

        refined_count = sum(1 for info in self._available_ids.values() if info[1])
        raw_count = sum(1 for info in self._available_ids.values() if not info[1])

        print(f"[GoldenGraphLoader] Refreshed: {len(self._available_ids)} graphs available")
        print(f"  → Refined: {refined_count}, Raw: {raw_count}")

    def get_cache_statistics(self) -> Dict[str, int]:

        with self._lock:
            total_cached = len(self._cache)
            valid_cached = sum(1 for g in self._cache.values() if g is not None)

        refined_available = sum(1 for info in self._available_ids.values() if info[1])
        raw_available = sum(1 for info in self._available_ids.values() if not info[1])

        return {
            "total_available": len(self._available_ids),
            "refined_available": refined_available,
            "raw_available": raw_available,
            "total_cached": total_cached,
            "valid_cached": valid_cached
        }

if __name__ == "__main__":

    loader = GoldenGraphLoader(
        golden_graph_dir="golden_graphs",
        refined_graph_dir="golden_graphs_refined"
    )

    print("\nAvailable diseases:", loader.get_available_diseases()[:5], "...")

    for d_id in ["01", "03", "45"]:
        info = loader.get_graph_info(d_id)
        if info:
            print(f"  {d_id}: {info['version']} - {info['filepath']}")
        else:
            print(f"  {d_id}: Not available")

    test_ids = ["01", "03", "99"]
    result = loader.load_for_candidates(test_ids)

    for d_id, graph in result.items():
        if graph:
            print(f"  {d_id}: {graph}")
        else:
            print(f"  {d_id}: None (not available)")

    print("\nCache stats:", loader.get_cache_statistics())
