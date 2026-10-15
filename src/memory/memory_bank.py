"""Store and retrieve compact validated-case exemplars."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .embedding_client import EmbeddingClient


class MemoryBankManager:
    """Persist and retrieve compact Exemplar Base records."""

    def __init__(
        self,
        output_dir: str = "memory_bank",
        embedding_client: Optional[EmbeddingClient] = None
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_client = embedding_client

        self.cases: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

        self._overturn_indices: List[int] = []
        self._confirm_indices: List[int] = []

    @property
    def embedding_client(self) -> EmbeddingClient:

        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()
        return self._embedding_client

    def add_case(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        initial_diagnosis_id: str,
        initial_diagnosis_name: str,
        final_diagnosis_id: str,
        final_diagnosis_name: str,
        p_nodes: List[Dict[str, Any]],
        retry_count: int = 0
    ) -> None:

        p_nodes_summary = self._generate_summary(p_nodes)

        try:
            embedding = self.embedding_client.embed_text(p_nodes_summary)
        except Exception as e:
            print(f"[MemoryBank] Embedding failed for {case_id}: {e}")

            embedding = np.random.randn(1536).astype(np.float32)

        outcome = "Confirm" if initial_diagnosis_id == final_diagnosis_id else "Overturn"

        # The JSON record points to its vector in embeddings.npy by row index.
        record = {
            "case_id": case_id,
            "type": case_type,
            "ground_truth_id": ground_truth_id,
            "ground_truth_name": ground_truth_name,
            "initial_diagnosis_id": initial_diagnosis_id,
            "initial_diagnosis_name": initial_diagnosis_name,
            "final_diagnosis_id": final_diagnosis_id,
            "final_diagnosis_name": final_diagnosis_name,
            "outcome": outcome,
            "p_nodes_summary": p_nodes_summary,
            "embedding_index": len(self.embeddings),
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat()
        }

        idx = len(self.cases)
        if outcome == "Overturn":
            self._overturn_indices.append(idx)
        else:
            self._confirm_indices.append(idx)

        self.cases.append(record)
        self.embeddings.append(embedding)

        print(f"[MemoryBank] Added {case_id} ({outcome}). Total: {len(self.cases)}")

    def _generate_summary(self, p_nodes: List[Dict[str, Any]]) -> str:

        present = []
        absent = []

        for p in p_nodes:
            content = p.get("content", "")
            status = p.get("status", "Present")

            if status == "Present":
                present.append(content)
            elif status == "Absent":
                absent.append(content)

        summary_parts = []

        if present:

            present_list = ", ".join(present[:15])
            summary_parts.append(f"Present: {present_list}")

        if absent:
            absent_list = ", ".join(absent[:10])
            summary_parts.append(f"Absent: {absent_list}")

        return "; ".join(summary_parts) if summary_parts else "No clinical features"

    def retrieve_similar(
        self,
        query_p_nodes: List[Dict[str, Any]],
        n_overturn: int = 2,
        n_confirm: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:

        if not self.cases:
            return {"overturn": [], "confirm": []}

        query_summary = self._generate_summary(query_p_nodes)
        try:
            query_embedding = self.embedding_client.embed_text(query_summary)
        except Exception as e:
            print(f"[MemoryBank] Query embedding failed: {e}")
            return {"overturn": [], "confirm": []}

        embeddings_matrix = np.stack(self.embeddings)
        similarities = self._cosine_similarity(query_embedding, embeddings_matrix)

        overturn_cases = self._retrieve_by_outcome(
            similarities, self._overturn_indices, n_overturn
        )
        confirm_cases = self._retrieve_by_outcome(
            similarities, self._confirm_indices, n_confirm
        )

        return {
            "overturn": overturn_cases,
            "confirm": confirm_cases
        }

    def _retrieve_by_outcome(
        self,
        similarities: np.ndarray,
        indices: List[int],
        n: int
    ) -> List[Dict[str, Any]]:

        if not indices or n <= 0:
            return []

        outcome_similarities = [(idx, similarities[idx]) for idx in indices]

        outcome_similarities.sort(key=lambda x: x[1], reverse=True)

        result = []
        for idx, sim in outcome_similarities[:n]:
            case = self.cases[idx].copy()
            case["similarity_score"] = float(sim)
            result.append(case)

        return result

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

        return np.dot(matrix_norm, query_norm)

    def save(self) -> None:

        data = {
            "version": "1.0",
            "total_cases": len(self.cases),
            "overturn_count": len(self._overturn_indices),
            "confirm_count": len(self._confirm_indices),
            "cases": self.cases
        }

        json_path = self.output_dir / "memory_bank.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self.embeddings:
            embeddings_matrix = np.stack(self.embeddings)
            npy_path = self.output_dir / "embeddings.npy"
            np.save(npy_path, embeddings_matrix)

        print(f"[MemoryBank] Saved {len(self.cases)} cases to {self.output_dir}")

    def load(self) -> None:

        json_path = self.output_dir / "memory_bank.json"
        npy_path = self.output_dir / "embeddings.npy"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cases = data.get("cases", [])

            self._overturn_indices = []
            self._confirm_indices = []

            for idx, case in enumerate(self.cases):
                if case.get("outcome") == "Overturn":
                    self._overturn_indices.append(idx)
                else:
                    self._confirm_indices.append(idx)

            print(f"[MemoryBank] Loaded {len(self.cases)} cases from {json_path}")

        if npy_path.exists():
            embeddings_matrix = np.load(npy_path)
            self.embeddings = [embeddings_matrix[i] for i in range(len(embeddings_matrix))]
            print(f"[MemoryBank] Loaded {len(self.embeddings)} embeddings from {npy_path}")

    def get_statistics(self) -> Dict[str, Any]:

        return {
            "total_cases": len(self.cases),
            "overturn_count": len(self._overturn_indices),
            "confirm_count": len(self._confirm_indices),
            "storage_path": str(self.output_dir)
        }

    def format_few_shot_context(
        self,
        similar_cases: Dict[str, List[Dict[str, Any]]]
    ) -> str:

        lines = []
        lines.append("### SIMILAR CASES FROM MEMORY BANK ###\n")

        # Overturn Cases
        overturn_cases = similar_cases.get("overturn", [])
        if overturn_cases:
            lines.append("**Successful Overturn Examples:**")
            for i, case in enumerate(overturn_cases, 1):
                lines.append(f"""
Case {i} (Similarity: {case.get('similarity_score', 0):.3f}):
- Patient Features: {case.get('p_nodes_summary', 'N/A')}
- Initial Diagnosis: {case.get('initial_diagnosis_name', 'N/A')}
- Final Diagnosis: {case.get('final_diagnosis_name', 'N/A')} (Correct)
- Lesson: The system correctly overturned the initial intuition.
""")

        # Confirm Cases
        confirm_cases = similar_cases.get("confirm", [])
        if confirm_cases:
            lines.append("\n**Successful Confirm Examples:**")
            for i, case in enumerate(confirm_cases, 1):
                lines.append(f"""
Case {i} (Similarity: {case.get('similarity_score', 0):.3f}):
- Patient Features: {case.get('p_nodes_summary', 'N/A')}
- Initial Diagnosis: {case.get('initial_diagnosis_name', 'N/A')}
- Final Diagnosis: {case.get('final_diagnosis_name', 'N/A')} (Correct)
- Lesson: The system correctly confirmed the initial intuition.
""")

        if not overturn_cases and not confirm_cases:
            lines.append("(No similar cases found in Memory Bank)")

        return "\n".join(lines)

if __name__ == "__main__":

    class MockEmbeddingClient:
        def embed_text(self, text: str) -> np.ndarray:

            np.random.seed(hash(text) % (2**32))
            return np.random.randn(1536).astype(np.float32)

    manager = MemoryBankManager(output_dir="memory_bank_test")
    manager._embedding_client = MockEmbeddingClient()

    manager.add_case(
        case_id="test_001",
        case_type="control",
        ground_truth_id="45",
        ground_truth_name="Pneumonia",
        initial_diagnosis_id="23",
        initial_diagnosis_name="Bronchitis",
        final_diagnosis_id="45",
        final_diagnosis_name="Pneumonia",
        p_nodes=[
            {"content": "Fever", "status": "Present"},
            {"content": "Productive cough", "status": "Present"},
            {"content": "Chest pain", "status": "Absent"}
        ],
        retry_count=1
    )

    manager.add_case(
        case_id="test_002",
        case_type="trap",
        ground_truth_id="23",
        ground_truth_name="Bronchitis",
        initial_diagnosis_id="23",
        initial_diagnosis_name="Bronchitis",
        final_diagnosis_id="23",
        final_diagnosis_name="Bronchitis",
        p_nodes=[
            {"content": "Cough", "status": "Present"},
            {"content": "Sore throat", "status": "Present"}
        ],
        retry_count=0
    )

    print("\nStatistics:", manager.get_statistics())

    similar = manager.retrieve_similar(
        query_p_nodes=[{"content": "Fever", "status": "Present"}],
        n_overturn=1,
        n_confirm=1
    )
    print("\nSimilar cases:", similar)

    manager.save()

    manager2 = MemoryBankManager(output_dir="memory_bank_test")
    manager2.load()
    print("\nReloaded statistics:", manager2.get_statistics())

