"""Semantic alignment between observations and graph nodes."""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
from dataclasses import dataclass


try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("[TextUtils] Warning: rapidfuzz not installed, fuzzy match fallback disabled")

@dataclass
class MatchResult:
    """Observation-to-graph match and its similarity."""
    gg_content: str
    patient_content: str
    patient_status: str
    similarity: float
    matched: bool
    match_method: str

class SemanticMatcher:
    """Match patient observations to Illness Graph nodes."""

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        fuzzy_threshold: float = 0.80,
        use_cache: bool = True,
        embedding_client=None
    ):

        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.use_cache = use_cache

        self._embedding_client = embedding_client
        self._embedding_available = embedding_client is not None
        self._cache: Dict[str, np.ndarray] = {}

        if embedding_client is None:
            self._init_embedding_client()

    def _init_embedding_client(self) -> None:

        try:
            from src.memory.embedding_client import EmbeddingClient

            if os.getenv("OPENAI_API_KEY"):
                self._embedding_client = EmbeddingClient()
                self._embedding_available = True
                print("[SemanticMatcher] Embedding client initialized successfully")
            else:
                print("[SemanticMatcher] Warning: No API key found, embedding disabled")

        except Exception as e:
            print(f"[SemanticMatcher] Warning: Failed to init embedding client: {e}")
            self._embedding_available = False

    def get_embedding(self, text: str) -> Optional[np.ndarray]:

        if not text or not text.strip():
            return None

        text_key = text.lower().strip()

        if self.use_cache and text_key in self._cache:
            return self._cache[text_key]

        if not self._embedding_available or self._embedding_client is None:
            return None

        try:
            embedding = self._embedding_client.embed_text(text)

            if self.use_cache:
                self._cache[text_key] = embedding

            return embedding

        except Exception as e:
            print(f"[SemanticMatcher] Embedding error for '{text[:50]}...': {e}")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:

        results = {}
        texts_to_fetch = []
        text_keys = []

        for text in texts:
            if not text or not text.strip():
                continue

            text_key = text.lower().strip()

            if self.use_cache and text_key in self._cache:
                results[text_key] = self._cache[text_key]
            else:
                texts_to_fetch.append(text)
                text_keys.append(text_key)

        if texts_to_fetch and self._embedding_available and self._embedding_client:
            try:
                embeddings = self._embedding_client.embed_texts(texts_to_fetch)

                for i, (text_key, embedding) in enumerate(zip(text_keys, embeddings)):
                    results[text_key] = embedding
                    if self.use_cache:
                        self._cache[text_key] = embedding

            except Exception as e:
                print(f"[SemanticMatcher] Batch embedding error: {e}")

        return results

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:

        if vec1 is None or vec2 is None:
            return 0.0

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def fuzzy_match_score(self, text1: str, text2: str) -> float:

        if not RAPIDFUZZ_AVAILABLE:

            t1 = text1.lower().strip()
            t2 = text2.lower().strip()

            if t1 == t2:
                return 1.0
            elif t1 in t2 or t2 in t1:
                return 0.85
            else:
                return 0.0

        score = fuzz.token_set_ratio(text1, text2) / 100.0
        return score

    def match_p_nodes(
        self,
        gg_p_nodes: List[Dict[str, Any]],
        patient_p_nodes: List[Dict[str, Any]]
    ) -> Tuple[List[MatchResult], bool]:

        results = []
        used_embedding = False

        if not gg_p_nodes or not patient_p_nodes:
            return results, used_embedding

        gg_contents = [p.get("content", "") for p in gg_p_nodes]
        patient_contents = [p.get("content", "") for p in patient_p_nodes]

        patient_map = {
            p.get("content", "").lower().strip(): p
            for p in patient_p_nodes
        }

        similarity_matrix = None

        if self._embedding_available:
            try:

                all_texts = gg_contents + patient_contents
                embeddings = self.get_embeddings_batch(all_texts)

                if embeddings:

                    similarity_matrix = np.zeros((len(gg_contents), len(patient_contents)))

                    for i, gg_content in enumerate(gg_contents):
                        gg_key = gg_content.lower().strip()
                        gg_emb = embeddings.get(gg_key)

                        if gg_emb is None:
                            continue

                        for j, patient_content in enumerate(patient_contents):
                            patient_key = patient_content.lower().strip()
                            patient_emb = embeddings.get(patient_key)

                            if patient_emb is not None:
                                similarity_matrix[i, j] = self.cosine_similarity(gg_emb, patient_emb)

                    used_embedding = True
                    print(f"[SemanticMatcher] Built similarity matrix: {similarity_matrix.shape}")

            except Exception as e:
                print(f"[Warning] Embedding API failed, falling back to fuzzy match: {e}")
                similarity_matrix = None

        if similarity_matrix is None:
            print("[SemanticMatcher] Using fuzzy match fallback")
            similarity_matrix = np.zeros((len(gg_contents), len(patient_contents)))

            for i, gg_content in enumerate(gg_contents):
                for j, patient_content in enumerate(patient_contents):
                    similarity_matrix[i, j] = self.fuzzy_match_score(gg_content, patient_content)

        threshold = self.similarity_threshold if used_embedding else self.fuzzy_threshold

        for i, gg_p in enumerate(gg_p_nodes):
            gg_content = gg_p.get("content", "")

            if not gg_content:
                continue

            max_sim = 0.0
            best_j = -1

            for j in range(len(patient_contents)):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    best_j = j

            matched = max_sim >= threshold

            if matched and best_j >= 0:
                patient_p = patient_p_nodes[best_j]
                result = MatchResult(
                    gg_content=gg_content,
                    patient_content=patient_p.get("content", ""),
                    patient_status=patient_p.get("status", "Present"),
                    similarity=max_sim,
                    matched=True,
                    match_method="embedding" if used_embedding else "fuzzy"
                )
                print(f"[Phase 2] P-Node Matching: GG node '{gg_content[:40]}...' "
                      f"matched Patient node '{patient_p.get('content', '')[:40]}...' "
                      f"(Sim: {max_sim:.2f})")
            else:
                result = MatchResult(
                    gg_content=gg_content,
                    patient_content="",
                    patient_status="",
                    similarity=max_sim,
                    matched=False,
                    match_method="embedding" if used_embedding else "fuzzy"
                )

            results.append(result)

        return results, used_embedding

    def clear_cache(self) -> None:

        self._cache.clear()
        print("[SemanticMatcher] Cache cleared")

    def get_cache_size(self) -> int:

        return len(self._cache)

_default_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher(
    similarity_threshold: float = 0.90,
    fuzzy_threshold: float = 0.80
) -> SemanticMatcher:

    global _default_matcher

    if _default_matcher is None:
        _default_matcher = SemanticMatcher(
            similarity_threshold=similarity_threshold,
            fuzzy_threshold=fuzzy_threshold
        )

    return _default_matcher

def match_p_nodes_semantic(
    gg_p_nodes: List[Dict[str, Any]],
    patient_p_nodes: List[Dict[str, Any]],
    similarity_threshold: float = 0.90
) -> Tuple[List[MatchResult], bool]:

    matcher = get_semantic_matcher(similarity_threshold=similarity_threshold)
    return matcher.match_p_nodes(gg_p_nodes, patient_p_nodes)

if __name__ == "__main__":
    print("Testing SemanticMatcher...")

    gg_p_nodes = [
        {"content": "Dyspnea"},
        {"content": "Chest pain radiating to left arm"},
        {"content": "Fever"},
    ]

    patient_p_nodes = [
        {"content": "Shortness of breath", "status": "Present"},
        {"content": "Severe chest pain with radiation to the left arm", "status": "Present"},
        {"content": "No fever", "status": "Absent"},
    ]

    matcher = SemanticMatcher(similarity_threshold=0.85)
    results, used_embedding = matcher.match_p_nodes(gg_p_nodes, patient_p_nodes)

    print(f"\nUsed Embedding: {used_embedding}")
    print(f"Results:")
    for r in results:
        print(f"  GG: '{r.gg_content}' -> Patient: '{r.patient_content}' "
              f"| Matched: {r.matched} | Sim: {r.similarity:.2f} | Status: {r.patient_status}")

