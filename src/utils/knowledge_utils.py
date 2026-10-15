"""Disease terminology and medical retrieval helpers."""
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


_SYNONYMS_CACHE: Optional[Dict[str, Dict]] = None


def load_disease_synonyms() -> Dict[str, Dict]:

    global _SYNONYMS_CACHE

    if _SYNONYMS_CACHE is not None:
        return _SYNONYMS_CACHE

    possible_paths = [
        Path(__file__).parent.parent.parent / "config" / "disease_synonyms.json",
        Path("config/disease_synonyms.json"),
    ]

    for synonyms_path in possible_paths:
        if synonyms_path.exists():
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    _SYNONYMS_CACHE = json.load(f)
                print(f"[KnowledgeUtils] Loaded {len(_SYNONYMS_CACHE)} disease synonyms from {synonyms_path}")
                return _SYNONYMS_CACHE
            except Exception as e:
                print(f"[KnowledgeUtils] Error loading synonyms from {synonyms_path}: {e}")
                continue

    print("[KnowledgeUtils] WARNING: disease_synonyms.json not found, using empty synonyms")
    _SYNONYMS_CACHE = {}
    return _SYNONYMS_CACHE

def get_synonyms_for_disease(disease_id: str, disease_name: str) -> List[str]:

    synonyms_map = load_disease_synonyms()

    clean_id = disease_id.replace("d_", "")

    if clean_id in synonyms_map:
        entry = synonyms_map[clean_id]
        synonyms = entry.get("synonyms", [])
        if synonyms:
            return synonyms

    return [disease_name]

def build_pubmed_or_query(
    disease_name: str,
    disease_id: str = "",
    require_review: bool = True,
    include_clinical_terms: bool = True
) -> str:

    synonyms = get_synonyms_for_disease(disease_id, disease_name)

    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms

    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]

    disease_terms = " OR ".join([f'"{s}"[Title/Abstract]' for s in synonyms])
    query_parts = [f"({disease_terms})"]

    if include_clinical_terms:
        clinical_terms = (
            '("clinical presentation"[Title/Abstract] OR '
            '"clinical features"[Title/Abstract] OR '
            '"signs and symptoms"[Title/Abstract] OR '
            '"diagnosis"[Title/Abstract])'
        )
        query_parts.append(clinical_terms)

    if require_review:
        review_filter = "(Review[Publication Type] OR Practice Guideline[Publication Type])"
        query_parts.append(review_filter)

    query = " AND ".join(query_parts)

    return query

def build_pubmed_statpearls_query(disease_name: str, disease_id: str = "") -> str:

    synonyms = get_synonyms_for_disease(disease_id, disease_name)

    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms

    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]

    disease_terms = " OR ".join([f'"{s}"[Title]' for s in synonyms])
    query = f'({disease_terms}) AND "StatPearls"[Source]'

    return query

def build_pubmed_clinical_query(disease_name: str, disease_id: str = "") -> str:

    synonyms = get_synonyms_for_disease(disease_id, disease_name)

    if disease_name not in synonyms:
        synonyms = [disease_name] + synonyms

    seen = set()
    unique_synonyms = []
    for s in synonyms:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen:
            seen.add(s_lower)
            unique_synonyms.append(s.strip())
    synonyms = unique_synonyms[:5]

    disease_terms = " OR ".join([f'"{s}"[Title/Abstract]' for s in synonyms])

    title_keywords = (
        '("Clinical Features"[Title] OR '
        '"Diagnosis"[Title] OR '
        '"Presentation"[Title])'
    )

    query = f'({disease_terms}) AND {title_keywords}'

    return query

class PubMedQueryBuilder:
    """Build disease-focused PubMed queries."""

    def __init__(self, disease_name: str, disease_id: str = ""):

        self.disease_name = disease_name
        self.disease_id = disease_id
        self._synonyms = None

    @property
    def synonyms(self) -> List[str]:

        if self._synonyms is None:
            synonyms = get_synonyms_for_disease(self.disease_id, self.disease_name)
            if self.disease_name not in synonyms:
                synonyms = [self.disease_name] + synonyms

            seen = set()
            unique = []
            for s in synonyms:
                s_lower = s.lower().strip()
                if s_lower and s_lower not in seen:
                    seen.add(s_lower)
                    unique.append(s.strip())
            self._synonyms = unique[:5]
        return self._synonyms

    def build_level1_statpearls(self) -> str:

        return build_pubmed_statpearls_query(self.disease_name, self.disease_id)

    def build_level2_review(self) -> str:

        return build_pubmed_or_query(
            self.disease_name,
            self.disease_id,
            require_review=True,
            include_clinical_terms=False
        )

    def build_level3_clinical(self) -> str:

        return build_pubmed_clinical_query(self.disease_name, self.disease_id)

    def get_all_queries(self) -> List[Tuple[str, str, int]]:

        return [
            ("StatPearls", self.build_level1_statpearls(), 3),
            ("Review", self.build_level2_review(), 3),
            ("Clinical_Fallback", self.build_level3_clinical(), 2)
        ]

def build_open_targets_search_terms(disease_name: str, disease_id: str = "") -> List[str]:

    synonyms = get_synonyms_for_disease(disease_id, disease_name)

    search_terms = []

    for syn in synonyms:

        cleaned = syn.strip()
        if cleaned and cleaned not in search_terms:
            search_terms.append(cleaned)

    if disease_name not in search_terms:
        search_terms.append(disease_name)

    return search_terms[:3]

def estimate_token_count(text: str) -> int:

    if not text:
        return 0
    return len(text) // 4

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:

    if not text:
        return ""

    estimated_chars = max_tokens * 4
    if len(text) <= estimated_chars:
        return text

    truncated = text[:estimated_chars]
    last_period = truncated.rfind('.')
    if last_period > estimated_chars * 0.7:
        truncated = truncated[:last_period + 1]

    return truncated + "..."

def validate_open_targets_match(
    search_term: str,
    returned_name: str,
    returned_synonyms: Optional[List[Dict[str, Any]]] = None
) -> bool:

    if not search_term or not returned_name:
        return False

    search_lower = search_term.lower().strip()
    name_lower = returned_name.lower().strip()

    if search_lower in name_lower:
        return True

    if name_lower in search_lower:
        return True

    if returned_synonyms:
        for syn_group in returned_synonyms:
            terms = syn_group.get("terms", [])
            for term in terms:
                if isinstance(term, str):
                    term_lower = term.lower().strip()
                    if search_lower in term_lower or term_lower in search_lower:
                        return True

    return False

def get_all_search_terms_for_disease(disease_id: str, disease_name: str) -> List[str]:

    synonyms = get_synonyms_for_disease(disease_id, disease_name)

    search_terms = []
    seen = set()

    if disease_name and disease_name.lower() not in seen:
        search_terms.append(disease_name)
        seen.add(disease_name.lower())

    for syn in synonyms:
        if syn and syn.lower() not in seen:
            search_terms.append(syn)
            seen.add(syn.lower())

    return search_terms

class ContextBudgetManager:
    """Bound retrieved text before model calls."""

    def __init__(self, max_total_tokens: int = 30000):

        self.max_total_tokens = max_total_tokens
        self.reserved_for_prompt = 3000
        self.reserved_for_output = 4000
        self.available_for_content = max_total_tokens - self.reserved_for_prompt - self.reserved_for_output

    def allocate_budget(
        self,
        candidates: List[Dict],
        open_targets_texts: Dict[str, str],
        pubmed_texts: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:

        result = {}
        n_candidates = len(candidates)

        if n_candidates == 0:
            return result

        ot_total_tokens = 0
        for d_id, text in open_targets_texts.items():
            ot_total_tokens += estimate_token_count(text)

        pubmed_budget = self.available_for_content - ot_total_tokens
        pubmed_per_candidate = max(500, pubmed_budget // n_candidates) if n_candidates > 0 else 0

        print(f"[ContextBudget] Total available: {self.available_for_content} tokens")
        print(f"[ContextBudget] Open Targets used: {ot_total_tokens} tokens")
        print(f"[ContextBudget] PubMed per candidate: {pubmed_per_candidate} tokens")

        for candidate in candidates:
            d_id = candidate.get("id", "")

            ot_text = open_targets_texts.get(d_id, "")

            pm_text = pubmed_texts.get(d_id, "")
            if pubmed_per_candidate > 0:
                pm_text = truncate_text_by_tokens(pm_text, pubmed_per_candidate)

            result[d_id] = {
                "open_targets": ot_text,
                "pubmed": pm_text
            }

        return result

