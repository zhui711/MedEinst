"""Prompt templates for Illness Graph refinement."""


REFINEMENT_SYSTEM_PROMPT = """You are a **Medical Ontologist** specializing in clinical terminology standardization.

## YOUR TASK
Analyze a list of medical knowledge nodes (K-Nodes) and identify **ONLY clinically equivalent synonyms** that should be merged into a single canonical term.

## CONSERVATIVE STRATEGY (CRITICAL)
You must follow an **extremely conservative** merging policy:
- **MERGE ONLY** if terms are 100% clinically interchangeable in a diagnostic context.
- **DO NOT MERGE** if there is ANY difference in:
  - **Laterality**: Left vs Right (e.g., "Left arm pain" ≠ "Right arm pain")
  - **Severity/Degree**: Mild vs Severe, Low vs High (e.g., "Fever" ≠ "High fever")
  - **Nature/Quality**: Dry vs Wet, Acute vs Chronic (e.g., "Dry cough" ≠ "Productive cough")
  - **Specificity**: Parent vs Child concepts (e.g., "Cough" ≠ "Dry cough")
  - **Timing**: Acute vs Chronic, Recent vs Long-standing
  - **Location specificity**: General vs Specific (e.g., "Pain" ≠ "Chest pain")

**When in doubt, KEEP SEPARATE.** It is far better to have duplicate terms than to incorrectly merge distinct clinical concepts.

## EXAMPLES

### ✅ SHOULD MERGE (Clinically Equivalent Synonyms)
| Term A | Term B | Reason |
|--------|--------|--------|
| Chest pain | Thoracic pain | Same anatomical location, professional vs common term |
| Shortness of breath | Dyspnea | Medical term vs lay term, same meaning |
| Heart attack | Myocardial infarction | Lay term vs medical term |
| Vomiting | Emesis | Same clinical finding |
| Belly pain | Abdominal pain | Colloquial vs medical term |
| High blood pressure | Hypertension | Lay vs medical terminology |

### ❌ SHOULD NOT MERGE (Clinically Distinct)
| Term A | Term B | Reason |
|--------|--------|--------|
| Dry cough | Productive cough | Different clinical nature (no sputum vs sputum) |
| Left arm pain | Right arm pain | Laterality matters (Left arm pain suggests cardiac) |
| Fever | High fever | Degree difference (High fever >39°C has clinical significance) |
| Cough | Dry cough | Parent-child relationship, different specificity |
| Acute pain | Chronic pain | Temporal distinction with different differential |
| Mild dyspnea | Severe dyspnea | Severity affects triage and diagnosis |
| Headache | Migraine | Generic vs specific diagnosis |
| Fatigue | Chronic fatigue | Duration distinction |

## INPUT FORMAT
You will receive a Markdown list of K-Nodes:
```
1. "Chest pain" (Count: 45)
2. "Thoracic pain" (Count: 12)
3. "Dyspnea" (Count: 30)
...
```

## OUTPUT FORMAT
Return a JSON array containing **ONLY merge clusters** (groups of 2+ synonyms).
**DO NOT include singleton clusters** (terms with no synonyms) - they will be kept automatically.

```json
[
  {
    "canonical_name": "Chest pain",
    "merge_indices": [1, 2],
    "rationale": "Thoracic pain is anatomically equivalent to chest pain"
  },
  {
    "canonical_name": "Shortness of breath",
    "merge_indices": [5, 12, 45],
    "rationale": "Dyspnea and breathlessness are clinical synonyms"
  }
]
```

If NO synonyms are found in the entire list, return an empty array: `[]`

## RULES FOR canonical_name SELECTION
1. **Prefer the term with higher occurrence count** (more commonly used in training data).
2. **Prefer standard medical terminology** over colloquial terms when counts are similar.
3. **Preserve clinical precision** - choose the more specific term if both are valid.

## IMPORTANT
- **ONLY output clusters where 2+ terms should be merged.**
- `merge_indices` must contain the 1-based indices of ALL terms in the merge group.
- The `canonical_name` MUST be one of the original input terms (use exact spelling).
- Each index should appear in at most ONE cluster.
- Keep `rationale` brief (max 15 words).
"""

# User Prompt Template
REFINEMENT_USER_PROMPT_TEMPLATE = """## Target Pathology
**Disease:** {pathology_name} (ID: {pathology_id})

## K-Nodes to Analyze
Total: {total_k_nodes} nodes

{k_nodes_markdown_list}

---

**Instructions:**
1. Analyze the above K-Nodes for the disease "{pathology_name}".
2. Identify ONLY clinically equivalent synonyms following the CONSERVATIVE strategy.
3. Return the clustering result as a JSON array.
4. Every index (1 to {total_k_nodes}) must appear in exactly one cluster.

Output JSON only, no additional text."""


def format_k_nodes_to_markdown(k_nodes: list) -> str:

    lines = []
    for idx, node in enumerate(k_nodes, start=1):

        if isinstance(node, dict):
            content = node.get("content", "")
            count = node.get("occurrence_count", 1)
        else:
            content = node.content
            count = node.occurrence_count

        lines.append(f'{idx}. "{content}" (Count: {count})')

    return "\n".join(lines)



def validate_clustering_result(clusters: list, total_nodes: int) -> tuple:

    if not isinstance(clusters, list):
        return False, "Result is not a list"


    if len(clusters) == 0:
        return True, "Valid (no merges)"


    all_indices = set()

    for i, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            return False, f"Cluster {i} is not a dict"

        if "canonical_name" not in cluster:
            return False, f"Cluster {i} missing 'canonical_name'"


        indices_key = "merge_indices" if "merge_indices" in cluster else "original_indices"
        if indices_key not in cluster:
            return False, f"Cluster {i} missing 'merge_indices'"

        indices = cluster[indices_key]
        if not isinstance(indices, list):
            return False, f"Cluster {i} '{indices_key}' is not a list"


        if len(indices) < 2:
            return False, f"Cluster {i} has less than 2 indices (not a merge)"

        for idx in indices:
            if not isinstance(idx, int):
                return False, f"Cluster {i} contains non-integer index: {idx}"

            if idx < 1 or idx > total_nodes:
                return False, f"Index {idx} out of range [1, {total_nodes}]"

            if idx in all_indices:
                return False, f"Duplicate index: {idx}"

            all_indices.add(idx)




    return True, "Valid"


def normalize_clustering_result(clusters: list, k_nodes_list: list) -> list:

    total_nodes = len(k_nodes_list)


    merged_indices = set()
    normalized_clusters = []

    for cluster in clusters:

        indices_key = "merge_indices" if "merge_indices" in cluster else "original_indices"
        indices = cluster[indices_key]

        merged_indices.update(indices)


        normalized_clusters.append({
            "canonical_name": cluster["canonical_name"],
            "original_indices": indices,
            "rationale": cluster.get("rationale", "Merged by LLM")
        })


    for idx in range(1, total_nodes + 1):
        if idx not in merged_indices:
            node = k_nodes_list[idx - 1]
            content = node.get("content", "") if isinstance(node, dict) else node.content
            normalized_clusters.append({
                "canonical_name": content,
                "original_indices": [idx],
                "rationale": "No synonyms found"
            })

    return normalized_clusters

