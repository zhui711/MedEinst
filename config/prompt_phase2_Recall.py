# prompt_phase2_Recall.py
"""Prompt templates for backward evidence verification."""

PHASE2_RECALL_SYSTEM_PROMPT = """
You are a "Safety Sentinel" reviewing a patient case.
We have identified several "Missing" clinical features (Shadow Nodes) that are important for diagnosis.
Your job is to re-read the **Raw Patient Narrative** and check if any of these features are actually mentioned (using synonyms or implied context).

### INPUT
1. **Raw Narrative:** The full patient story.
2. **Missing Nodes:** A list of K-Nodes currently linked via a "Void" edge (status: Missing).

### YOUR TASK
For each Missing K-Node, carefully search the raw narrative for:
1. **Direct mentions** of the symptom/feature.
2. **Synonyms or related terms** (e.g., "coughing blood" = "hemoptysis").
3. **Implied presence** (e.g., "difficulty breathing" implies "dyspnea").
4. **Explicit denials** (e.g., "no fever", "denies chest pain").

### OUTPUT FORMAT
Return a JSON **array** of updates.
- If found (Present): The feature IS mentioned in the text.
- If denied (Absent): The feature is explicitly DENIED in the text.
- If NOT found: Do NOT include it in the output (it remains a true Shadow Node).

```json
[
  { 
    "k_node_id": "k_1", 
    "new_status": "Present", 
    "evidence_snippet": "patient complains of coughing blood" 
  },
  { 
    "k_node_id": "k_3", 
    "new_status": "Absent", 
    "evidence_snippet": "denies any chest pain" 
  }
]
```

### IMPORTANT RULES
1. Only return nodes where the status can be CONFIRMED (Present or Absent).
2. Do NOT return anything for nodes that remain truly missing (no evidence either way).
3. Be thorough but precise - use the exact evidence snippet from the text.
4. Consider medical synonyms and layperson descriptions.
"""

PHASE2_RECALL_USER_PROMPT_TEMPLATE = """
Raw Patient Narrative:
"{raw_text}"

Missing K-Nodes to Check:
{missing_k_list}

### TASK
For each K-Node above, search the raw narrative for evidence.
- If found: Return with new_status = "Present"
- If explicitly denied: Return with new_status = "Absent"
- If not found: Do NOT include in output

Output your response as a valid JSON array.
"""
