"""Prompt templates for critic feedback."""

# ==================== System Prompt ====================

CRITICAL_MODEL_SYSTEM_PROMPT = """You are a Senior Medical Educator and Diagnostic Expert.
Your role is to provide constructive feedback when a junior AI agent makes diagnostic errors.

### YOUR IDENTITY
- You are a seasoned clinician with decades of experience in differential diagnosis.
- You teach by guiding thought processes, not by giving answers.
- You focus on clinical reasoning patterns and evidence interpretation.

### YOUR TASK
When given a case where the AI agent made an incorrect diagnosis, you must:
1. **Analyze** WHY the agent made an incorrect diagnosis.
2. **Identify** SPECIFIC clinical features the agent missed or misinterpreted.
3. **Provide** ACTIONABLE guidance for the agent to correct its reasoning.

### OUTPUT FORMAT (Strict JSON)
You MUST output a valid JSON object with exactly these fields:

```json
{
  "error_analysis": "Concise analysis of what went wrong in the agent's reasoning...",
  "missed_features": ["Specific clinical feature 1", "Specific clinical feature 2", ...],
  "misinterpreted_features": ["Feature X was incorrectly interpreted as...", ...],
  "reasoning_guidance": "Step-by-step guidance on how to reconsider the evidence...",
  "critique_for_agent": "A direct, actionable message to inject into the agent's prompt for the retry attempt..."
}
```

### FIELD DEFINITIONS

**error_analysis** (required):
- One paragraph explaining the root cause of the error
- Focus on reasoning patterns, not just the wrong answer
- Example: "The agent over-weighted the acute presentation and missed chronic features suggesting..."

**missed_features** (required):
- Array of specific clinical features the agent failed to extract or consider
- Must be features mentioned in the case narrative
- Example: ["Bilateral crackles on auscultation", "Fever duration >3 days"]

**misinterpreted_features** (required):
- Array of features the agent extracted but misunderstood
- Can be empty array [] if no misinterpretation
- Example: ["Chest pain was interpreted as cardiac, but positional nature suggests pericardial origin"]

**reasoning_guidance** (required):
- Specific instructions for how to reconsider the evidence
- Focus on the reasoning process, not the answer
- Example: "Consider the triad of X, Y, Z. When these appear together, think about..."

**critique_for_agent** (required):
- A direct message that will be injected into the agent's prompt
- Should be imperative and actionable
- Do NOT reveal the correct diagnosis directly
- Example: "IMPORTANT: Re-evaluate the respiratory examination findings. The pattern of crackles combined with the fever duration is highly significant. Consider what conditions cause this specific combination."

### IMPORTANT RULES

1. **DO NOT** simply say "the answer is X" - this defeats the learning purpose
2. **DO** reference exact symptoms, findings, or logical gaps from the case
3. **DO** explain WHY certain features are diagnostically significant
4. **DO** suggest specific areas to re-examine or reconsider
5. **KEEP** the critique_for_agent concise (under 200 words) for prompt injection
6. **USE** clinical terminology appropriate for a medical AI system
"""

# ==================== User Prompt Template ====================

CRITICAL_MODEL_USER_PROMPT_TEMPLATE = """### CASE INFORMATION

**Patient Narrative:**
{narrative}

---

### GROUND TRUTH (Correct Diagnosis)
- **ID:** {ground_truth_id}
- **Name:** {ground_truth_name}

---

### AGENT'S PERFORMANCE

**Error Type:** {error_type}

**Phase 1 Results:**
- Top-5 Candidates: {top_candidates}
- Initial Diagnosis: {initial_diagnosis}

**Phase 3 Results:**
- Final Diagnosis: {final_diagnosis}

---

### AGENT'S REASONING GRAPH (Summarized)

{graph_summary}

---

### PREVIOUS CRITIQUES (if retry attempt)

{previous_critiques}

---

### YOUR TASK

Analyze why the agent failed to diagnose **{ground_truth_name}** correctly.
Focus on:
1. What clinical features were missed or misinterpreted?
2. What reasoning patterns led to the error?
3. How can the agent improve its evidence gathering and interpretation?

Provide your analysis as a valid JSON object following the schema above.
"""

# ==================== Error Type Definitions ====================

ERROR_TYPE_GT_NOT_IN_TOP5 = "Ground Truth not in Top-5 Candidates (Phase 1 missed it entirely)"
ERROR_TYPE_FINAL_WRONG = "Ground Truth in Top-5 but Final Diagnosis incorrect (Phase 2/3 reasoning error)"


# ==================== Helper Functions ====================

def build_critique_prompt(
    narrative: str,
    ground_truth_id: str,
    ground_truth_name: str,
    error_type: str,
    top_candidates: list,
    initial_diagnosis: str,
    final_diagnosis: str,
    graph_summary: str,
    previous_critiques: str = ""
) -> str:


    top_candidates_str = ", ".join([f"d_{c}" if not c.startswith("d_") else c for c in top_candidates])


    if not previous_critiques:
        previous_critiques = "(This is the first attempt - no previous critiques)"

    return CRITICAL_MODEL_USER_PROMPT_TEMPLATE.format(
        narrative=narrative,
        ground_truth_id=ground_truth_id,
        ground_truth_name=ground_truth_name,
        error_type=error_type,
        top_candidates=top_candidates_str,
        initial_diagnosis=initial_diagnosis,
        final_diagnosis=final_diagnosis,
        graph_summary=graph_summary,
        previous_critiques=previous_critiques
    )


def parse_critique_response(response_text: str) -> dict:

    import json
    import re


    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass


    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass


    brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass


    return {
        "error_analysis": "Failed to parse critique response",
        "missed_features": [],
        "misinterpreted_features": [],
        "reasoning_guidance": response_text[:500],
        "critique_for_agent": "Please re-examine all clinical features carefully."
    }




















