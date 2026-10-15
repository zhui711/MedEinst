"""Prompt templates for retrieved-evidence filtering."""
from typing import List, Optional, Dict, Any
import json


PHASE2_BATCH_FILTER_SYSTEM_PROMPT = """You are a strict Medical Data Curator for a clinical decision support system.

You will receive a list of text snippets. For EACH snippet, independently determine if it contains USEFUL clinical information for diagnosis.

### ACCEPT (Output: true):
- Clinical presentation, signs, and symptoms
- Diagnostic criteria or differential diagnosis guidelines  
- Pathophysiology that helps understand disease manifestations
- Review articles summarizing clinical features
- Treatment guidelines that mention clinical indicators

### REJECT (Output: false):
- Funding acknowledgments or grant information
- Animal studies or in-vitro experiments only
- Administrative, policy, or health economics data
- Single case reports with n=1 (high noise, low generalizability)
- Purely molecular/genetic studies without clinical correlation
- Epidemiology statistics without symptom descriptions
- Drug advertisements or commercial content
- Text about a COMPLETELY DIFFERENT disease than the target

### CRITICAL RULES:
1. Process each snippet INDEPENDENTLY
2. Output MUST be a valid JSON object
3. The "results" array length MUST equal the number of input snippets
4. Use lowercase boolean values: true or false (NOT "YES"/"NO")"""


PHASE2_BATCH_FILTER_USER_PROMPT_TEMPLATE = """Evaluate the following {n} text snippets for clinical relevance.

For EACH snippet, determine if it contains useful clinical information (symptoms, signs, diagnosis criteria) relevant to the indicated target disease.

{snippets_section}

---

Output your evaluation as a JSON object with exactly {n} boolean values:

```json
{{
  "results": [true/false, true/false, ...]
}}
```

IMPORTANT: The "results" array must have exactly {n} elements, one for each snippet above."""


def build_batch_filter_prompt(snippets: List[Dict[str, str]]) -> str:

    n = len(snippets)


    snippet_lines = []
    for i, item in enumerate(snippets, 1):
        disease = item.get("disease", "Unknown")
        text = item.get("text", "")

        if len(text) > 2000:
            text = text[:2000] + "... [truncated]"

        snippet_lines.append(f"--- SNIPPET {i} (Target Disease: {disease}) ---")
        snippet_lines.append(text)
        snippet_lines.append("")

    snippets_section = "\n".join(snippet_lines)

    return PHASE2_BATCH_FILTER_USER_PROMPT_TEMPLATE.format(
        n=n,
        snippets_section=snippets_section
    )


def parse_batch_filter_response(response: str, expected_count: int) -> List[bool]:

    if not response:
        print("[BatchFilter] Empty response, defaulting to all False")
        return [False] * expected_count


    try:

        cleaned = response.strip()
        if cleaned.startswith("```"):

            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[start_idx:end_idx])

        data = json.loads(cleaned)


        results = data.get("results", [])

        if not isinstance(results, list):
            print(f"[BatchFilter] 'results' is not a list, defaulting to all False")
            return [False] * expected_count


        if len(results) != expected_count:
            print(f"[BatchFilter] Length mismatch: got {len(results)}, expected {expected_count}")

            if len(results) < expected_count:
                results.extend([False] * (expected_count - len(results)))
            else:
                results = results[:expected_count]


        bool_results = []
        for r in results:
            if isinstance(r, bool):
                bool_results.append(r)
            elif isinstance(r, str):
                bool_results.append(r.lower() in ["true", "yes", "1"])
            else:
                bool_results.append(bool(r))

        return bool_results

    except json.JSONDecodeError as e:
        print(f"[BatchFilter] JSON parse error: {e}")
        print(f"[BatchFilter] Response was: {response[:200]}...")
        return [False] * expected_count
    except Exception as e:
        print(f"[BatchFilter] Unexpected error: {e}")
        return [False] * expected_count




PHASE2_FILTER_SYSTEM_PROMPT = """You are a strict medical data curator for a clinical decision support system.

Your task is to determine if a given text contains USEFUL clinical information for diagnosis.

### ACCEPT (Return YES):
- Clinical presentation, signs, and symptoms
- Diagnostic criteria or differential diagnosis guidelines
- Pathophysiology that helps understand disease manifestations
- Review articles summarizing clinical features
- Treatment guidelines that mention clinical indicators

### REJECT (Return NO):
- Funding acknowledgments or grant information
- Animal studies or in-vitro experiments only
- Administrative, policy, or health economics data
- Single case reports with n=1 (high noise, low generalizability)
- Purely molecular/genetic studies without clinical correlation
- Epidemiology statistics without symptom descriptions
- Drug advertisements or commercial content

### OUTPUT FORMAT
Respond with ONLY one word: YES or NO

Do not provide explanations. Just YES or NO."""

PHASE2_FILTER_USER_PROMPT_TEMPLATE = """Evaluate the following medical text:

---
{text_snippet}
---

Does this text contain useful clinical information (symptoms, signs, diagnosis criteria)?
Answer YES or NO only."""


def parse_filter_response(response: str) -> bool:

    if not response:
        return False


    cleaned = response.strip().upper()


    if "YES" in cleaned:
        return True


    if "NO" in cleaned:
        return False


    return False
