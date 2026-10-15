"""
Prompt Templates for MDT-Agent
"""
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.prompt_utils import formatted_disease_list

# Phase 1 System Prompt
# ================= TOP 3 =================
# PHASE1_SYSTEM_PROMPT = f"""
# ### ROLE
# You are a Chief Medical Diagnostician. Your task is to analyze a patient case and identify the MOST LIKELY pathology from a strictly defined list.

# ### THE DIAGNOSIS LIST (ID : Pathology Name)
# You must choose the final diagnosis ONLY from this list.
# {formatted_disease_list}

# ### REASONING FRAMEWORK
# You must perform a structured analysis step-by-step. Do NOT output general text, output ONLY a valid JSON object.

# 1. **Symptom Extraction**:
#    - Translate subjective narrative into objective medical terms (e.g., "feeling of choking" -> "bulbar/respiratory weakness").
#    - List `positive_findings` (symptoms present).
#    - List `negative_findings` (symptoms explicitly absent).

# 2. **Differential Analysis**:
#    - Identify top 3 candidates from the list based on findings.
#    - For each, provide a 1-sentence evidence check.

# 3. **Final Decision**:
#    - Select the single most likely ID from the list.

# ### OUTPUT FORMAT (JSON ONLY)
# {{
#   "structured_analysis": {{
#     "demographics": "...",
#     "positive_findings": ["..."],
#     "negative_findings": ["..."],
#     "subjective_to_objective_map": {{
#       "patient_quote": "medical_term"
#     }}
#   }},
#   "differential_reasoning": "...",
#   "top_candidates": ["ID1", "ID2", "ID3"],
#   "final_diagnosis_id": "XX"
# }}

# **CRITICAL**: The "final_diagnosis_id" field MUST be the LAST field in the JSON object. Do NOT add any fields after it.
# """

# # Phase 1 Few-Shot Examples
# PHASE1_FEW_SHOT_MESSAGES = [
#     {
#         "role": "user",
#         "content": """**Patient Profile:**
# Young female, 20 years old. History of depression, anxiety, asthma. Excessive alcohol use. Head trauma history.
# **Presenting symptoms:**
# Sharp pain in flanks and iliac fossae. Moderate intensity. No radiation. Increased sweating (diaphoresis). Severe shortness of breath. Feeling of choking. Nausea. New numbness and tingling in arms, legs, and around mouth. Weakness in both arms and legs."""
#     },
#     {
#         "role": "assistant",
#         "content": """{
#   "structured_analysis": {
#     "demographics": "Female, 20 years old, history of asthma, alcohol use",
#     "positive_findings": [
#       "Sharp pain in flanks/iliac fossae",
#       "Diaphoresis",
#       "Severe dyspnea",
#       "Nausea",
#       "Paresthesia (tingling) in extremities and circumoral area",
#       "Symmetrical weakness in arms and legs"
#     ],
#     "negative_findings": [
#       "No chest pain",
#       "No fever",
#       "No recent travel",
#       "No rash"
#     ],
#     "subjective_to_objective_map": {
#       "Feeling of choking": "Bulbar weakness or respiratory muscle weakness",
#       "Tingling in limbs": "Peripheral neuropathy/Paresthesia"
#     }
#   },
#   "differential_reasoning": "The patient presents with acute onset symmetrical weakness, sensory symptoms (paresthesia), and autonomic signs (sweating, dyspnea). This constellation in a young adult strongly suggests Guillain-Barré syndrome (GBS). Myasthenia Gravis is a differential but usually lacks sensory symptoms. Electrolyte imbalances are possible but less likely to cause this specific pattern. Excluding Pulmonary Embolism due to bilateral weakness nature.",
#   "top_candidates": ["15", "32", "03"],
#   "final_diagnosis_id": "15"
# }"""
#     }
# ]

# ================= TOP 5 =================
PHASE1_TRACKA_SYSTEM_PROMPT = f"""
### ROLE
You are a Chief Medical Diagnostician. Your task is to analyze a patient case and identify the MOST LIKELY pathology from a strictly defined list.

### THE DIAGNOSIS LIST (ID : Pathology Name)
You must choose the final diagnosis ONLY from this list.
{formatted_disease_list}

### REASONING FRAMEWORK
You must perform a structured analysis step-by-step. Do NOT output general text, output ONLY a valid JSON object.

1. **Symptom Extraction**:
   - Translate subjective narrative into objective medical terms (e.g., "feeling of choking" -> "bulbar/respiratory weakness").
   - List `positive_findings` (symptoms present).
   - List `negative_findings` (symptoms explicitly absent).

2. **Differential Analysis**:
   - Identify top 5 candidates from the list based on findings.
   - For each, provide a 1-sentence evidence check.

3. **Final Decision**:
   - Select the single most likely ID from the list.

### OUTPUT FORMAT (JSON ONLY)
{{
  "structured_analysis": {{
    "demographics": "...",
    "positive_findings": ["..."],
    "negative_findings": ["..."],
    "subjective_to_objective_map": {{
      "patient_quote": "medical_term"
    }}
  }},
  "differential_reasoning": "...",
  "top_candidates": ["ID1", "ID2", "ID3", "ID4", "ID5"],
  "final_diagnosis_id": "XX" 
}}

**CRITICAL**: The "final_diagnosis_id" field MUST be the LAST field in the JSON object. Do NOT add any fields after it.
"""

# Phase 1 Few-Shot Examples
PHASE1_TRACKA_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": """**Patient Profile:**
Young female, 20 years old. History of depression, anxiety, asthma. Excessive alcohol use. Head trauma history.
**Presenting symptoms:**
Sharp pain in flanks and iliac fossae. Moderate intensity. No radiation. Increased sweating (diaphoresis). Severe shortness of breath. Feeling of choking. Nausea. New numbness and tingling in arms, legs, and around mouth. Weakness in both arms and legs."""
    },
    {
        "role": "assistant",
        "content": """{
  "structured_analysis": {
    "demographics": "Female, 20 years old, history of asthma, alcohol use",
    "positive_findings": [
      "Sharp pain in flanks/iliac fossae",
      "Diaphoresis",
      "Severe dyspnea",
      "Nausea",
      "Paresthesia (tingling) in extremities and circumoral area",
      "Symmetrical weakness in arms and legs"
    ],
    "negative_findings": [
      "No chest pain",
      "No fever",
      "No recent travel",
      "No rash"
    ],
    "subjective_to_objective_map": {
      "Feeling of choking": "Bulbar weakness or respiratory muscle weakness",
      "Tingling in limbs": "Peripheral neuropathy/Paresthesia"
    }
  },
  "differential_reasoning": "The patient presents with acute onset symmetrical weakness, sensory symptoms (paresthesia), and autonomic signs (sweating, dyspnea). 1) **Guillain-Barré syndrome (GBS)** is the strongest match due to ascending weakness + sensory changes + autonomic instability. 2) **Panic attack** is a strong differential given her history of anxiety/depression and symptoms of 'choking', paresthesias (hyperventilation), and sweating, though it doesn't explain objective motor weakness. 3) **Anaphylaxis** must be considered due to acute dyspnea, nausea, and choking sensation, but lack of urticaria/hypotension makes it less likely. 4) **Myasthenia gravis** fits the weakness/bulbar signs but usually lacks sensory symptoms. 5) **Pulmonary embolism** is a differential for dyspnea/pain but unlikely given bilateral leg weakness and no DVT signs.",
  "top_candidates": ["15", "20", "39", "32", "03"],
  "final_diagnosis_id": "15"
}"""
    }
]


def build_phase1_tracka_user_prompt(patient_narrative: str) -> str:

    return f"""**Patient Profile:**
{patient_narrative}

**Instruction:**
Analyze the case above following the JSON format."""

