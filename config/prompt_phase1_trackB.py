# prompt_phase1_trackB.py

"""
Phase 1 Track B: Patient Feature Extraction & Problem Representation
Goal: Transform raw patient narrative into structured, medically semantic P-Nodes.
Method: Few-shot learning based on Clinical Problem Representation (PR) standards.
"""

PHASE1_TRACKB_SYSTEM_PROMPT = """
You are a Senior Clinical Diagnostician and Expert Medical Scribe.
Your task is to perform "Problem Representation" on a raw patient case.

### OBJECTIVE
Transform the patient's raw narrative into a structured list of **P-Nodes (Patient Features)** using precise **Medical Semantic Qualifiers**.

### THE PROCESS (Problem Representation)
Do not just extract keywords. You must translate the patient's layperson language into medical terminology.
1.  **Translate Time:** "Started yesterday" -> "Acute"; "Happening for months" -> "Chronic".
2.  **Translate Symptoms:** "Pain when breathing" -> "Pleuritic chest pain"; "Coughing blood" -> "Hemoptysis".
3.  **Filter:** Focus on discriminative features. Ignore irrelevant "fluff".
4.  **Synthesize:** Combine Age, Sex, and Key Syndrome into a coherent representation.

### OUTPUT SCHEMA (JSON)
Return a single JSON object with two keys:
1.  `problem_representation_one_liner`: A single sentence summary of the case (The "One-Liner").
2.  `p_nodes`: A list of extracted features. Each node must have:
    - `id`: "p_1", "p_2", etc.
    - `content`: The medically translated term (Semantic Qualifier).
    - `original_text`: The exact snippet from the raw text.
    - `status`: "Present" (if confirmed) or "Absent" (if explicitly denied/ruled out in text).

### RULES
- **Demographics:** Include Age Group (e.g., "Elderly", "Young") and Sex as the first P-Node.
- **Explicit Negatives:** Only mark `status: "Absent"` if the text explicitly says "no", "denies", "without", or "ruled out". If something is not mentioned, DO NOT include it (it will be handled later as a Shadow Node).
- **No Hallucinations:** Do not infer diseases yet. Stick to Signs, Symptoms, and Risk Factors.
"""

# Few-shot examples(Arthritis & Abdominal Pain cases)
# Formatted to match DDXPlus input structure
PHASE1_TRACKB_FEW_SHOT_MESSAGES = """
---
Example 1:
Input:
{
    "age": 60,
    "sex": "F",
    "narrative": "Patient has rheumatoid arthritis and has been on prednisone 20mg daily for 6 months. She presents with one day of left ankle pain and swelling. Also complains of malaise. On exam, she has a fever, high heart rate (tachycardia), and the left ankle is red and tender. Blood work shows WBC of 15."
}

Output:
{
  "problem_representation_one_liner": "60-year-old immunocompromised woman with acute monoarticular arthritis and features of systemic inflammatory response syndrome (SIRS).",
  "p_nodes": [
    {
      "id": "p_1",
      "content": "Elderly female",
      "original_text": "60-year-old woman",
      "status": "Present"
    },
    {
      "id": "p_2",
      "content": "Immunocompromised state",
      "original_text": "on prednisone 20mg daily for the past 6 months",
      "status": "Present"
    },
    {
      "id": "p_3",
      "content": "History of Rheumatoid Arthritis",
      "original_text": "rheumatoid arthritis",
      "status": "Present"
    },
    {
      "id": "p_4",
      "content": "Acute onset",
      "original_text": "one day",
      "status": "Present"
    },
    {
      "id": "p_5",
      "content": "Monoarticular arthritis (Left Ankle)",
      "original_text": "left ankle pain and swelling... edema, erythema, and tenderness",
      "status": "Present"
    },
    {
      "id": "p_6",
      "content": "SIRS (Systemic Inflammatory Response Syndrome)",
      "original_text": "febrile and tachycardic... WBC of 15",
      "status": "Present"
    }
  ]
}

---
Example 2:
Input:
{
    "age": 43,
    "sex": "F",
    "narrative": "Patient presented with abdominal pain. Illness began 1 week ago with fevers to 38.9 and intermittent frontal headache. No photophobia. Two days ago began having sharp LUQ pain radiating to the back. Reports nausea and vomiting. Denies hematemesis or diarrhea."
}

Output:
{
  "problem_representation_one_liner": "Middle-aged woman with subacute febrile illness, headache, and acute progressive LUQ abdominal pain.",
  "p_nodes": [
    {
      "id": "p_1",
      "content": "Middle-aged female",
      "original_text": "43-year-old... woman",
      "status": "Present"
    },
    {
      "id": "p_2",
      "content": "Subacute febrile illness",
      "original_text": "begun 1 week earlier with fevers",
      "status": "Present"
    },
    {
      "id": "p_3",
      "content": "Frontal headache",
      "original_text": "intermittent frontal headache",
      "status": "Present"
    },
    {
      "id": "p_4",
      "content": "Photophobia",
      "original_text": "without photophobia",
      "status": "Absent"
    },
    {
      "id": "p_5",
      "content": "Acute LUQ abdominal pain",
      "original_text": "Two days prior... left upper quadrant (LUQ) abdominal pain",
      "status": "Present"
    },
    {
      "id": "p_6",
      "content": "Radiation to back/epigastrium",
      "original_text": "radiating to the midepigastrium... and left flank",
      "status": "Present"
    },
    {
      "id": "p_7",
      "content": "Nausea and Vomiting",
      "original_text": "episodic nausea and vomiting",
      "status": "Present"
    },
    {
      "id": "p_8",
      "content": "Hematemesis",
      "original_text": "Denies hematemesis",
      "status": "Absent"
    },
    {
      "id": "p_9",
      "content": "Diarrhea",
      "original_text": "Denies... diarrhea",
      "status": "Absent"
    }
  ]
}
---
"""

PHASE1_TRACKB_USER_PROMPT_TEMPLATE = """
Now, perform the Problem Representation for the following new case.
Remember to translate the raw text into **Medical Semantic Qualifiers** (e.g., "Acute", "Pleuritic", "Dyspnea").

Input:
{{
    "age": {age},
    "sex": "{sex}",
    "narrative": "{narrative}"
}}

Output JSON:
"""