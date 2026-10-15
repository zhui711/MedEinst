"""Analytic pathway for objective patient observations."""
import json
import re
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.json_utils import parse_json_from_text, robust_parse_p_nodes, extract_json_field
from config.prompt_phase1_trackB import (
    PHASE1_TRACKB_SYSTEM_PROMPT,
    PHASE1_TRACKB_FEW_SHOT_MESSAGES,
    PHASE1_TRACKB_USER_PROMPT_TEMPLATE
)


class Phase1TrackBAgent:
    """Extract an objective problem representation and patient nodes."""

    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):

        self.llm_client = llm_client
        self.model_name = model_name

    def _extract_age_sex_from_narrative(self, narrative: str) -> tuple:

        age = 0
        age_patterns = [
            r'(\d+)[\s-]*year[\s-]*old',
            r'(\d+)[\s-]*yo\b',
            r'age[\s:]*(\d+)',
            r'aged[\s:]*(\d+)',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                break

        sex = "Unknown"
        sex_patterns = [
            (r'\b(male|man|boy|gentleman)\b', 'M'),
            (r'\b(female|woman|girl|lady)\b', 'F'),
            (r'\bsex[\s:]*["\']?(M|F|male|female)["\']?', None),
        ]

        for pattern, default_sex in sex_patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                if default_sex:
                    sex = default_sex
                else:
                    matched_sex = match.group(1).upper()
                    sex = 'M' if matched_sex in ['M', 'MALE'] else 'F'
                break

        return age, sex

    def process(
        self,
        raw_narrative: str,
        age: int = None,
        sex: str = None,
        global_hint: Optional[str] = None
    ) -> Dict[str, Any]:

        print("[Phase1 Track B] Starting Problem Representation...")

        if age is None or sex is None:
            extracted_age, extracted_sex = self._extract_age_sex_from_narrative(raw_narrative)
            age = age if age is not None else extracted_age
            sex = sex if sex is not None else extracted_sex

        system_prompt = PHASE1_TRACKB_SYSTEM_PROMPT
        if global_hint and global_hint.strip():
            system_prompt = system_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        messages.append({
            "role": "user",
            "content": f"Here are some examples of Problem Representation:\n{PHASE1_TRACKB_FEW_SHOT_MESSAGES}"
        })
        messages.append({
            "role": "assistant",
            "content": "I understand. I will perform Problem Representation on the new case, translating raw text into Medical Semantic Qualifiers and outputting structured P-Nodes."
        })

        user_prompt = PHASE1_TRACKB_USER_PROMPT_TEMPLATE.format(
            age=age,
            sex=sex,
            narrative=raw_narrative
        )
        messages.append({"role": "user", "content": user_prompt})

        result = self.llm_client.generate_json(
            messages=messages,
            model=self.model_name,
            logprobs=False,
            temperature=0.2,
            max_tokens=4096
        )

        if result["error"]:
            print(f"[Phase1 Track B] LLM error: {result['error']}")
            return {
                "problem_representation_one_liner": "",
                "p_nodes": [],
                "error": result["error"]
            }

        response_content = result["content"]
        print(f"[Phase1 Track B] LLM response length: {len(response_content)} chars")

        parsed_json = parse_json_from_text(response_content, verbose=True)

        p_nodes = []
        one_liner = ""

        if parsed_json is not None and isinstance(parsed_json, dict):
            p_nodes = parsed_json.get("p_nodes", [])
            one_liner = parsed_json.get("problem_representation_one_liner", "")

        if not p_nodes:
            print(f"[Phase1 Track B] Trying robust P-Nodes extraction...")
            p_nodes = robust_parse_p_nodes(response_content, verbose=True)

        if not one_liner:
            one_liner = extract_json_field(response_content, "problem_representation_one_liner", verbose=False)
            if one_liner:
                print(f"[Phase1 Track B] Extracted one_liner via field extraction")
            else:
                one_liner = ""

        if not p_nodes:
            print(f"[Phase1 Track B] Failed to extract any P-Nodes")
            print(f"[Phase1 Track B] Raw response preview: {response_content[:500]}...")
            return {
                "problem_representation_one_liner": one_liner,
                "p_nodes": [],
                "error": "Failed to parse JSON from response"
            }

        p_nodes = self._validate_p_nodes(p_nodes)

        print(f"[Phase1 Track B] Extracted {len(p_nodes)} P-Nodes")
        print(f"[Phase1 Track B] One-liner: {one_liner[:100]}...")

        return {
            "problem_representation_one_liner": one_liner,
            "p_nodes": p_nodes,
            "error": None
        }

    def _validate_p_nodes(self, p_nodes_raw: List[Any]) -> List[Dict[str, Any]]:

        validated_nodes = []

        if not isinstance(p_nodes_raw, list):
            return []

        for idx, node in enumerate(p_nodes_raw):
            if not isinstance(node, dict):
                continue

            node_id = node.get("id", f"p_{idx + 1}")
            if not node_id.startswith("p_"):
                node_id = f"p_{idx + 1}"

            content = node.get("content", "")
            if not content:
                continue

            status = node.get("status", "Present")
            if status not in ["Present", "Absent"]:
                status = "Present"

            original_text = node.get("original_text", "")

            validated_nodes.append({
                "id": node_id,
                "content": content,
                "original_text": original_text,
                "status": status
            })

        return validated_nodes

