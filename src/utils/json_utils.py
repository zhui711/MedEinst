"""Parse structured responses from diagnostic model calls."""
import json
import re
from typing import Dict, Any, Optional, List, Union


def _clean_llm_output(text: str) -> str:

    if not text:
        return text

    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)

    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text, flags=re.IGNORECASE)

    text = re.sub(r'<\?xml[^>]*\?>', '', text)

    return text.strip()

def _fix_common_json_issues(json_str: str) -> str:

    if not json_str:
        return json_str

    json_str = json_str.strip().lstrip('\ufeff')

    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    json_str = re.sub(r':\s*None\b', ': null', json_str)
    json_str = re.sub(r':\s*True\b', ': true', json_str)
    json_str = re.sub(r':\s*False\b', ': false', json_str)

    json_str = re.sub(r'}\s*{', '}, {', json_str)

    json_str = re.sub(r']\s*\[', '], [', json_str)

    json_str = re.sub(r'"\s*\n\s*"([a-zA-Z_])', r'",\n"\1', json_str)

    json_str = re.sub(r'(true|false|null)\s*\n\s*"([a-zA-Z_])', r'\1,\n"\2', json_str)

    json_str = re.sub(r'(\d)\s*\n\s*"([a-zA-Z_])', r'\1,\n"\2', json_str)

    json_str = re.sub(r'}\s*\n\s*"([a-zA-Z_])', r'},\n"\1', json_str)
    json_str = re.sub(r']\s*\n\s*"([a-zA-Z_])', r'],\n"\1', json_str)

    def fix_string_value(match):
        value = match.group(1)

        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        value = value.replace('\t', '\\t')

        value = re.sub(r'[\x00-\x1f\x7f]', '', value)
        return f'"{value}"'

    try:
        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string_value, json_str)
    except:
        pass

    return json_str

def _try_fix_single_quotes(json_str: str) -> Optional[Any]:

    try:

        fixed = json_str.replace("'", '"')
        return json.loads(fixed)
    except:
        return None

def _extract_json_braces(text: str, start_char: str = '{', end_char: str = '}') -> Optional[str]:

    start_idx = text.find(start_char)
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]

    return None

def _extract_all_json_candidates(text: str) -> List[str]:

    candidates = []

    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    for block in code_blocks:
        block = block.strip()
        if block.startswith(('{', '[')):
            candidates.append(block)

    array_json = _extract_json_braces(text, '[', ']')
    if array_json:
        candidates.append(array_json)

    obj_json = _extract_json_braces(text, '{', '}')
    if obj_json:
        candidates.append(obj_json)

    return candidates

def _try_parse_json(json_str: str, verbose: bool = False) -> Optional[Any]:

    if not json_str:
        return None

    last_error = None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        last_error = e

    fixed_str = _fix_common_json_issues(json_str)
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError as e:
        last_error = e

    result = _try_fix_single_quotes(fixed_str)
    if result is not None:
        return result

    truncation_fixes = [

        '}', ']', ']}', ']}',

        '"}', '"]', '"]}', '"]}',
        '1"]', '1"]}', '1"]}',
        '"]]}', '"]}]', '"]}}',

        '"]', '"]}', '"]}}', '"]}]',
        '1"]', '1"]}', '1"]}}',
        '3"]', '3"]}', '3"]}}',  # d_03
        '4"]', '4"]}', '4"]}}',  # d_24, d_44
        '0"]', '0"]}', '0"]}}',  # d_10, d_40
    ]

    for suffix in truncation_fixes:
        try:
            result = json.loads(fixed_str + suffix)
            print(f"[JSON Parser] Recovered truncated JSON with suffix: '{suffix}'")
            return result
        except:
            pass

    try:

        last_complete = fixed_str.rfind('},')
        if last_complete > 0:
            truncated = fixed_str[:last_complete + 1] + ']}'
            result = json.loads(truncated)
            print(f"[JSON Parser] Recovered by truncating to last complete element")
            return result
    except Exception:
        pass

    try:

        k_nodes_start = fixed_str.find('"k_nodes"')
        if k_nodes_start > 0:

            array_start = fixed_str.find('[', k_nodes_start)
            if array_start > 0:

                last_brace = -1
                in_string = False
                escape_next = False
                for i in range(array_start, len(fixed_str)):
                    char = fixed_str[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if not in_string and char == '}':
                        last_brace = i

                if last_brace > array_start:

                    truncated = fixed_str[:last_brace + 1] + ']}'
                    result = json.loads(truncated)
                    k_count = len(result.get('k_nodes', []))
                    print(f"[JSON Parser] Recovered k_nodes array (found {k_count} complete nodes)")
                    return result
    except Exception:
        pass

    try:

        import re
        k_node_pattern = r'\{\s*"content"\s*:\s*"[^"]*"[^}]*"supported_candidates"\s*:\s*\[[^\]]*\][^}]*\}'
        matches = re.findall(k_node_pattern, fixed_str, re.DOTALL)
        if matches:
            k_nodes = []
            for match in matches:
                try:
                    node = json.loads(match)
                    k_nodes.append(node)
                except:
                    continue
            if k_nodes:
                print(f"[JSON Parser] Recovered {len(k_nodes)} k_nodes via regex extraction")
                return {"k_nodes": k_nodes, "edges": []}
    except Exception:
        pass

    if verbose:
        print(f"[JSON Parser] All parse attempts failed for: {json_str[:100]}...")
        if last_error:
            print(f"[JSON Parser] Last error: {last_error}")

    return None

def parse_json_from_text(text: str, verbose: bool = True) -> Optional[Union[Dict[str, Any], List]]:

    if not text or not isinstance(text, str):
        if verbose:
            print("[JSON Parser] Empty or invalid input")
        return None

    original_text = text

    text = _clean_llm_output(text)
    text = text.strip()

    if not text:
        if verbose:
            print("[JSON Parser] Text is empty after cleaning LLM tags")
        return None

    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed from ```json block ({len(json_str)} chars)")
            return result

    json_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1).strip()
        if json_str.startswith(('{', '[')):
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed from ``` block ({len(json_str)} chars)")
                return result

    stripped_text = text.lstrip()
    if stripped_text.startswith('['):

        json_str = _extract_json_braces(stripped_text, '[', ']')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON array ({len(json_str)} chars)")
                return result

    if stripped_text.startswith('{'):
        json_str = _extract_json_braces(stripped_text, '{', '}')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if result is not None:
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON object ({len(json_str)} chars)")
                return result

    json_str = _extract_json_braces(text, '{', '}')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON object ({len(json_str)} chars)")
            return result

    json_str = _extract_json_braces(text, '[', ']')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if result is not None:
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON array ({len(json_str)} chars)")
            return result

    result = _try_parse_json(text, verbose)
    if result is not None:
        if verbose:
            print(f"[JSON Parser] Successfully parsed entire text ({len(text)} chars)")
        return result

    if verbose:
        print(f"[JSON Parser] All parsing methods failed")
        print(f"[JSON Parser] Original text length: {len(original_text)}")
        print(f"[JSON Parser] Text preview (first 500 chars):")
        print(f"  {original_text[:500]}")
        if len(original_text) > 500:
            print(f"  ... (truncated)")

    return None

def parse_json_array_from_text(text: str, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:

    if not text or not isinstance(text, str):
        if verbose:
            print("[JSON Parser] Empty or invalid input for array parsing")
        return None

    original_text = text
    text = _clean_llm_output(text)
    text = text.strip()

    if not text:
        if verbose:
            print("[JSON Parser] Text is empty after cleaning LLM tags")
        return None

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        if json_str.startswith('['):
            result = _try_parse_json(json_str, verbose)
            if isinstance(result, list):
                if verbose:
                    print(f"[JSON Parser] Successfully parsed array from code block ({len(json_str)} chars)")
                return result

    stripped_text = text.lstrip()
    if stripped_text.startswith('['):
        json_str = _extract_json_braces(stripped_text, '[', ']')
        if json_str:
            result = _try_parse_json(json_str, verbose)
            if isinstance(result, list):
                if verbose:
                    print(f"[JSON Parser] Successfully parsed JSON array directly ({len(json_str)} chars)")
                return result

    json_str = _extract_json_braces(text, '[', ']')
    if json_str:
        result = _try_parse_json(json_str, verbose)
        if isinstance(result, list):
            if verbose:
                print(f"[JSON Parser] Successfully parsed embedded JSON array ({len(json_str)} chars)")
            return result

    parsed = parse_json_from_text(text, verbose=False)

    if isinstance(parsed, list):
        if verbose:
            print(f"[JSON Parser] Parsed as array via general parser")
        return parsed

    if isinstance(parsed, dict):

        array_keys = ['results', 'items', 'data', 'updates', 'nodes', 'edges',
                      'array', 'list', 'content', 'output', 'response', 'p_nodes', 'k_nodes']
        for key in array_keys:
            if key in parsed and isinstance(parsed[key], list):
                if verbose:
                    print(f"[JSON Parser] Extracted array from '{key}' field")
                return parsed[key]

        if len(parsed) == 1:
            only_value = list(parsed.values())[0]
            if isinstance(only_value, list):
                if verbose:
                    print(f"[JSON Parser] Extracted array from single-key dict")
                return only_value

    if verbose:
        print(f"[JSON Parser] Failed to parse as array. Result type: {type(parsed)}")
        print(f"[JSON Parser] Original text preview: {original_text[:300]}...")

    return None

def extract_json_field(text: str, field_name: str, verbose: bool = False) -> Optional[Any]:

    if not text or not field_name:
        return None

    text = _clean_llm_output(text)

    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict) and field_name in parsed:
        return parsed[field_name]

    patterns = [

        rf'"{re.escape(field_name)}"\s*:\s*(\[[^\]]*\])',

        rf'"{re.escape(field_name)}"\s*:\s*(\{{[^\}}]*\}})',

        rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)"',

        rf'"{re.escape(field_name)}"\s*:\s*([\d.]+|true|false|null)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value_str = match.group(1)
            try:
                return json.loads(value_str)
            except:

                return value_str

    if verbose:
        print(f"[JSON Parser] Failed to extract field '{field_name}'")

    return None

def safe_json_loads(json_str: str, default: Any = None, verbose: bool = False) -> Any:

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        if verbose:
            print(f"[JSON Parser] safe_json_loads failed: {e}")
        return default

def format_json_for_prompt(data: Union[Dict, List], indent: int = 2) -> str:

    return json.dumps(data, ensure_ascii=False, indent=indent)

def robust_parse_p_nodes(text: str, verbose: bool = True) -> List[Dict[str, Any]]:

    if not text:
        return []

    text = _clean_llm_output(text)

    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        p_nodes = parsed.get('p_nodes', [])
        if isinstance(p_nodes, list) and len(p_nodes) > 0:
            if verbose:
                print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes from complete JSON")
            return p_nodes

    p_nodes = extract_json_field(text, 'p_nodes', verbose=False)
    if isinstance(p_nodes, list) and len(p_nodes) > 0:
        if verbose:
            print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes via field extraction")
        return p_nodes

    array_result = parse_json_array_from_text(text, verbose=False)
    if array_result and len(array_result) > 0:

        if _looks_like_p_nodes(array_result):
            if verbose:
                print(f"[JSON Parser] Extracted {len(array_result)} P-Nodes from array")
            return array_result

    p_nodes = []

    pattern = r'\{\s*"id"\s*:\s*"p_\d+"[^}]+\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            node = json.loads(match)
            if 'id' in node and 'content' in node:
                p_nodes.append(node)
        except:
            continue

    if p_nodes:
        if verbose:
            print(f"[JSON Parser] Extracted {len(p_nodes)} P-Nodes via regex fallback")
        return p_nodes

    if verbose:
        print(f"[JSON Parser] Failed to extract any P-Nodes")

    return []

def _looks_like_p_nodes(items: List) -> bool:

    if not items:
        return False

    first = items[0]
    if isinstance(first, dict):

        return 'content' in first or 'id' in first

    return False

def robust_parse_track_a_response(text: str, verbose: bool = True) -> Dict[str, Any]:

    result = {
        "top_candidates": [],
        "final_diagnosis_id": None,
        "structured_analysis": None,
        "differential_reasoning": None
    }

    if not text:
        return result

    text = _clean_llm_output(text)

    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        result["top_candidates"] = parsed.get("top_candidates", [])
        result["final_diagnosis_id"] = parsed.get("final_diagnosis_id")
        result["structured_analysis"] = parsed.get("structured_analysis")
        result["differential_reasoning"] = parsed.get("differential_reasoning")

        if result["top_candidates"] and result["final_diagnosis_id"]:
            if verbose:
                print(f"[JSON Parser] Track A: Complete parse success")
            return result

    id_patterns = [
        r'"final_diagnosis_id"\s*:\s*"(\d+)"',
        r'"final_diagnosis_id"\s*:\s*(\d+)',
        r'final_diagnosis_id["\s:]+(\d+)',
    ]

    for pattern in id_patterns:
        match = re.search(pattern, text)
        if match:
            result["final_diagnosis_id"] = match.group(1)
            break

    candidates_patterns = [

        r'"top_candidates"\s*:\s*\[([\s\S]*?)\]',

        r'"top_candidates"\s*:\s*\[([^\]]+)',
    ]

    for pattern in candidates_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            array_content = match.group(1)

            ids = re.findall(r'"(\d{1,2})"', array_content)
            if ids:

                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]
                break

    if not result["top_candidates"]:

        truncated_match = re.search(r'"top\w*"\s*,\s*((?:"[^"]*"\s*,?\s*)+)', text)
        if truncated_match:
            array_content = truncated_match.group(1)
            ids = re.findall(r'"(\d{1,2})"', array_content)
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]

    if not result["top_candidates"]:

        no_bracket_match = re.search(
            r'"top_candidates"\s*:\s*"(\d{1,2})"((?:\s*,\s*"(\d{1,2})")*)',
            text
        )
        if no_bracket_match:
            first_id = no_bracket_match.group(1)
            rest = no_bracket_match.group(2)
            ids = [first_id] + re.findall(r'"(\d{1,2})"', rest)
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]

    if not result["top_candidates"]:

        array_match = re.search(r'\[\s*((?:"[^"]*"\s*,?\s*)+)\s*\]', text)
        if array_match:
            ids = re.findall(r'"(\d{1,2})"', array_match.group(1))
            if ids:
                seen = set()
                unique_ids = []
                for id_val in ids:
                    if id_val not in seen:
                        seen.add(id_val)
                        unique_ids.append(id_val)
                result["top_candidates"] = unique_ids[:5]

    reasoning_patterns = [
        r'"differential_reasoning"\s*:\s*"([\s\S]*?)"(?=\s*,|\s*})',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, text)
        if match:
            result["differential_reasoning"] = match.group(1)
            break

    if verbose:
        if result["top_candidates"] or result["final_diagnosis_id"]:
            print(f"[JSON Parser] Track A: Fallback extraction - "
                  f"candidates: {len(result['top_candidates'])}, "
                  f"diagnosis_id: {result['final_diagnosis_id']}")
        else:
            print(f"[JSON Parser] Track A: Failed to extract key fields")

    return result

def robust_parse_k_gen_response(text: str, verbose: bool = True) -> Dict[str, Any]:

    result = {"k_nodes": [], "edges": []}

    if not text:
        return result

    text = _clean_llm_output(text)

    parsed = parse_json_from_text(text, verbose=False)
    if isinstance(parsed, dict):
        result["k_nodes"] = parsed.get("k_nodes", [])
        result["edges"] = parsed.get("edges", [])

        if result["k_nodes"] or result["edges"]:
            if verbose:
                print(f"[JSON Parser] K-Gen: Extracted {len(result['k_nodes'])} K-Nodes, "
                      f"{len(result['edges'])} edges")
            return result

    k_nodes = extract_json_field(text, 'k_nodes', verbose=False)
    if isinstance(k_nodes, list):
        result["k_nodes"] = k_nodes

    edges = extract_json_field(text, 'edges', verbose=False)
    if isinstance(edges, list):
        result["edges"] = edges

    if result["k_nodes"] or result["edges"]:
        if verbose:
            print(f"[JSON Parser] K-Gen (fallback): Extracted {len(result['k_nodes'])} K-Nodes, "
                  f"{len(result['edges'])} edges")
    else:
        if verbose:
            print(f"[JSON Parser] K-Gen: Failed to extract K-Nodes or edges")

    return result
