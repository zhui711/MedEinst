#!/usr/bin/env python3
"""Refine aggregated disease Illness Graphs with the configured critic."""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
import yaml

from src.utils.api_client import LLMClient
from src.utils.runtime_config import configured_base_url, config_relative_path
from src.graph.canonical_graph import CanonicalGraph, CanonicalKNode, CanonicalEdge
from config.prompt_refinement import (
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_USER_PROMPT_TEMPLATE,
    format_k_nodes_to_markdown,
    validate_clustering_result,
    normalize_clustering_result
)




def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:

    config_path = Path(config_path or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def call_llm_for_clustering(
    client: LLMClient,
    model: str,
    pathology_id: str,
    pathology_name: str,
    k_nodes: List[CanonicalKNode],
    max_retries: int = 3
) -> Tuple[Optional[List[Dict]], str]:

    k_nodes_list = [node.to_dict() for node in k_nodes]
    markdown_list = format_k_nodes_to_markdown(k_nodes_list)

    user_prompt = REFINEMENT_USER_PROMPT_TEMPLATE.format(
        pathology_name=pathology_name,
        pathology_id=pathology_id,
        total_k_nodes=len(k_nodes),
        k_nodes_markdown_list=markdown_list
    )

    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        result = client.generate_json(
            messages=messages,
            model=model,
            logprobs=False,
            temperature=0.0,
            max_tokens=8192,
            max_retries=1
        )

        if result["error"]:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"LLM API Error: {result['error']}"

        content = result["content"]

        try:

            json_str = extract_json_from_response(content)

            merge_clusters = json.loads(json_str)

            is_valid, error_msg = validate_clustering_result(merge_clusters, len(k_nodes))

            if is_valid:

                full_clusters = normalize_clustering_result(merge_clusters, k_nodes_list)
                return full_clusters, ""
            else:
                if attempt < max_retries - 1:
                    print(f"    [Retry {attempt + 1}] Validation failed: {error_msg}")
                    time.sleep(2)
                    continue
                return None, f"Validation Error: {error_msg}"

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"    [Retry {attempt + 1}] JSON parse error: {e}")
                time.sleep(2)
                continue
            return None, f"JSON Parse Error: {str(e)}"

    return None, "Max retries exceeded"

def extract_json_from_response(content: str) -> str:

    content = content.strip()

    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            json_part = parts[1]
            if "```" in json_part:
                return json_part.split("```")[0].strip()

    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:

            return parts[1].strip()

    start_idx = content.find('[')
    if start_idx != -1:

        bracket_count = 0
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return content[start_idx:i+1]

    return content



def reconstruct_graph(
    original_graph: CanonicalGraph,
    clusters: List[Dict],
    k_nodes_list: List[CanonicalKNode]
) -> Tuple[CanonicalGraph, Dict[str, Any]]:

    new_graph = CanonicalGraph(
        pathology_id=original_graph.pathology_id,
        pathology_name=original_graph.pathology_name
    )
    new_graph.version = "1.0-refined"
    new_graph.source_cases = original_graph.source_cases.copy()

    old_to_canonical: Dict[str, str] = {}  # old_content.lower() -> canonical_name
    merged_clusters_record: List[Dict] = []

    importance_priority = {
        "Pathognomonic": 4,
        "Essential": 3,
        "Strong": 2,
        "Common": 1,
        "Weak": 0
    }

    for cluster in clusters:
        canonical_name = cluster["canonical_name"]
        original_indices = cluster["original_indices"]
        rationale = cluster.get("rationale", "")

        merged_nodes = [k_nodes_list[idx - 1] for idx in original_indices]

        total_count = sum(node.occurrence_count for node in merged_nodes)

        max_importance = "Common"
        max_priority = 0
        for node in merged_nodes:
            priority = importance_priority.get(node.importance, 0)
            if priority > max_priority:
                max_priority = priority
                max_importance = node.importance

        has_pivot = any(node.k_type == "Pivot" for node in merged_nodes)
        merged_k_type = "Pivot" if has_pivot else "General"

        all_sources = []
        for node in merged_nodes:
            all_sources.extend(node.original_sources)
        unique_sources = list(set(all_sources))[:5]

        new_k_node = CanonicalKNode(
            content=canonical_name,
            k_type=merged_k_type,
            importance=max_importance,
            occurrence_count=total_count,
            original_sources=unique_sources
        )
        new_graph.k_nodes[canonical_name.lower().strip()] = new_k_node

        merged_contents = []
        for node in merged_nodes:
            old_content_key = node.content.lower().strip()
            old_to_canonical[old_content_key] = canonical_name
            if node.content != canonical_name:
                merged_contents.append(node.content)

        if len(original_indices) > 1:
            merged_clusters_record.append({
                "canonical": canonical_name,
                "merged": merged_contents,
                "original_count": len(original_indices),
                "rationale": rationale
            })

    for p_node in original_graph.p_nodes.values():
        new_graph.p_nodes[p_node.content.lower().strip()] = p_node

    new_edges: Dict[Tuple[str, str, str], CanonicalEdge] = {}

    strength_priority = {
        "Pathognomonic": 4,
        "Essential": 3,
        "Strong": 2,
        "Weak": 1,
        None: 0
    }

    for edge in original_graph.edges:

        if edge.edge_type == "P-K":

            new_source = edge.source_content
            target_key = edge.target_content.lower().strip()
            new_target = old_to_canonical.get(target_key, edge.target_content)
        elif edge.edge_type == "K-D":

            source_key = edge.source_content.lower().strip()
            new_source = old_to_canonical.get(source_key, edge.source_content)
            new_target = edge.target_content
        else:

            new_source = edge.source_content
            new_target = edge.target_content

        edge_key = (
            new_source.lower().strip(),
            new_target.lower().strip(),
            edge.relation
        )

        if edge_key in new_edges:
            existing_edge = new_edges[edge_key]

            existing_edge.occurrence_count += edge.occurrence_count

            if edge.strength:
                existing_priority = strength_priority.get(existing_edge.strength, 0)
                new_priority = strength_priority.get(edge.strength, 0)
                if new_priority > existing_priority:
                    existing_edge.strength = edge.strength
        else:

            new_edges[edge_key] = CanonicalEdge(
                source_content=new_source,
                target_content=new_target,
                edge_type=edge.edge_type,
                relation=edge.relation,
                strength=edge.strength,
                occurrence_count=edge.occurrence_count
            )

    new_graph._edges = new_edges

    stats = {
        "k_nodes_before": len(original_graph.k_nodes),
        "k_nodes_after": len(new_graph.k_nodes),
        "edges_before": len(original_graph.edges),
        "edges_after": len(new_graph.edges),
        "merged_clusters_count": len(merged_clusters_record),
        "merged_clusters": merged_clusters_record
    }

    return new_graph, stats



def add_refinement_metadata(graph_dict: Dict, stats: Dict, original_k_nodes_count: int) -> Dict:

    graph_dict["metadata"]["refinement_version"] = "1.0"
    graph_dict["metadata"]["refinement_date"] = datetime.now().strftime("%Y-%m-%d")
    graph_dict["metadata"]["original_k_nodes_count"] = original_k_nodes_count
    graph_dict["metadata"]["merged_clusters"] = stats["merged_clusters"]

    graph_dict["statistics"]["original_k_nodes_count"] = original_k_nodes_count
    graph_dict["statistics"]["merged_k_nodes_count"] = stats["merged_clusters_count"]
    graph_dict["statistics"]["compression_ratio"] = round(
        stats["k_nodes_after"] / stats["k_nodes_before"] if stats["k_nodes_before"] > 0 else 1.0,
        3
    )

    return graph_dict



def process_single_graph(
    filepath: Path,
    output_dir: Path,
    client: LLMClient,
    model: str,
    dry_run: bool = False
) -> Dict[str, Any]:

    filename = filepath.stem  # e.g., "01_Possible_NSTEMI___STEMI_GG"
    start_time = time.time()

    log_entry = {
        "filename": filename,
        "status": "pending",
        "k_nodes_before": 0,
        "k_nodes_after": 0,
        "merged_clusters_count": 0,
        "processing_time_sec": 0,
        "error_msg": ""
    }

    try:

        print(f"  [1/5] Loading {filepath.name}...")
        original_graph = CanonicalGraph.load(filepath)

        k_nodes_list = list(original_graph.k_nodes.values())
        log_entry["k_nodes_before"] = len(k_nodes_list)

        print(f"        → Loaded: {len(k_nodes_list)} K-Nodes, {len(original_graph.p_nodes)} P-Nodes, {len(original_graph.edges)} Edges")

        if dry_run:
            log_entry["status"] = "dry_run"
            log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
            return log_entry

        print(f"  [2/5] Calling LLM for clustering...")
        clusters, error = call_llm_for_clustering(
            client=client,
            model=model,
            pathology_id=original_graph.pathology_id,
            pathology_name=original_graph.pathology_name,
            k_nodes=k_nodes_list
        )

        if error:
            log_entry["status"] = "failed"
            log_entry["error_msg"] = error
            log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
            print(f"        ✗ LLM Error: {error}")
            return log_entry

        merge_count = sum(1 for c in clusters if len(c.get("original_indices", [])) > 1)
        print(f"        → LLM returned {len(clusters)} clusters ({merge_count} merges)")

        print(f"  [3/5] Reconstructing graph...")
        new_graph, stats = reconstruct_graph(original_graph, clusters, k_nodes_list)

        log_entry["k_nodes_after"] = stats["k_nodes_after"]
        log_entry["merged_clusters_count"] = stats["merged_clusters_count"]

        print(f"        → K-Nodes: {stats['k_nodes_before']} → {stats['k_nodes_after']} "
              f"(reduced by {stats['k_nodes_before'] - stats['k_nodes_after']})")
        print(f"        → Edges: {stats['edges_before']} → {stats['edges_after']}")

        print(f"  [4/5] Adding metadata...")
        graph_dict = new_graph.to_dict()
        graph_dict = add_refinement_metadata(graph_dict, stats, len(k_nodes_list))

        parts = filename.replace("_GG", "").split("_", 1)
        if len(parts) >= 2:
            pathology_id = parts[0]
            pathology_name = parts[1]
        else:
            pathology_id = original_graph.pathology_id
            pathology_name = original_graph.pathology_name.replace(" ", "_").replace("/", "_")

        output_filename = f"{pathology_id}_{pathology_name}_Refined_GG.json"
        output_path = output_dir / output_filename

        print(f"  [5/5] Saving to {output_path.name}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)

        log_entry["status"] = "success"
        log_entry["output_file"] = str(output_path)
        log_entry["processing_time_sec"] = round(time.time() - start_time, 2)

        print(f"        ✓ Saved successfully! ({log_entry['processing_time_sec']}s)")

        return log_entry

    except Exception as e:
        import traceback
        log_entry["status"] = "failed"
        log_entry["error_msg"] = f"{type(e).__name__}: {str(e)}"
        log_entry["traceback"] = traceback.format_exc()
        log_entry["processing_time_sec"] = round(time.time() - start_time, 2)
        print(f"        ✗ Exception: {log_entry['error_msg']}")
        return log_entry



def main():

    parser = argparse.ArgumentParser(
        description="Refine Illness Graph knowledge nodes with the configured critic"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the processing plan without executing"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process only a single pathology by ID (e.g., --single 01)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Raw Illness Graph directory (overrides offline_training.golden_graph_dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Refined Illness Graph output directory (overrides online_inference.refined_graph_dir)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML configuration file (default: config/settings.yaml)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Refinement log directory (overrides offline_training.log_dir)")

    args = parser.parse_args()

    config_path = Path(args.config or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    config = load_config(config_path)
    offline_config = config.get("offline_training", {})
    online_config = config.get("online_inference", {})

    def runtime_path(cli_value: Optional[str], configured: str) -> Path:
        if cli_value:
            return Path(cli_value).expanduser().resolve()
        return Path(config_relative_path(configured, config_path))

    input_dir = runtime_path(args.input_dir, offline_config.get("golden_graph_dir", "../golden_graphs"))
    output_dir = runtime_path(args.output_dir, online_config.get("refined_graph_dir", "../golden_graphs_refined"))
    log_dir = runtime_path(args.log_dir, offline_config.get("log_dir", "../logs"))

    print("=" * 70)
    print("Illness Graph Refinement")
    print("=" * 70)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Dry Run Mode:     {args.dry_run}")
    print("=" * 70)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    gg_files = sorted(input_dir.glob("*_GG.json"))

    if args.single:

        gg_files = [f for f in gg_files if f.stem.startswith(f"{args.single}_")]
        if not gg_files:
            print(f"Error: No Illness Graph found for ID '{args.single}'")
            sys.exit(1)

    print(f"\nFound {len(gg_files)} Illness Graph(s) to process:\n")
    for f in gg_files:
        print(f"  - {f.name}")
    print()

    if args.dry_run:
        print("[DRY RUN MODE] No actual processing will be performed.\n")

    critical_config = config.get("critical_model", {})

    client = LLMClient(
        base_url=configured_base_url(critical_config),
        api_key_env=critical_config.get("api_key_env", "OPENAI_API_KEY"),
        timeout=critical_config.get("timeout", 180)
    )
    model = critical_config.get("model_name", "gpt-4o")

    print(f"LLM Model: {model}")
    print(f"API Base:  {configured_base_url(critical_config) or 'official OpenAI API'}")
    print("=" * 70)

    all_logs = {}
    total_start_time = time.time()

    for filepath in tqdm(gg_files, desc="Processing Illness Graphs", unit="file"):
        print(f"\n[Processing] {filepath.name}")
        print("-" * 60)

        log_entry = process_single_graph(
            filepath=filepath,
            output_dir=output_dir,
            client=client,
            model=model,
            dry_run=args.dry_run
        )

        all_logs[filepath.stem] = log_entry

    total_time = round(time.time() - total_start_time, 2)

    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)

    success_count = sum(1 for log in all_logs.values() if log["status"] == "success")
    failed_count = sum(1 for log in all_logs.values() if log["status"] == "failed")

    print(f"Total Processed:  {len(all_logs)}")
    print(f"  - Success:      {success_count}")
    print(f"  - Failed:       {failed_count}")
    print(f"  - Dry Run:      {sum(1 for log in all_logs.values() if log['status'] == 'dry_run')}")
    print(f"Total Time:       {total_time}s")

    total_before = sum(log.get("k_nodes_before", 0) for log in all_logs.values())
    total_after = sum(log.get("k_nodes_after", 0) for log in all_logs.values() if log["status"] == "success")
    total_merged = sum(log.get("merged_clusters_count", 0) for log in all_logs.values())

    print(f"\nK-Nodes Reduction:")
    print(f"  - Before:       {total_before}")
    print(f"  - After:        {total_after}")
    print(f"  - Reduced:      {total_before - total_after} ({round((1 - total_after/total_before)*100, 1) if total_before > 0 else 0}%)")
    print(f"  - Total Merges: {total_merged}")

    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "refinement_log.json"

        log_output = {
            "run_timestamp": datetime.now().isoformat(),
            "total_files": len(all_logs),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_processing_time_sec": total_time,
            "model_used": model,
            "details": all_logs
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_output, f, indent=2, ensure_ascii=False)

        print(f"\nLog saved to: {log_path}")

    print("\n" + "=" * 70)
    print("Refinement Pipeline Complete!")
    print("=" * 70)

    if failed_count > 0:
        print(f"\nWarning: {failed_count} file(s) failed to process. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

