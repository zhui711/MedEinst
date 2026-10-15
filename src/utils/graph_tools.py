"""Score and summarize diagnostic causal graphs."""
from typing import Dict, Any, List, Tuple, Optional
from src.graph.schema import MedicalGraph


def rebuild_graph_from_json(graph_json: Dict[str, Any]) -> MedicalGraph:

    graph = MedicalGraph()

    case_metadata = graph_json.get("case_metadata", {})
    graph.meta["raw_text"] = case_metadata.get("raw_text", "")
    graph.meta["case_id"] = case_metadata.get("case_id", "")

    phase1_context = graph_json.get("phase1_context", {})
    graph.meta["initial_candidates"] = phase1_context.get("initial_candidates", [])
    graph.meta["initial_reasoning"] = phase1_context.get("initial_reasoning", "")

    nodes = graph_json.get("graph", {}).get("nodes", {})
    edges = graph_json.get("graph", {}).get("edges", {})

    p_nodes = nodes.get("p_nodes", [])
    k_nodes = nodes.get("k_nodes", [])
    d_nodes = nodes.get("d_nodes", [])
    p_k_links = edges.get("p_k_links", [])
    k_d_links = edges.get("k_d_links", [])

    for p_data in p_nodes:
        p_id = p_data.get("id")
        if p_id:
            graph.graph.add_node(
                p_id,
                type="P",
                content=p_data.get("content", ""),
                original_text=p_data.get("original_text", ""),
                status=p_data.get("status", "Present"),
                source=p_data.get("source", "Unknown")
            )

    for k_data in k_nodes:
        k_id = k_data.get("id")
        if k_id:
            graph.graph.add_node(
                k_id,
                type="K",
                content=k_data.get("content", ""),
                k_type=k_data.get("k_type", "General"),
                source=k_data.get("source", "Unknown"),
                importance=k_data.get("importance", "Common")
            )

    for d_data in d_nodes:
        d_id = d_data.get("id")
        if d_id:
            graph.graph.add_node(
                d_id,
                type="D",
                name=d_data.get("name", ""),
                original_id=d_data.get("original_id", ""),
                initial_rank=d_data.get("initial_rank", 999),
                risk_level=d_data.get("risk_level"),
                state=d_data.get("state", "Active")
            )

    for pk_link in p_k_links:
        source = pk_link.get("source")
        target = pk_link.get("target")
        relation = pk_link.get("relation", "")
        if source and target:
            graph.graph.add_edge(source, target, relation=relation)

    for kd_link in k_d_links:
        source = kd_link.get("source")
        target = kd_link.get("target")
        relation = kd_link.get("relation", "Support")
        strength = kd_link.get("strength", "Weak")
        if source and target:
            graph.graph.add_edge(source, target, relation=relation, strength=strength)

    return graph

def serialize_graph_to_summary(graph: MedicalGraph) -> str:

    lines = []
    lines.append("=" * 60)
    lines.append("EVIDENCE SUMMARY BY CANDIDATE")
    lines.append("=" * 60)
    lines.append("")

    d_nodes = graph.get_d_nodes()
    d_nodes_sorted = sorted(d_nodes, key=lambda d: d.get("initial_rank", 999))

    k_nodes = graph.get_k_nodes()
    k_node_map = {k["id"]: k for k in k_nodes}

    p_nodes = graph.get_p_nodes()
    p_node_map = {p["id"]: p for p in p_nodes}

    p_k_edges = []  # (p_id, k_id, relation)
    k_d_edges = []  # (k_id, d_id, relation, strength)

    for source, target, data in graph.graph.edges(data=True):
        source_data = graph.graph.nodes.get(source, {})
        target_data = graph.graph.nodes.get(target, {})

        if source_data.get("type") == "P" and target_data.get("type") == "K":
            p_k_edges.append((source, target, data.get("relation", "")))
        elif source_data.get("type") == "K" and target_data.get("type") == "D":
            k_d_edges.append((source, target, data.get("relation", ""), data.get("strength", "")))

    for d_node in d_nodes_sorted:
        d_id = d_node["id"]
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")
        d_state = d_node.get("state", "Active")

        state_marker = "🔴 PRUNED" if d_state == "Pruned" else ""

        lines.append(f"### Candidate: {d_name} ({d_id}) [Rank: {d_rank}] {state_marker}")
        lines.append("")

        related_k_ids = [k_id for k_id, d_id_target, _, _ in k_d_edges if d_id_target == d_id]

        supporting = []
        conflicting = []
        missing = []

        general_k_count = 0
        pivot_k_count = 0

        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", "Unknown")
            k_type = k_node.get("k_type", "General")
            k_importance = k_node.get("importance", "Common")

            if k_type == "General":
                general_k_count += 1
            else:
                pivot_k_count += 1

            pk_relation = "Unknown"
            p_content = ""
            for p_id, k_id_target, relation in p_k_edges:
                if k_id_target == k_id:
                    pk_relation = relation
                    p_node = p_node_map.get(p_id, {})
                    p_content = p_node.get("content", "")
                    break

            importance_marker = f"[{k_importance}]" if k_importance in ["Essential", "Pathognomonic"] else ""
            evidence_desc = f"{k_content} {importance_marker}".strip()

            if pk_relation == "Match":
                supporting.append(evidence_desc)
            elif pk_relation == "Conflict":
                conflicting.append(evidence_desc)
            elif pk_relation == "Void":
                missing.append(evidence_desc)

        if supporting:
            lines.append(f"**✅ Supporting Evidence ({len(supporting)}):**")
            for ev in supporting[:10]:
                lines.append(f"  - {ev}")
            if len(supporting) > 10:
                lines.append(f"  - ... and {len(supporting) - 10} more")
            lines.append("")

        if conflicting:
            lines.append(f"**❌ Conflicting Evidence ({len(conflicting)}):**")
            for ev in conflicting[:10]:
                lines.append(f"  - {ev}")
            if len(conflicting) > 10:
                lines.append(f"  - ... and {len(conflicting) - 10} more")
            lines.append("")

        if missing:
            lines.append(f"**❓ Missing/Shadow Evidence ({len(missing)}):**")
            for ev in missing[:10]:
                lines.append(f"  - {ev}")
            if len(missing) > 10:
                lines.append(f"  - ... and {len(missing) - 10} more")
            lines.append("")

        lines.append(f"**📊 Statistics:**")
        lines.append(f"  - Knowledge Sources: {general_k_count} General, {pivot_k_count} Pivot")
        lines.append(f"  - Evidence Counts: {len(supporting)} Match, {len(conflicting)} Conflict, {len(missing)} Shadow")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)

def calculate_deterministic_score(graph: MedicalGraph) -> Dict[str, float]:

    scores = {}

    d_nodes = graph.get_d_nodes()

    p_k_edges = {}  # k_id -> (p_id, relation)
    k_d_edges = {}  # d_id -> [k_id, ...]

    for source, target, data in graph.graph.edges(data=True):
        source_data = graph.graph.nodes.get(source, {})
        target_data = graph.graph.nodes.get(target, {})

        if source_data.get("type") == "P" and target_data.get("type") == "K":

            if target not in p_k_edges:
                p_k_edges[target] = []
            p_k_edges[target].append((source, data.get("relation", "")))
        elif source_data.get("type") == "K" and target_data.get("type") == "D":

            if target not in k_d_edges:
                k_d_edges[target] = []
            k_d_edges[target].append(source)

    for d_node in d_nodes:
        d_id = d_node["id"]

        match_count = 0
        conflict_count = 0
        shadow_count = 0

        related_k_ids = k_d_edges.get(d_id, [])

        for k_id in related_k_ids:

            pk_relations = p_k_edges.get(k_id, [])
            for _, relation in pk_relations:
                if relation == "Match":
                    match_count += 1
                elif relation == "Conflict":
                    conflict_count += 1
                elif relation == "Void":
                    shadow_count += 1

        score = (match_count * 1.0) - (conflict_count * 1.5) - (shadow_count * 0.1)
        scores[d_id] = round(score, 2)

    return scores

def get_evidence_breakdown(graph: MedicalGraph, d_id: str) -> Dict[str, Any]:

    result = {
        "match_count": 0,
        "conflict_count": 0,
        "shadow_count": 0,
        "match_evidence": [],
        "conflict_evidence": [],
        "shadow_evidence": []
    }

    k_nodes_with_edges = graph.get_k_nodes_for_d(d_id)

    for k_node, kd_edge in k_nodes_with_edges:
        k_id = k_node["id"]
        k_content = k_node.get("content", "")

        p_nodes_with_edges = graph.get_p_nodes_for_k(k_id)

        for p_node, pk_edge in p_nodes_with_edges:
            relation = pk_edge.get("relation", "")
            p_content = p_node.get("content", "")

            evidence_item = {
                "k_content": k_content,
                "p_content": p_content,
                "importance": k_node.get("importance", "Common")
            }

            if relation == "Match":
                result["match_count"] += 1
                result["match_evidence"].append(evidence_item)
            elif relation == "Conflict":
                result["conflict_count"] += 1
                result["conflict_evidence"].append(evidence_item)
            elif relation == "Void":
                result["shadow_count"] += 1
                result["shadow_evidence"].append(evidence_item)

    return result

def summarize_graph_for_critique(
    graph_json: Dict[str, Any],
    ground_truth_id: str
) -> str:

    from collections import defaultdict

    lines = []
    lines.append("=" * 60)
    lines.append("DIAGNOSTIC REASONING ANALYSIS (For Teacher Review)")
    lines.append("=" * 60)

    graph_data = graph_json.get("graph", {})
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", {})

    k_node_map = {k["id"]: k for k in nodes.get("k_nodes", [])}
    p_node_map = {p["id"]: p for p in nodes.get("p_nodes", [])}
    d_node_map = {d["id"]: d for d in nodes.get("d_nodes", [])}

    pk_by_k = defaultdict(list)  # k_id -> [(p_id, relation), ...]
    kd_by_d = defaultdict(list)  # d_id -> [(k_id, relation, strength), ...]

    for edge in edges.get("p_k_links", []):
        pk_by_k[edge.get("target", "")].append(
            (edge.get("source", ""), edge.get("relation", ""))
        )

    for edge in edges.get("k_d_links", []):
        kd_by_d[edge.get("target", "")].append(
            (edge.get("source", ""), edge.get("relation", ""), edge.get("strength", ""))
        )

    # === Section 1: Missing Evidence (Shadow Nodes / Void Relations) ===
    lines.append("\n### 🔍 MISSING EVIDENCE (Shadow Nodes / Void Relations)")
    lines.append("These are clinical features the agent could not find in the patient narrative:")

    shadow_count = 0
    for k_id, pk_list in pk_by_k.items():
        for p_id, relation in pk_list:
            if relation == "Void":
                k_node = k_node_map.get(k_id, {})
                k_content = k_node.get("content", k_id)
                importance = k_node.get("importance", "Common")
                k_type = k_node.get("k_type", "General")

                marker = ""
                if importance in ["Essential", "Pathognomonic"]:
                    marker = "⚠️ CRITICAL"
                elif k_type == "Pivot":
                    marker = "📌 PIVOT"

                lines.append(f"  - {k_content} [{importance}, {k_type}] {marker}")
                shadow_count += 1

    if shadow_count == 0:
        lines.append("  (No shadow nodes found - all features were matched)")
    else:
        lines.append(f"\n  Total Shadow Nodes: {shadow_count}")

    # === Section 2: Conflicting Evidence ===
    lines.append("\n### ❌ CONFLICTING EVIDENCE")
    lines.append("These are features where patient status contradicts expected findings:")

    conflict_count = 0
    for k_id, pk_list in pk_by_k.items():
        for p_id, relation in pk_list:
            if relation == "Conflict":
                k_node = k_node_map.get(k_id, {})
                p_node = p_node_map.get(p_id, {})

                k_content = k_node.get("content", k_id)
                p_content = p_node.get("content", p_id)
                p_status = p_node.get("status", "Unknown")

                lines.append(
                    f"  - K-Node: '{k_content}' "
                    f"<-- P-Node: '{p_content}' (Status: {p_status})"
                )
                conflict_count += 1

    if conflict_count == 0:
        lines.append("  (No conflicts found)")
    else:
        lines.append(f"\n  Total Conflicts: {conflict_count}")

    # === Section 3: Ground Truth Analysis ===

    gt_d_id = f"d_{ground_truth_id}" if not ground_truth_id.startswith("d_") else ground_truth_id
    gt_name = d_node_map.get(gt_d_id, {}).get("name", "Unknown")

    lines.append(f"\n### 🎯 GROUND TRUTH ANALYSIS: {gt_name} ({gt_d_id})")

    gt_evidence = kd_by_d.get(gt_d_id, [])
    if gt_evidence:
        lines.append(f"  Evidence supporting the CORRECT diagnosis:")

        support_list = []
        rule_out_list = []

        for k_id, relation, strength in gt_evidence:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", k_id)
            importance = k_node.get("importance", "Common")

            pk_relations = pk_by_k.get(k_id, [])
            pk_summary = []
            for p_id, pk_rel in pk_relations:
                pk_summary.append(pk_rel)
            pk_str = ", ".join(set(pk_summary)) if pk_summary else "No P-K links"

            if relation == "Support":
                support_list.append(f"    ✓ {k_content} [{strength}] (P-K: {pk_str})")
            elif relation == "Rule_Out":
                rule_out_list.append(f"    ✗ {k_content} [{strength}] (P-K: {pk_str})")

        if support_list:
            lines.append("  **Supporting K-Nodes:**")
            lines.extend(support_list[:10])

        if rule_out_list:
            lines.append("  **Rule-Out K-Nodes:**")
            lines.extend(rule_out_list[:5])

        essential_support = sum(
            1 for k_id, rel, _ in gt_evidence
            if rel == "Support" and k_node_map.get(k_id, {}).get("importance") in ["Essential", "Pathognomonic"]
        )
        lines.append(f"\n  Essential/Pathognomonic Support Count: {essential_support}")

    else:
        lines.append("  ⚠️ NO EVIDENCE found supporting the correct diagnosis!")
        lines.append("  This suggests the agent completely missed key features for this condition.")

    # === Section 4: All Candidates Summary ===
    lines.append("\n### 📊 ALL CANDIDATES EVIDENCE SUMMARY")

    for d_node in nodes.get("d_nodes", []):
        d_id = d_node["id"]
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")

        evidence = kd_by_d.get(d_id, [])

        support_count = sum(1 for _, rel, _ in evidence if rel == "Support")
        rule_out_count = sum(1 for _, rel, _ in evidence if rel == "Rule_Out")

        match_count = 0
        conflict_count = 0
        void_count = 0

        for k_id, _, _ in evidence:
            for p_id, pk_rel in pk_by_k.get(k_id, []):
                if pk_rel == "Match":
                    match_count += 1
                elif pk_rel == "Conflict":
                    conflict_count += 1
                elif pk_rel == "Void":
                    void_count += 1

        marker = " ← GROUND TRUTH" if d_id == gt_d_id else ""

        lines.append(
            f"  - {d_name} ({d_id}) [Rank: {d_rank}]{marker}"
        )
        lines.append(
            f"    K-D: {support_count} Support, {rule_out_count} Rule_Out | "
            f"P-K: {match_count} Match, {conflict_count} Conflict, {void_count} Void"
        )

    # === Section 5: Patient Features Summary ===
    lines.append("\n### 👤 PATIENT FEATURES (P-Nodes)")

    present_features = []
    absent_features = []

    for p_node in nodes.get("p_nodes", []):
        content = p_node.get("content", "")
        status = p_node.get("status", "Present")

        if status == "Present":
            present_features.append(content)
        elif status == "Absent":
            absent_features.append(content)

    if present_features:
        lines.append(f"  **Present ({len(present_features)}):** " + ", ".join(present_features[:15]))
        if len(present_features) > 15:
            lines.append(f"    ... and {len(present_features) - 15} more")

    if absent_features:
        lines.append(f"  **Absent ({len(absent_features)}):** " + ", ".join(absent_features[:10]))
        if len(absent_features) > 10:
            lines.append(f"    ... and {len(absent_features) - 10} more")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)

def build_prompt_with_hint(base_prompt: str, global_hint: Optional[str]) -> str:

    if global_hint and global_hint.strip():
        return base_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"
    return base_prompt

def summarize_graph_for_judge(
    graph_json: Dict[str, Any],
    naive_scores: Optional[Dict[str, float]] = None
) -> str:

    from collections import defaultdict

    graph_data = graph_json.get("graph", {})
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", {})

    p_nodes = nodes.get("p_nodes", [])
    k_nodes = nodes.get("k_nodes", [])
    d_nodes = nodes.get("d_nodes", [])
    p_k_links = edges.get("p_k_links", [])
    k_d_links = edges.get("k_d_links", [])

    p_node_map = {p["id"]: p for p in p_nodes}
    k_node_map = {k["id"]: k for k in k_nodes}
    d_node_map = {d["id"]: d for d in d_nodes}

    pk_by_k = defaultdict(list)  # k_id -> [(p_id, relation), ...]
    for edge in p_k_links:
        k_id = edge.get("target", "")
        p_id = edge.get("source", "")
        relation = edge.get("relation", "")
        pk_by_k[k_id].append((p_id, relation))

    kd_by_d = defaultdict(list)  # d_id -> [k_id, ...]
    for edge in k_d_links:
        d_id = edge.get("target", "")
        k_id = edge.get("source", "")
        kd_by_d[d_id].append(k_id)

    if naive_scores is None:
        graph = rebuild_graph_from_json(graph_json)
        naive_scores = calculate_deterministic_score(graph)

    d_nodes_sorted = sorted(d_nodes, key=lambda d: d.get("initial_rank", 999))

    lines = []
    lines.append("=" * 60)
    lines.append("RICH STRUCTURED SUMMARY (Phase 3 Evidence)")
    lines.append("=" * 60)
    lines.append("")

    for d_node in d_nodes_sorted:
        d_id = d_node.get("id", "")
        d_name = d_node.get("name", "Unknown")
        d_rank = d_node.get("initial_rank", "?")
        d_state = d_node.get("state", "Active")

        score = naive_scores.get(d_id, 0.0)
        score_str = f"+{score:.1f}" if score >= 0 else f"{score:.1f}"

        state_marker = " 🔴 PRUNED" if d_state == "Pruned" else ""

        # Header
        lines.append(f"### [{d_name}] (ID: {d_id}) – Naive Score: **{score_str}**{state_marker}")
        lines.append(f"    Phase 1 Rank: {d_rank}")
        lines.append("")

        match_evidence = []
        conflict_evidence = []
        shadow_pathognomonic = []
        has_pivot_match = False

        related_k_ids = kd_by_d.get(d_id, [])

        for k_id in related_k_ids:
            k_node = k_node_map.get(k_id, {})
            k_content = k_node.get("content", k_id)
            k_importance = k_node.get("importance", "Weak")
            k_type = k_node.get("k_type", "General")

            pk_relations = pk_by_k.get(k_id, [])

            for p_id, relation in pk_relations:
                p_node = p_node_map.get(p_id, {})
                p_content = p_node.get("content", p_id)
                p_status = p_node.get("status", "Present")

                importance_tag = f"[{k_importance}]" if k_importance in ["Essential", "Pathognomonic", "Strong"] else ""
                pivot_tag = " [PIVOT]" if k_type == "Pivot" else ""

                if relation == "Match":
                    match_evidence.append(f'{p_content} ↔ "{k_content}" {importance_tag}{pivot_tag}')

                    if k_type == "Pivot":
                        has_pivot_match = True

                elif relation == "Conflict":
                    conflict_evidence.append(f'{p_content} (Status: {p_status}) ✗ "{k_content}" {importance_tag}')

                elif relation == "Void":

                    if k_importance == "Pathognomonic":
                        shadow_pathognomonic.append(f'"{k_content}" [Pathognomonic] - MISSING')

        # [+] MATCH
        if match_evidence:
            lines.append(f"• **[+] MATCH ({len(match_evidence)}):**")
            for ev in match_evidence[:8]:
                lines.append(f"    - {ev}")
            if len(match_evidence) > 8:
                lines.append(f"    - ... and {len(match_evidence) - 8} more")
        else:
            lines.append("• **[+] MATCH (0):** None")
        lines.append("")

        # [-] CONFLICT
        if conflict_evidence:
            lines.append(f"• **[-] CONFLICT ({len(conflict_evidence)}):**")
            for ev in conflict_evidence[:5]:
                lines.append(f"    - {ev}")
            if len(conflict_evidence) > 5:
                lines.append(f"    - ... and {len(conflict_evidence) - 5} more")
        else:
            lines.append("• **[-] CONFLICT (0):** None")
        lines.append("")

        # [?] CRITICAL MISSING (Shadow - Pathognomonic Only)
        if shadow_pathognomonic:
            lines.append(f"• **[?] CRITICAL MISSING ({len(shadow_pathognomonic)} Pathognomonic):**")
            for ev in shadow_pathognomonic[:5]:
                lines.append(f"    - {ev}")
            if len(shadow_pathognomonic) > 5:
                lines.append(f"    - ... and {len(shadow_pathognomonic) - 5} more")
        else:
            lines.append("• **[?] CRITICAL MISSING:** None (No Pathognomonic features missing)")
        lines.append("")

        # Pivot Status
        pivot_status = "✅ YES (Pivot Feature Matched)" if has_pivot_match else "❌ NO"
        lines.append(f"• **Pivot Status:** {pivot_status}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    return "\n".join(lines)

def calculate_naive_scores(graph_json: Dict[str, Any]) -> Dict[str, float]:

    graph = rebuild_graph_from_json(graph_json)
    return calculate_deterministic_score(graph)

