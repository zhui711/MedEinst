# ECR-Agent Architecture

[ECR-Agent](https://arxiv.org/abs/2601.06636) combines **Dynamic Causal Inference (DCI)** with **Critic-Driven Graph & Memory Evolution (CGME)**. CGME builds reusable medical knowledge and experience from support cases. DCI uses these resources to reason about new clinical cases.

## Components

| Component | Main implementation | Responsibility |
| --- | --- | --- |
| Dual-Pathway Perception | `Phase1Manager`, `phase1_perception.py`, `phase1_trackB.py` | Generates differential diagnosis candidates and a structured problem representation. |
| Causal Graph Initialization | `MedicalGraph`, `Phase2HybridEngine`, `GoldenGraphLoader` | Initializes patient-specific causal graphs using observations, candidate diagnoses, and available Illness Graphs. |
| Forward Causal Reasoning | `Phase2GraphReasoning`, `Phase2HybridEngine`, `PubMedTool`, `OpenTargetsTool` | Retrieves and connects medical evidence to candidate diseases. |
| Backward Causal Reasoning | Phase 2 reasoning modules, `MedicalGraph.create_shadow_p_node` | Checks expected evidence and represents unmatched observations with Shadow nodes. |
| Evidence Audit | `graph_tools.py`, `JudgeAgent`, `MemoryAugmentedJudge` | Scores and summarizes graph evidence; online inference also retrieves similar exemplars. |
| Critic-Driven Evolution | `OfflineTrainer`, `ConcurrentOfflineTrainer`, `CriticalModelAgent` | Evaluates and refines reasoning on support cases through critic feedback. |
| Illness Graph Evolution | `AggregationManager`, `CanonicalGraph`, `scripts/refine_illness_graphs.py` | Aggregates validated evidence by disease and supports optional graph refinement. |
| Exemplar Base | `MemoryBankManager`, `EmbeddingClient` | Maintains case records and embeddings for experience retrieval. |

## Offline CGME

```text
Reference Set + 853-case support manifest
  → scripts/run_cgme.py
  → Dual-Pathway Perception
  → Phase2GraphReasoning (live evidence)
  → JudgeAgent and critic-guided refinement
  → Illness Graph aggregation + Exemplar Base
  → optional scripts/refine_illness_graphs.py
```

Offline CGME builds Illness Graphs from support cases rather than requiring prebuilt disease graphs. Its successful cases contribute to the graph collection and Exemplar Base.

## Online DCI

```text
Benchmark cases + prebuilt Illness Graphs + Exemplar Base
  → scripts/run_ecr_agent.py
  → Dual-Pathway Perception
  → Phase2HybridEngine (graph initialization + live evidence)
  → Forward and Backward Causal Reasoning
  → graph scoring and evidence audit
  → final diagnosis and case-level outputs
```

Online DCI uses the generated Illness Graphs to initialize reasoning while continuing to retrieve relevant evidence. The graph loader prefers `*_Refined_GG.json` files and can use raw `*_GG.json` files when available. If a candidate disease has no prebuilt Illness Graph, the hybrid reasoning path can continue with live evidence retrieval.

Patient observations are represented as **Present**, **Absent**, or **Missing**. Knowledge nodes include **Pivot** and **General** evidence, while **Shadow** nodes capture expected but unmatched observations. The Exemplar Base contains compact case records and aligned embeddings used for retrieval.

For the associated input formats, model settings, and artifact layouts, see [Dataset Interface](dataset_interface.md), [Configuration](configuration.md), and [Artifacts](artifacts.md).
