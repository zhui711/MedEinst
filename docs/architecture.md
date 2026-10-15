# ECR-Agent architecture and source map

The [paper](https://arxiv.org/abs/2601.06636) names two connected components. **Dynamic Causal Inference (DCI)** diagnoses a case; **Critic-Driven Graph & Memory Evolution (CGME)** runs DCI on support cases, critiques unsuccessful outcomes, and evolves reusable Illness Graphs and an Exemplar Base. Historical class names are retained where they are part of the implementation.

| Paper component | Release implementation | Role |
|---|---|---|
| Dual-Pathway Perception | `Phase1Manager`, `phase1_perception.py`, `phase1_trackB.py` | Generates differential candidates and an objective problem representation with patient observations. |
| Causal Graph Initialization | `MedicalGraph`, `Phase2GraphReasoning`, `Phase2HybridEngine`, `GoldenGraphLoader` | Creates disease, patient, and knowledge nodes; Online inference can initialize from a refined or raw Illness Graph. |
| Forward Causal Reasoning | Both Phase 2 engines; `PubMedTool`, `OpenTargetsTool`; knowledge extraction prompts | Retrieves and connects Pivot and General evidence to candidate diseases. |
| Backward Causal Reasoning | Both Phase 2 engines; `MedicalGraph.create_shadow_p_node` | Checks expected evidence and represents unmatched observations as Shadow nodes. |
| Evidence Audit | `graph_tools.py`, Offline `JudgeAgent`, Online `MemoryAugmentedJudge` | Scores and summarizes graphs; the Online judge also retrieves exemplars before final diagnosis. |
| Critic-driven evolution | `OfflineTrainer`, `ConcurrentOfflineTrainer`, `CriticalModelAgent` | Runs support cases, judges outcomes, and retries unsuccessful trajectories with critic feedback. |
| Illness Graph evolution | `AggregationManager`, `CanonicalGraph`, `scripts/refine_illness_graphs.py` | Aggregates validated cases by disease and optionally refines knowledge nodes. |
| Exemplar Base | `MemoryBankManager`, `EmbeddingClient` | Stores compact validated-case records and their aligned embeddings for retrieval. |

## Execution paths

```text
Reference Set + fixed 853-record manifest
  → scripts/run_cgme.py → OfflineTrainer
  → Phase1Manager → Phase2GraphReasoning → JudgeAgent
  → CriticalModelAgent when correction is needed
  → AggregationManager + MemoryBankManager
  → raw Illness Graphs + Exemplar Base
  → optional scripts/refine_illness_graphs.py → refined Illness Graphs

Benchmark Test Set + prebuilt Illness Graphs + Exemplar Base
  → scripts/run_ecr_agent.py
  → Phase1Manager → Phase2HybridEngine
  → GoldenGraphLoader + live PubMed/OpenTargets reasoning
  → graph scoring and summary → MemoryAugmentedJudge
  → case-level predictions and graph outputs
```

Offline reasoning uses `Phase2GraphReasoning` because it creates the initial graphs. Online reasoning uses `Phase2HybridEngine` to incorporate prebuilt graphs while retaining live evidence retrieval. They share some reasoning functions but are not interchangeable. The Online loader prefers `*_Refined_GG.json`, then `*_GG.json`; a disease without a prebuilt graph follows the existing live-retrieval path. `MedicalGraph` represents patient observations as Present, Absent, or Missing; its knowledge nodes include Pivot and General evidence, while Shadow nodes record expected but unmatched observations.

The paper describes an Exemplar Base of validated reasoning trajectories. The released implementation serializes **compact case records and embeddings**, not full model transcripts. This is the original source representation; see [artifacts.md](artifacts.md). The exact historical command that produced the paper's numerical results was not recovered. This release retains the approved full graph-and-memory implementation without claiming that exact command provenance.
