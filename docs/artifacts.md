# Illness Graph and Exemplar Base artifacts

The official artifacts are distributed separately: [Illness Graphs](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs) and [Exemplar Base](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base). They are not bundled with the source. Configure paths to the **extracted directories containing the files**, not to an archive or repository parent.

The supplied graph archive contains 46 refined `*_Refined_GG.json` files under `golden_graphs_refined_1/`, with no raw `*_GG.json` graphs. Point `--refined-graph-dir` to that directory. `--raw-graph-dir` can point to an optional raw graph directory; the existing `GoldenGraphLoader` prefers refined files and falls back to raw files when present. Its disease-specific live-retrieval fallback remains intact for missing graphs.

The supplied Exemplar Base archive contains `memory_bank/memory_bank.json` and `memory_bank/embeddings.npy`. Point `--memory-bank-dir` to that `memory_bank/` directory. The supplied bank has 700 records and a 700 × 1536 embedding matrix. The configured embedding model must produce compatible vectors for retrieval; the checked-in `text-embedding-3-small` setting matches the supplied dimension. Serialization and retrieval behavior are unchanged.

Check that the two memory files exist at the configured path before running. The original memory loader does not raise when either file is absent, so an incorrect directory can appear as an empty bank.

Representative Online invocation after extracting the artifacts and configuring model credentials:

```bash
python scripts/run_ecr_agent.py --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/golden_graphs_refined_1 \
  --memory-bank-dir /path/to/memory_bank \
  --output-dir /path/to/online_results
```

For Offline evolution, `run_cgme.py --output-dir /path/to/cgme_output` writes raw graphs under `cgme_output/golden_graphs/` and memory files under `cgme_output/memory_bank/`. Optional `refine_illness_graphs.py --input-dir /path/to/cgme_output/golden_graphs --output-dir /path/to/refined_graphs` produces refined graphs for subsequent Online inference. The serialized graph filenames retain the original `_GG.json` and `_Refined_GG.json` conventions.
