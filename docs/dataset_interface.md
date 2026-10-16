# MedEinst Dataset Interface

The [MedEinst dataset](https://huggingface.co/datasets/zhui711/MedEinst) provides a **Reference Set** for support-case selection and a **Benchmark Test Set** for diagnostic evaluation. ECR-Agent also includes a fixed 853-record support manifest used by Offline CGME.

## Data resources

| Resource | File | Description |
| --- | --- | --- |
| Reference Set | `train.jsonl` | 10,689 counterfactual case pairs used as a training resource. |
| Benchmark Test Set | `test.jsonl` | 5,383 counterfactual pairs spanning 49 diseases. |
| Balanced Seed Subset | [`data/medeinst_support_853.json`](../data/medeinst_support_853.json) | 853 individual records selected for Offline CGME: 522 control and 331 trap variants across 46 disease IDs. |

The fixed support manifest stores case identities and labels, not patient narratives. The `run_cgme.py` command resolves those records against the Reference Set in manifest order.

## Record format

Each JSONL record contains these fields:

| Field | Meaning |
| --- | --- |
| `case_id` | Identifier shared by the paired control and trap variants |
| `case_type` | `control` or `trap` |
| `narrative` | Clinical case description provided to the agent |
| `ground_truth` | Correct diagnosis for the corresponding variant |

The adapter also accepts optional `age` and `sex` fields. Control and trap variants are paired by `case_id` while remaining distinct observations. Disease names are mapped to the corresponding disease IDs, and the clinical narratives are preserved as provided.

As an alternative to JSONL, the interface accepts a directory of paired files in the `case_*/final_benchmark_pair.json` format.

## Running with the public dataset

Offline CGME with the fixed support set:

```bash
python scripts/run_cgme.py \
  --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json
```

Online ECR-Agent inference:

```bash
python scripts/run_ecr_agent.py \
  --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/refined_graphs \
  --memory-bank-dir /path/to/memory_bank
```

The Online command also accepts `--limit N`, `--case-ids id1,id2`, `--skip-control`, and `--skip-trap` to select cases or variants. The Offline command supports serial and concurrent processing options.

## Optional support-set sampling

To construct a new candidate support manifest rather than use the fixed selection:

```bash
python scripts/prepare_support_set.py \
  --reference-set /path/to/train.jsonl \
  --output /path/to/candidate_support.json \
  --seed 42 --cap 20
```

This command samples individual case variants with a per-pathology cap and writes a separate candidate manifest. It does not replace the fixed `medeinst_support_853.json` selection.

Online inference produces `results_detail.jsonl`, `inference_summary.json`, and per-case graph output files. The benchmark evaluation metrics are defined in the [paper](https://arxiv.org/abs/2601.06636).

For model configuration and knowledge artifacts, see [Configuration](configuration.md) and [Artifacts](artifacts.md).
