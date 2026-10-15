# MedEinst dataset interface

## Three distinct resources

| Paper term | Public split or release file | Use |
|---|---|---|
| Reference Set | `zhui711/MedEinst` `train` split (`train.jsonl`) | Training resource; CGME resolves its fixed support records here. The dataset card describes 10,689 pairs / 21,378 rows. |
| MedEinst Benchmark Test Set | `zhui711/MedEinst` `test` split (`test.jsonl`) | Online inference and external evaluation; the card describes 5,383 pairs / 10,766 rows. |
| Balanced Seed Subset, `medeinst_support_853` | `data/medeinst_support_853.json` | Fixed ordered selection of **853 individual records**, comprising 522 control and 331 trap variants over 46 disease IDs. It contains identities and labels, not narratives. |

The public dataset is [zhui711/MedEinst](https://huggingface.co/datasets/zhui711/MedEinst). Download its `train.jsonl` and `test.jsonl` separately from the source release. The old internal folder names are not required. Use the included fixed manifest for the published support selection; fresh seed-42 sampling produces a separate candidate manifest.

## Input schema and pairing

Each public JSONL line is an object with `case_id`, `case_type` (`control` or `trap`), `narrative`, and `ground_truth`; `age` and `sex` are accepted but are not passed separately to Phase 1 because they are already represented in the unchanged narrative. The two variants share a `case_id` and have distinct ground-truth diagnoses. `src/data/medeinst_adapter.py` groups by `(case_id, case_type)`, rejects duplicates and incomplete pairs, maps disease names to the original two-digit IDs, and leaves narrative text untouched. The agent and output keep the historical internal variant names `control_case` and `trap_case`.

A directory of `case_*/final_benchmark_pair.json` files remains an optional compatibility input for the same `--reference-set` and `--benchmark-set` options. Its two objects must have `narrative` and `ground_truth`. Both formats yield the same in-memory `CasePair` records. The release does not materialize a second dataset copy.

The fixed manifest is resolved in its original order by `(case_id, case_type, ground_truth_id)`. A missing or differently labeled public row raises an error listing its exact identity; no substitute is sampled. The former `config/offline_train_list.json` path-bearing copy has been removed from runtime. Its origin and the Stage 1 comparison are preserved in `docs/source_map.md`.

## Commands

Run from the release root after installing dependencies and configuring the required services:

```bash
python scripts/run_cgme.py --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json

python scripts/run_ecr_agent.py --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/refined_graphs \
  --memory-bank-dir /path/to/memory_bank
```

`--limit N` and `--case-ids id1,id2` select a pair subset in Online inference. `--skip-control` and `--skip-trap` select the variants to run. The Offline runner retains serial/concurrent options, worker count, batch size, output directory and checkpoint resume. Its original resume bookkeeping is unchanged.

Optional **new resampling**, distinct from the fixed paper-support path:

```bash
python scripts/prepare_support_set.py --reference-set /path/to/train.jsonl \
  --output /path/to/candidate_support.json --seed 42 --cap 20
```

The command samples individual case variants per disease, shuffles with seed 42, caps each disease at 20, writes a path-free candidate manifest, and compares it with the fixed 853-record file. It refuses to overwrite that fixed file. Its candidate is not silently substituted into `run_cgme.py`.

The Online runner writes `results_detail.jsonl`, `inference_summary.json`, and per-case graph JSON files. Benchmark metric evaluation is outside this core agent release; the paper defines the evaluation protocol.

Model and artifact settings are described in [configuration.md](configuration.md) and [artifacts.md](artifacts.md).
