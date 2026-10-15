# MedEinst and ECR-Agent

[![arXiv](https://img.shields.io/badge/arXiv-2601.06636-b31b1b.svg)](https://arxiv.org/abs/2601.06636)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zhui711/MedEinst)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[Paper](https://arxiv.org/abs/2601.06636) · [Project page](https://zhui711.github.io/MedEinst/) · [MedEinst dataset](https://huggingface.co/datasets/zhui711/MedEinst) · [Illness Graphs](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs) · [Exemplar Base](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base)

This repository provides the main-method implementation of **ECR-Agent (Evidence-based Causal Reasoning Agent)** from *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis* (ACL 2026). The MedEinst Benchmark Test Set contains 5,383 counterfactual pairs spanning 49 diseases. ECR-Agent combines case-level causal reasoning with knowledge and experience evolved from support cases. The public clinical narratives are derived and processed from [DDXPlus](https://github.com/mila-iqia/ddxplus); we thank its authors for releasing the source data.

## Method

**Dynamic Causal Inference (DCI)** uses Dual-Pathway Perception to generate diagnostic candidates and an objective problem representation. It initializes a causal graph, performs forward evidence retrieval and backward verification, then audits graph evidence and retrieved exemplars to produce a diagnosis.

**Critic-Driven Graph & Memory Evolution (CGME)** processes support cases with the diagnostic pipeline, uses critic feedback to refine unsuccessful attempts, and stores validated evidence in disease-specific **Illness Graphs** and an **Exemplar Base**. Offline CGME builds graphs through live reasoning; Online DCI initializes from prebuilt graphs and continues live reasoning. See [architecture.md](docs/architecture.md) for the paper-to-code mapping.

## Installation and configuration

Use Python **3.10+** in an environment of your choice, then install the listed Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the base, critic, and embedding models and endpoints in `config/settings.yaml`, or pass `--config /path/to/settings.yaml`. The default `qwen3-32b` base model needs a compatible endpoint; the critic model is configurable. Keep credentials in environment variables.

For example, with a compatible Qwen service and the official OpenAI API for critic and embeddings:

```bash
export ECR_BASE_URL=https://your-compatible-service.example/v1
export ECR_BASE_API_KEY=your-base-model-key
export OPENAI_API_KEY=your-openai-key
export PUBMED_EMAIL=you@example.org
```

The model roles can use separate endpoints and credentials. PubMed and OpenTargets provide external medical evidence. See [configuration.md](docs/configuration.md) for settings and the paper's GPT-5 critic setting.

## Quickstart: Online ECR-Agent with prebuilt artifacts

1. Download the **MedEinst Benchmark Test Set** (`test.jsonl`) from the [dataset repository](https://huggingface.co/datasets/zhui711/MedEinst). It contains flat control/trap records.
2. Download and extract the [Illness Graphs](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs) and [Exemplar Base](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base) outside this source tree. Check the required graph and memory files:

   ```bash
   test -d /path/to/artifacts/graphs/golden_graphs_refined_1
   test -f /path/to/artifacts/exemplars/memory_bank/memory_bank.json
   test -f /path/to/artifacts/exemplars/memory_bank/embeddings.npy
   ```

3. From the repository root, run the full graph-and-memory implementation:

   ```bash
   python scripts/run_ecr_agent.py \
     --benchmark-set /path/to/test.jsonl \
     --refined-graph-dir /path/to/artifacts/graphs/golden_graphs_refined_1 \
     --memory-bank-dir /path/to/artifacts/exemplars/memory_bank \
     --output-dir /path/to/online_results
   ```

See [artifacts.md](docs/artifacts.md) for the extracted layout and embedding requirements. The existing live-retrieval path handles a disease without a stored graph.

The Online runner writes case-level predictions to `results_detail.jsonl` and a run summary to `inference_summary.json` under `--output-dir`. The paper defines Baseline Accuracy, Robust Accuracy, and Bias Trap Rate; metric evaluation is outside this agent package.

## Full workflow: Reference Set → CGME → DCI

Download `train.jsonl`, the paper's **Reference Set**, from the [MedEinst dataset repository](https://huggingface.co/datasets/zhui711/MedEinst). The included `data/medeinst_support_853.json` identifies the fixed **Balanced Seed Subset** of 853 individual control or trap records; narratives come from the Reference Set.

```bash
python scripts/run_cgme.py \
  --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json \
  --output-dir /path/to/cgme_output
```

CGME writes raw Illness Graphs under `cgme_output/golden_graphs/` and an Exemplar Base under `cgme_output/memory_bank/`. Optional Illness Graph refinement uses the configured critic:

```bash
python scripts/refine_illness_graphs.py \
  --input-dir /path/to/cgme_output/golden_graphs \
  --output-dir /path/to/refined_graphs \
  --log-dir /path/to/refinement_logs
```

For Online inference, use `--raw-graph-dir /path/to/cgme_output/golden_graphs`, optionally `--refined-graph-dir /path/to/refined_graphs`, and `--memory-bank-dir /path/to/cgme_output/memory_bank`. `scripts/prepare_support_set.py` can create a new support selection; the included manifest preserves the fixed subset. See [dataset_interface.md](docs/dataset_interface.md) for details.

## Citation and license

```bibtex
@misc{chen2026medeinstbenchmarkingeinstellungeffect,
  title={MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis},
  author={Wenting Chen and Zhongrui Zhu and Guolin Huang and Wenxuan Wang},
  year={2026},
  eprint={2601.06636},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.06636}
}
```

This repository is licensed under Apache-2.0; see [LICENSE](LICENSE). The MedEinst dataset and third-party dependencies have their own terms. For questions about the project, contact [Zhongrui Zhu](mailto:zhongruizhuzz@outlook.com).
