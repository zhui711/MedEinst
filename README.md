<div align="center">

# MedEinst

**Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**

<p>
  <a href="https://arxiv.org/abs/2601.06636"><img src="https://img.shields.io/badge/Paper-arXiv%3A2601.06636-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" /></a>
  <a href="https://zhui711.github.io/MedEinst/"><img src="https://img.shields.io/badge/Project-Website-2563EB?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Website" /></a>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst"><img src="https://img.shields.io/badge/Benchmark-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="MedEinst Dataset" /></a>
</p>
<p>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs"><img src="https://img.shields.io/badge/Illness-Graphs-0E7490?style=flat-square" alt="Illness Graphs" /></a>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base"><img src="https://img.shields.io/badge/Exemplar-Base-7C3AED?style=flat-square" alt="Exemplar Base" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-64748B?style=flat-square" alt="Apache-2.0 License" /></a>
</p>

</div>

## Overview

Large language models can favor familiar diagnoses even when patient-specific evidence points elsewhere—a manifestation of the **Einstellung Effect**. **MedEinst** studies this failure mode through counterfactual differential diagnosis and introduces **ECR-Agent (Evidence-based Causal Reasoning Agent)** to support more evidence-grounded diagnostic reasoning.

This repository provides the **MedEinst benchmark resources and the main-method ECR-Agent source code**.

## 🧪 MedEinst Benchmark & Dataset

MedEinst contains **5,383 counterfactual clinical case pairs spanning 49 diseases** in its benchmark test set. Each pair includes a *control* case and a *trap* case: discriminative evidence changes the correct diagnosis while much of the clinical context remains similar. The paper evaluates **17 LLMs** and measures diagnostic accuracy and susceptibility to this bias using **Baseline Accuracy, Robust Accuracy, and Bias Trap Rate**.

| Resource | Contents | Purpose |
| --- | --- | --- |
| **Reference Set** (`train.jsonl`) | 10,689 case pairs | Training resource and support-case selection |
| **Benchmark Test Set** (`test.jsonl`) | 5,383 case pairs · 49 diseases | Evaluation of diagnostic robustness and the Einstellung Effect |
| **Balanced Seed Subset** (`data/medeinst_support_853.json`) | 853 individual case records | Fixed support selection for offline CGME |

**Dataset:** [🤗 MedEinst on Hugging Face](https://huggingface.co/datasets/zhui711/MedEinst) · [Dataset interface](docs/dataset_interface.md)

The clinical narratives are derived and processed from [DDXPlus](https://github.com/mila-iqia/ddxplus). We thank the DDXPlus authors for making their dataset available.

## 🩺 ECR-Agent

To mitigate the Einstellung Effect, the paper proposes **ECR-Agent**, which consists of two complementary components:

**Dynamic Causal Inference (DCI).** Performs dual-pathway perception, causal graph initialization, forward evidence retrieval, backward verification, and evidence auditing to produce a diagnosis.

**Critic-Driven Graph & Memory Evolution (CGME).** Uses support cases and critic feedback to evolve disease-specific **Illness Graphs** and an **Exemplar Base**. Offline CGME builds this knowledge from live reasoning; online DCI initializes from the prebuilt knowledge and continues evidence-based reasoning.

See the [method architecture](docs/architecture.md) and the [project page](https://zhui711.github.io/MedEinst/) for the framework diagram.

## ⚙️ Installation & Configuration

Python 3.10+ is recommended. Install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure your base LLM, critic, embedding model, and API endpoints in `config/settings.yaml`. A compatible model endpoint is required for the default `qwen3-32b` base model. Credentials should be supplied through environment variables; see [configuration.md](docs/configuration.md) for details.

## 🚀 Quickstart: Online ECR-Agent

Download the Benchmark Test Set (`test.jsonl`), [Illness Graphs](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs), and [Exemplar Base](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base). Extract the prebuilt artifacts and configure the model services before running:

```bash
python scripts/run_ecr_agent.py \
  --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/golden_graphs_refined_1 \
  --memory-bank-dir /path/to/memory_bank \
  --output-dir /path/to/online_results
```

The runner writes case-level predictions to `results_detail.jsonl` and a run summary to `inference_summary.json`. See [artifact layout](docs/artifacts.md) for the expected files and directories. The three benchmark metrics are defined in the paper; a separate evaluation utility is not included in this agent release.

## 🔄 Offline CGME: Build the Knowledge & Memory

The included `data/medeinst_support_853.json` preserves the fixed support selection from the paper. Download `train.jsonl` from the [Reference Set](https://huggingface.co/datasets/zhui711/MedEinst), then run:

```bash
python scripts/run_cgme.py \
  --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json \
  --output-dir /path/to/cgme_output
```

Optionally refine the generated Illness Graphs:

```bash
python scripts/refine_illness_graphs.py \
  --input-dir /path/to/cgme_output/golden_graphs \
  --output-dir /path/to/refined_graphs \
  --log-dir /path/to/refinement_logs
```

The resulting Illness Graphs and Exemplar Base can be used for online inference. See [dataset interface](docs/dataset_interface.md) and [artifact layout](docs/artifacts.md) for further details.

## 📖 Citation

If you use MedEinst or ECR-Agent in your research, please cite our paper:

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

## 📬 Contact

**Zhongrui Zhu** — [zhongruizhuzz@outlook.com](mailto:zhongruizhuzz@outlook.com)
