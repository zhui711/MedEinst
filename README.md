# MedEinst

[![arXiv](https://img.shields.io/badge/arXiv-2601.06636-b31b1b.svg)](https://arxiv.org/abs/2601.06636)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zhui711/MedEinst)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official repository for the paper **"[MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis](https://arxiv.org/abs/2601.06636)"**.

## 📢 Updates
- **[Jan 2026]**: The paper is now available on arXiv! 
- The codebase for **ECR-Agent** (Evidence-based Causal Reasoning Agent) and the complete benchmark dataset are currently undergoing final formatting and will be fully released here shortly. Please `Watch` or `Star` this repository for updates.

## 🗂️ Dataset (MedEinst Benchmark)
The MedEinst benchmark consists of 5,383 counterfactual pairs of clinical narratives spanning 49 diseases. 
The dataset is structured into `train` and `test` splits, with individual patient cases organized in subfolders. 
You can access the full dataset on Hugging Face:[https://huggingface.co/datasets/zhui711/MedEinst]

## 🛠️ ECR-Agent (Coming Soon)
Our proposed ECR-Agent mitigates the Einstellung Effect by aligning LLM reasoning with Evidence-Based Medicine (EBM) via Dynamic Causal Inference (DCI). 

**Note on Dependencies:** 
Running the full ECR-Agent pipeline will require access to certain external APIs (e.g., Qwen3, GPT-5, Gemini) and medical databases (PubMed, OpenTargets). A detailed `requirements.txt` and configuration guide for API keys will be provided upon the code release.

## 📝 Citation
If you find our benchmark or framework useful in your research, please consider citing our work using the following BibTeX entry:

```bibtex
@article{chen2026medeinst,
  title={MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis},
  author={Chen, Wenting and Zhu, Zhongrui and Huang, Guolin and Wang, Wenxuan},
  journal={arXiv preprint arXiv:2601.06636},
  year={2026}
}
