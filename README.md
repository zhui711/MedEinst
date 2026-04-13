# MedEinst: Benchmarking the Einstellung Effect in Medical LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2601.06636-b31b1b.svg)](https://arxiv.org/abs/2601.06636)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zhui711/MedEinst)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data License](https://img.shields.io/badge/Data%20License-CC--BY--4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

This is the official repository for the paper **"[MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis](https://arxiv.org/abs/2601.06636)"**.

## 📢 Updates
- **[Jan 2026]**: The paper is now available on arXiv! 
- The codebase for **ECR-Agent** (Evidence-based Causal Reasoning Agent) and the complete benchmark dataset are currently undergoing final formatting and will be fully released here shortly. Please `Watch` or `Star` this repository for updates.

## 🗂️ Dataset (MedEinst Benchmark)
The MedEinst benchmark consists of 5,383 counterfactual pairs of clinical narratives spanning 49 diseases. 
The dataset is structured into `train` and `test` splits, with individual patient cases organized in subfolders. 

You can access the full dataset on Hugging Face: **[MedEinst Dataset](https://huggingface.co/datasets/zhui711/MedEinst)**

> **⚠️ Data Source & Attribution:**
> The MedEinst benchmark is derived and processed from the **DDXPlus** dataset. In compliance with the CC-BY 4.0 license, we gratefully acknowledge the authors of DDXPlus. If you use the MedEinst dataset in your research, please ensure you cite both our paper and the original DDXPlus paper.

## 🛠️ ECR-Agent (Coming Soon)
Our proposed ECR-Agent mitigates the Einstellung Effect by aligning LLM reasoning with Evidence-Based Medicine (EBM) via Dynamic Causal Inference (DCI). 

**Note on Dependencies:** 
Running the full ECR-Agent pipeline will require access to certain external APIs (e.g., Qwen3, GPT-5, Gemini) and medical databases (PubMed, OpenTargets). A detailed `requirements.txt` and configuration guide for API keys will be provided upon the code release.

## 📝 Citation
If you find our benchmark or framework useful in your research, please consider citing our work:

```bibtex
@article{chen2026medeinst,
  title={MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis},
  author={Chen, Wenting and Zhu, Zhongrui and Huang, Guolin and Wang, Wenxuan},
  journal={arXiv preprint arXiv:2601.06636},
  year={2026}
}
```

If you use the MedEinst dataset, please also cite the original DDXPlus dataset:

```bibtex
@inproceedings{tchango2022ddxplus,
  title={DDXPlus: A New Dataset For Automatic Medical Diagnosis},
  author={Fansi Tchango, Ars{\`e}ne and Goel, Rishab and Wen, Zhi and Martel, Julien and Ghosn, Joumana},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31306--31318},
  year={2022}
}
```

## 📧 Contact
**Zhongrui Zhu** -[zhongruizhuzz@outlook.com](mailto:zhongruizhuzz@outlook.com)
