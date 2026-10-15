# ECR-Agent configuration

The three launchers read `config/settings.yaml` by default; use `--config /path/to/settings.yaml` to select another file. No credential belongs in the YAML. API keys come from the environment. The configured model IDs, prompts, inference parameters, critic retries, graph thresholds, and embedding dimension retain the extracted implementation values.

| YAML section | Purpose | Default credential variable | Optional URL override |
|---|---|---|---|
| `api` | Base diagnostic model (`qwen3-32b`) | `ECR_BASE_API_KEY` | `ECR_BASE_URL` |
| `critical_model` | CGME critic and graph refinement (`gpt-4o` in the checked-in config) | `OPENAI_API_KEY` | `ECR_CRITIC_URL` |
| `embedding` | Exemplar retrieval and Online P-node matching (`text-embedding-3-small`, 1536 dimensions) | `OPENAI_API_KEY` | `ECR_EMBEDDING_URL` |

Each section also accepts `base_url` and `api_key_env` in YAML. A nonempty URL environment override takes precedence over `base_url`; if both are absent, the official OpenAI API is used. The base model's default `qwen3-32b` requires a compatible service, so set `ECR_BASE_URL` and `ECR_BASE_API_KEY` (or change the corresponding YAML fields). For an official OpenAI base model, set its model ID, set `api.api_key_env: OPENAI_API_KEY`, and leave `api.base_url: null`. Separate keys and URLs permit different providers for each role. The critic remains configurable: the paper names GPT-5, while the preserved checked-in model setting is `gpt-4o`.

For example, with a compatible Qwen endpoint and official OpenAI critic/embeddings:

```bash
export ECR_BASE_URL=https://your-compatible-service.example/v1
export ECR_BASE_API_KEY=your-base-model-key
export OPENAI_API_KEY=your-openai-key
```

These are placeholders, not issued credentials. `python-dotenv` can load a local `.env`, which is excluded by `.gitignore`.

Paths in YAML are resolved relative to the YAML file. CLI paths are resolved relative to the caller's current directory; absolute paths also work. Offline output paths are `offline_training.golden_graph_dir`, `memory_bank_dir`, and `log_dir`, or a single `run_cgme.py --output-dir` override. Online paths are `online_inference.golden_graph_dir` (raw), `refined_graph_dir`, `memory_bank_dir`, and `output_dir`; corresponding CLI overrides are `--raw-graph-dir`, `--refined-graph-dir`, `--memory-bank-dir`, and `--output-dir`. Graph refinement accepts `--input-dir`, `--output-dir`, and `--log-dir`.

CGME requires `--reference-set` and defaults to `data/medeinst_support_853.json`; Online inference requires `--benchmark-set`. See [dataset_interface.md](dataset_interface.md). PubMed uses `PUBMED_EMAIL` and optional `PUBMED_API_KEY`; OpenTargets uses its existing public endpoint. This stage did not contact any external service.
