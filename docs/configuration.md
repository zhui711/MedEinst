# ECR-Agent Configuration

Model, API, and output settings are defined in [`config/settings.yaml`](../config/settings.yaml). The CGME, Online inference, and Illness Graph refinement commands load this file by default; pass `--config /path/to/settings.yaml` to use a different configuration.

Credentials are supplied through environment variables rather than stored in the YAML file.

## Model endpoints

| YAML section | Default model | Credential environment variable | Optional endpoint override |
| --- | --- | --- | --- |
| `api` | `qwen3-32b` | `ECR_BASE_API_KEY` | `ECR_BASE_URL` |
| `critical_model` | `gpt-4o` | `OPENAI_API_KEY` | `ECR_CRITIC_URL` |
| `embedding` | `text-embedding-3-small` (1536 dimensions) | `OPENAI_API_KEY` | `ECR_EMBEDDING_URL` |

Each section supports `base_url` and `api_key_env` settings. An environment endpoint override takes precedence over the section's `base_url`. If no custom URL is configured, the OpenAI API endpoint is used.

The default `qwen3-32b` model requires a compatible model-serving endpoint. For example, when using a separately hosted base model and OpenAI for critic and embedding requests:

```bash
export ECR_BASE_URL=https://your-compatible-endpoint.example/v1
export ECR_BASE_API_KEY=your-base-model-key
export OPENAI_API_KEY=your-openai-key
```

Adjust model IDs and endpoints to match the available services. The paper uses GPT-5 as the critic model; the example configuration defaults to `gpt-4o` and can be changed in `critical_model.model_name`.

A local `.env` file may also be loaded through `python-dotenv`; it should not be committed to the repository.

## Data and output paths

Relative paths in YAML are resolved relative to the YAML file. Relative CLI paths are resolved from the current working directory; absolute paths are also supported.

| Use | Configuration / CLI |
| --- | --- |
| Reference Set for CGME | `run_cgme.py --reference-set` |
| Fixed support manifest | `run_cgme.py --support-manifest` (default `data/medeinst_support_853.json`) |
| Benchmark Test Set | `run_ecr_agent.py --benchmark-set` |
| Raw Illness Graphs | `online_inference.golden_graph_dir` / `--raw-graph-dir` |
| Refined Illness Graphs | `online_inference.refined_graph_dir` / `--refined-graph-dir` |
| Exemplar Base | `online_inference.memory_bank_dir` / `--memory-bank-dir` |
| Online results | `online_inference.output_dir` / `--output-dir` |
| Offline CGME outputs | `offline_training.golden_graph_dir`, `memory_bank_dir`, `log_dir`; or `run_cgme.py --output-dir` |

Graph refinement supports `--input-dir`, `--output-dir`, and `--log-dir`.

## External knowledge sources

PubMed retrieval uses `PUBMED_EMAIL` and optionally `PUBMED_API_KEY`. OpenTargets retrieval uses its configured public service endpoint.

For data formats and artifact directories, see [Dataset Interface](dataset_interface.md) and [Artifacts](artifacts.md).
