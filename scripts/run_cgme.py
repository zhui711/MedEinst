#!/usr/bin/env python3
"""Run offline CGME on the fixed MedEinst support subset."""

import sys
import argparse
from pathlib import Path
from datetime import datetime


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml
from tqdm import tqdm

from src.training.offline_trainer import OfflineTrainer, ConcurrentOfflineTrainer
from src.utils.api_client import LLMClient
from src.memory.embedding_client import EmbeddingClient
from src.data.medeinst_adapter import load_pairs, resolve_support_manifest
from src.utils.runtime_config import configured_base_url, config_relative_path


def load_config(config_path: str = None) -> dict:

    if config_path is None:
        config_path = project_root / "config" / "settings.yaml"
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    offline = config.get("offline_training", {})
    for key in ("golden_graph_dir", "memory_bank_dir", "log_dir"):
        if key in offline:
            offline[key] = config_relative_path(offline[key], config_path)
    return config

def main():

    parser = argparse.ArgumentParser(description="Run Offline CGME on MedEinst support cases")
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        help="Only process first N cases (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--reference-set",
        type=str,
        required=True,
        help="Public MedEinst train.jsonl (Reference Set), or a paired-case directory"
    )
    parser.add_argument(
        "--support-manifest",
        type=str,
        default=str(project_root / "data" / "medeinst_support_853.json"),
        help="Fixed CGME support manifest (default: data/medeinst_support_853.json)"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Enable concurrent processing mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for concurrent processing (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides config)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ECR-Agent Offline CGME")
    print("=" * 70)
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Dry Run: {args.dry_run if args.dry_run else 'No'}")
    print(f"Resume: {args.resume}")
    print(f"Mode: {'Concurrent' if args.concurrent else 'Sequential'}")
    print("=" * 70)

    # Load Config
    print("\n[Setup] Loading configuration...")
    config = load_config(args.config)

    api_config = config.get("api", {})
    critical_config = config.get("critical_model", {})
    embedding_config = config.get("embedding", {})
    offline_config = config.get("offline_training", {})

    # Override config with --output-dir if provided
    if args.output_dir:
        output_path = Path(args.output_dir).expanduser().resolve()
        print(f"[Override] Using custom output directory: {output_path}")
        offline_config["golden_graph_dir"] = str(output_path / "golden_graphs")
        offline_config["memory_bank_dir"] = str(output_path / "memory_bank")
        offline_config["log_dir"] = str(output_path / "logs")

    print(f"  - Main Model: {api_config.get('model_name', 'N/A')}")
    print(f"  - Critical Model: {critical_config.get('model_name', 'N/A')}")
    print(f"  - Embedding Model: {embedding_config.get('model_name', 'N/A')}")
    print(f"  - Max Retries: {offline_config.get('max_critique_retries', 3)}")
    print(f"  - Aggregation Step: {offline_config.get('aggregation_step', 5)}")

    # Resolve the fixed support records against the public Reference Set.
    print("\n[Setup] Resolving MedEinst support manifest...")
    reference_pairs = load_pairs(args.reference_set)
    train_list = resolve_support_manifest(args.support_manifest, reference_pairs)
    reference_by_id = {pair.case_id: pair for pair in reference_pairs}

    def load_case_narrative(case_id: str, variant: str) -> str:
        return reference_by_id[case_id].get(variant).narrative

    print(f"  - Reference Set pairs: {len(reference_pairs)}")
    print(f"  - Fixed support records: {len(train_list)}")

    if args.dry_run:
        train_list = train_list[:args.dry_run]
        print(f"  - Dry-run mode: processing only {len(train_list)} cases")

    print("\n[Setup] Initializing LLM clients...")

    main_client = LLMClient(
        base_url=configured_base_url(api_config),
        api_key_env=api_config.get("api_key_env", "OPENAI_API_KEY"),
        timeout=api_config.get("timeout", 120)
    )

    critical_client = LLMClient(
        base_url=configured_base_url(critical_config),
        api_key_env=critical_config.get("api_key_env", "OPENAI_API_KEY"),
        timeout=critical_config.get("timeout", 180)
    )

    try:
        embedding_client = EmbeddingClient(
            base_url=configured_base_url(embedding_config),
            api_key_env=embedding_config.get("api_key_env", "OPENAI_API_KEY"),
            model=embedding_config.get("model_name", "text-embedding-3-small"),
            dimension=embedding_config.get("dimension", 1536)
        )
        print("  - Embedding client initialized")
    except Exception as e:
        print(f"  - Warning: Embedding client init failed: {e}")
        print("  - Memory Bank will use fallback embeddings")
        embedding_client = None

    max_workers = args.workers or offline_config.get("max_workers", 5)

    if args.concurrent:
        print(f"\n[Setup] Initializing ConcurrentOfflineTrainer (workers={max_workers})...")
        trainer = ConcurrentOfflineTrainer(
            max_workers=max_workers,
            llm_client=main_client,
            critical_client=critical_client,
            embedding_client=embedding_client,
            model_name=api_config.get("model_name", "qwen3-32b"),
            critical_model_name=critical_config.get("model_name", "gpt-5"),
            max_retries=offline_config.get("max_critique_retries", 3),
            aggregation_step=offline_config.get("aggregation_step", 5),
            golden_graph_dir=offline_config.get("golden_graph_dir", "golden_graphs"),
            memory_bank_dir=offline_config.get("memory_bank_dir", "memory_bank"),
            log_dir=offline_config.get("log_dir", "logs")
        )

    if not args.concurrent:
        print("\n[Setup] Initializing OfflineTrainer (Sequential)...")
        trainer = OfflineTrainer(
            llm_client=main_client,
            critical_client=critical_client,
            embedding_client=embedding_client,
            model_name=api_config.get("model_name", "qwen3-32b"),
            critical_model_name=critical_config.get("model_name", "gpt-5"),
            max_retries=offline_config.get("max_critique_retries", 3),
            aggregation_step=offline_config.get("aggregation_step", 5),
            golden_graph_dir=offline_config.get("golden_graph_dir", "golden_graphs"),
            memory_bank_dir=offline_config.get("memory_bank_dir", "memory_bank"),
            log_dir=offline_config.get("log_dir", "logs"),
        )

    if args.resume:
        print("\n[Setup] Loading checkpoint...")
        trainer.load_checkpoint()

    pending_cases = [
        case for case in train_list
        if not trainer.is_processed(case["case_id"])
    ]
    print(f"\n[Setup] Cases to process: {len(pending_cases)} (skipped: {len(train_list) - len(pending_cases)})")

    if not pending_cases:
        print("\n[Info] All cases already processed!")
        return

    print("\n" + "=" * 70)
    print(f"Starting Training Loop ({'Concurrent' if args.concurrent else 'Sequential'})")
    print("=" * 70 + "\n")

    if args.concurrent:

        batch_size = args.batch_size
        total_batches = (len(pending_cases) + batch_size - 1) // batch_size

        print(f"[Concurrent] Processing {len(pending_cases)} cases in {total_batches} batches (size={batch_size})")

        pbar = tqdm(total=len(pending_cases), desc="Training", position=0, leave=True)

        def progress_callback(completed, total, result):

            status_emoji = "✓" if result.get("final_status") == "success" else "✗"
            pbar.set_postfix_str(f"Last: {status_emoji} {result.get('case_id', 'N/A')[:20]}")
            pbar.update(1)

        all_results = []

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pending_cases))
            batch_cases = pending_cases[start_idx:end_idx]

            print(f"\n[Batch {batch_idx + 1}/{total_batches}] Processing {len(batch_cases)} cases...")

            batch_results = trainer.train_batch(
                cases=batch_cases,
                load_narrative_fn=load_case_narrative,
                progress_callback=progress_callback
            )

            all_results.extend(batch_results)

            success_count = sum(1 for r in batch_results if r.get("final_status") == "success")
            failed_count = sum(1 for r in batch_results if r.get("final_status") == "failed")
            error_count = sum(1 for r in batch_results if r.get("final_status") == "error")
            print(f"    Batch result: ✓ {success_count} success | ✗ {failed_count} failed | ⚠ {error_count} errors")

        pbar.close()

    else:

        for idx, case_record in enumerate(tqdm(pending_cases, desc="Training")):
            case_id = case_record["case_id"]
            case_type = case_record["type"]
            ground_truth = case_record["ground_truth"]
            ground_truth_id = case_record["ground_truth_id"]
            case_path = case_record["path"]

            try:

                narrative = load_case_narrative(case_path, case_type)

                if not narrative:
                    print(f"\n[Warning] Empty narrative for {case_id}, skipping...")
                    continue

                result = trainer.process_case(
                    case_id=case_id,
                    case_type=case_type,
                    ground_truth_id=ground_truth_id,
                    ground_truth_name=ground_truth,
                    narrative=narrative
                )

                trainer.log_result(result)

                status_emoji = "✓" if result["final_status"] == "success" else "✗"
                print(f"\n[{idx+1}/{len(pending_cases)}] {status_emoji} {case_id}: {result['final_status']}")

            except Exception as e:
                print(f"\n[Error] Failed to process {case_id}: {e}")
                import traceback
                traceback.print_exc()

                error_result = {
                    "case_id": case_id,
                    "type": case_type,
                    "ground_truth_id": ground_truth_id,
                    "ground_truth_name": ground_truth,
                    "timestamp": datetime.now().isoformat(),
                    "attempts": [],
                    "final_status": "error",
                    "error": str(e),
                    "total_retries": 0,
                    "aggregated": False
                }
                trainer.log_result(error_result)
                continue

    print("\n" + "=" * 70)
    print("Training Complete - Finalizing")
    print("=" * 70 + "\n")

    final_stats = trainer.finalize()

    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    print(f"  - Total Processed: {final_stats.get('total_processed', 0)}")
    print(f"  - Success (First Try): {final_stats.get('success_first_try', 0)}")
    print(f"  - Success (After Retry): {final_stats.get('success_after_retry', 0)}")
    print(f"  - Failed: {final_stats.get('failed', 0)}")

    if "aggregation" in final_stats:
        agg = final_stats["aggregation"]
        print(f"\n  Illness Graphs:")
        print(f"    - Unique Pathologies: {agg.get('unique_pathologies', 0)}")
        print(f"    - Total Aggregated: {agg.get('total_aggregated', 0)}")

    if "memory_bank" in final_stats:
        mb = final_stats["memory_bank"]
        print(f"\n  Memory Bank:")
        print(f"    - Total Cases: {mb.get('total_cases', 0)}")
        print(f"    - Overturn: {mb.get('overturn_count', 0)}")
        print(f"    - Confirm: {mb.get('confirm_count', 0)}")

    print("\n" + "=" * 70)
    print(f"End Time: {datetime.now().isoformat()}")
    print("=" * 70)

if __name__ == "__main__":
    main()

