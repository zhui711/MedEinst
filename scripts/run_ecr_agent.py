#!/usr/bin/env python3
"""Run full ECR-Agent inference on MedEinst case pairs."""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.graph.state import AgentState
from src.agents.phase1_manager import Phase1Manager
from src.agents.phase2_hybrid_engine import Phase2HybridEngine
from src.agents.phase3_memory_judge import create_judge_agent
from src.memory.memory_bank import MemoryBankManager
from src.memory.embedding_client import EmbeddingClient
from src.utils.graph_tools import summarize_graph_for_judge, calculate_naive_scores
from src.data.medeinst_adapter import CasePair, load_pairs
from src.utils.runtime_config import configured_base_url, config_relative_path


@dataclass
class OnlineResult:
    """One case-level inference result."""
    case_id: str
    case_type: str
    status: str  # "success" | "error"
    ground_truth_id: Optional[str] = None
    ground_truth_name: Optional[str] = None
    phase1_diagnosis_id: Optional[str] = None
    phase1_diagnosis_name: Optional[str] = None
    final_diagnosis_id: Optional[str] = None
    final_diagnosis_name: Optional[str] = None
    verdict_status: Optional[str] = None  # "Confirm" | "Overturn" | "Fallback"
    is_correct: bool = False
    processing_time: float = 0.0
    error_message: Optional[str] = None
    hybrid_stats: Optional[Dict[str, int]] = None
    memory_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OnlineOutputManager:
    """Persist Online results, summaries, and case graphs."""

    def __init__(self, output_dir: str = "output_online"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.graphs_dir = self.output_dir / "graphs"
        self.graphs_dir.mkdir(exist_ok=True)

        self.results_file = self.output_dir / "results_detail.jsonl"
        self.summary_file = self.output_dir / "inference_summary.json"
        self.error_log_file = self.output_dir / "error_log.jsonl"

        print(f"[OutputManager] Output directory: {self.output_dir}")

    def save_result(self, result: OnlineResult) -> None:

        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def save_graph(self, case_id: str, graph_json: Dict[str, Any]) -> None:

        filepath = self.graphs_dir / f"{case_id}_online_graph.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)

    def save_error(self, case_id: str, case_type: str, error_message: str) -> None:

        error_record = {
            "case_id": case_id,
            "case_type": case_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    def save_summary(self, summary: Dict[str, Any]) -> None:

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def load_processed_cases(self) -> Dict[str, str]:

        processed = {}
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            key = f"{record['case_id']}_{record['case_type']}"
                            processed[key] = record.get("status", "unknown")
                        except:
                            pass
        return processed

    def get_error_cases(self) -> List[Tuple[str, str]]:

        error_cases = []
        processed = self.load_processed_cases()

        for key, status in processed.items():
            if status == "error":
                for variant in ("control_case", "trap_case"):
                    suffix = f"_{variant}"
                    if key.endswith(suffix):
                        error_cases.append((key[:-len(suffix)], variant))
                        break

        return error_cases

class OnlineInferenceRunner:
    """Coordinate the full Online DCI pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_manager: OnlineOutputManager,
        golden_graph_dir: str = "golden_graphs",
        refined_graph_dir: str = "golden_graphs_refined",
        memory_bank_dir: str = "memory_bank",
        parallel: bool = False,
        max_workers: int = 4,
        config_path: Optional[str] = None
    ):

        self.config = config
        self.output_manager = output_manager
        self.parallel = parallel
        self.max_workers = max_workers

        api_config = config.get("api", {})
        self.llm_client = LLMClient(
            base_url=configured_base_url(api_config),
            api_key_env=api_config.get("api_key_env", "OPENAI_API_KEY"),
            timeout=api_config.get("timeout", 120)
        )
        self.model_name = api_config.get("model_name", "qwen3-32b")
        embedding_config = config.get("embedding", {})
        embedding_client = EmbeddingClient(
            base_url=configured_base_url(embedding_config),
            api_key_env=embedding_config.get("api_key_env", "OPENAI_API_KEY"),
            model=embedding_config.get("model_name", "text-embedding-3-small"),
            dimension=embedding_config.get("dimension", 1536)
        )

        self.phase1 = Phase1Manager(self.llm_client, self.model_name)

        # Online DCI starts from Illness Graphs, then extends them with live evidence.
        self.phase2 = Phase2HybridEngine(
            llm_client=self.llm_client,
            model_name=self.model_name,
            golden_graph_dir=golden_graph_dir,
            refined_graph_dir=refined_graph_dir,
            embedding_client=embedding_client,
            config_path=config_path
        )

        self.memory_bank = MemoryBankManager(
            output_dir=memory_bank_dir, embedding_client=embedding_client
        )
        try:
            self.memory_bank.load()
            print(f"[OnlineRunner] Memory Bank loaded: {self.memory_bank.get_statistics()}")
        except Exception as e:
            print(f"[OnlineRunner] Warning: Failed to load Memory Bank: {e}")
            self.memory_bank = None

        self.phase3 = create_judge_agent(
            llm_client=self.llm_client,
            model_name=self.model_name,
            memory_bank=self.memory_bank,
            memory_bank_dir=memory_bank_dir
        )

        print(f"[OnlineRunner] Initialized with:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Raw Illness Graphs: {golden_graph_dir}")
        print(f"  - Memory Bank: {'Enabled' if self.memory_bank else 'Disabled'}")
        print(f"  - Parallel: {parallel} (workers: {max_workers})")

    def process_single_case(
        self,
        case_dir: CasePair,
        case_type: str
    ) -> OnlineResult:

        case_id = case_dir.name
        start_time = time.time()

        try:
            case_info = case_dir.get(case_type)
            narrative = case_info.narrative
            ground_truth = case_info.ground_truth
            gt_id = case_info.ground_truth_id

            # Phase 1
            phase1_result = self.phase1.process(narrative)

            if phase1_result.get("error"):
                raise Exception(f"Phase 1 error: {phase1_result['error']}")

            state: AgentState = {
                "case_id": case_id,
                "input_case": {
                    "narrative": narrative,
                    "ground_truth": ground_truth,
                    "ground_truth_id": gt_id
                },
                "phase1_result": phase1_result,
                "graph_json": None,
                "graph_summary": None,
                "naive_scores": None,
                "final_output": None,
                "status": "processing",
                "error_log": None,
                "global_hint": None,
                "retry_count": 0,
                "memory_context": None,
                "memory_records": None
            }

            # Phase 2 (Hybrid Engine)
            state = self.phase2.process(state)

            if state.get("status") == "failed":
                raise Exception(f"Phase 2 error: {state.get('error_log')}")

            graph_json = state.get("graph_json", {})

            graph_summary = summarize_graph_for_judge(graph_json)
            state["graph_summary"] = graph_summary

            naive_scores = calculate_naive_scores(graph_json)
            state["naive_scores"] = naive_scores

            # Phase 3 (Memory-Augmented Judge)
            state = self.phase3.process(state)

            if state.get("status") == "failed":
                raise Exception(f"Phase 3 error: {state.get('error_log')}")

            final_output = state.get("final_output", {})
            final_id = final_output.get("final_diagnosis_id")
            final_name = final_output.get("final_diagnosis_name")
            verdict_status = final_output.get("status")

            phase1_id = phase1_result.get("final_diagnosis_id")
            phase1_name = DIAGNOSIS_ID_MAP.get(phase1_id, "Unknown")

            is_correct = (final_id == gt_id) if gt_id else False

            self.output_manager.save_graph(f"{case_id}_{case_type}", graph_json)

            processing_time = time.time() - start_time

            hybrid_stats = graph_json.get("hybrid_engine_stats", {})
            memory_used = final_output.get("memory_retrieval_used", False)

            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="success",
                ground_truth_id=gt_id,
                ground_truth_name=ground_truth,
                phase1_diagnosis_id=phase1_id,
                phase1_diagnosis_name=phase1_name,
                final_diagnosis_id=final_id,
                final_diagnosis_name=final_name,
                verdict_status=verdict_status,
                is_correct=is_correct,
                processing_time=processing_time,
                error_message=None,
                hybrid_stats=hybrid_stats,
                memory_used=memory_used
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"

            self.output_manager.save_error(case_id, case_type, error_msg)

            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="error",
                processing_time=processing_time,
                error_message=str(e)
            )

    def run(
        self,
        case_dirs: List[CasePair],
        case_types: List[str],
        retry_failed: bool = False
    ) -> List[OnlineResult]:

        tasks = []

        if retry_failed:

            error_cases = self.output_manager.get_error_cases()
            for case_id, case_type in error_cases:
                if case_type not in case_types:
                    continue
                for case_dir in case_dirs:
                    if case_dir.name == case_id:
                        tasks.append((case_dir, case_type))
                        break
            print(f"[OnlineRunner] Retrying {len(tasks)} failed cases")
        else:

            processed = self.output_manager.load_processed_cases()

            for case_dir in case_dirs:
                for case_type in case_types:
                    key = f"{case_dir.name}_{case_type}"
                    if key not in processed or processed[key] == "error":
                        tasks.append((case_dir, case_type))

            print(f"[OnlineRunner] Processing {len(tasks)} cases")

        if not tasks:
            print("[OnlineRunner] No cases to process")
            return []

        if self.parallel:
            results = self._run_parallel(tasks)
        else:
            results = self._run_sequential(tasks)

        return results

    def _run_sequential(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:

        results = []

        with tqdm(total=len(tasks), desc="Online Inference") as pbar:
            for case_dir, case_type in tasks:
                result = self.process_single_case(case_dir, case_type)
                results.append(result)

                self.output_manager.save_result(result)

                status_icon = "✓" if result.status == "success" else "✗"
                correct_icon = "🎯" if result.is_correct else ""
                pbar.set_postfix_str(f"{status_icon} {result.case_id} {correct_icon}")
                pbar.update(1)

        return results

    def _run_parallel(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:

        results = []

        with tqdm(total=len(tasks), desc="Online Inference (Parallel)") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.process_single_case, case_dir, case_type): (case_dir, case_type)
                    for case_dir, case_type in tasks
                }

                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                    except Exception as e:
                        case_dir, case_type = future_to_task[future]
                        result = OnlineResult(
                            case_id=case_dir.name,
                            case_type=case_type,
                            status="error",
                            error_message=str(e)
                        )

                    results.append(result)
                    self.output_manager.save_result(result)

                    status_icon = "✓" if result.status == "success" else "✗"
                    pbar.set_postfix_str(f"{status_icon} {result.case_id}")
                    pbar.update(1)

        return results

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:

    config_path = Path(config_path or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():

    parser = argparse.ArgumentParser(
        description="ECR-Agent Online DCI with Illness Graphs and Exemplar Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all Benchmark Test Set cases
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl
  
  # Limit the number of pairs
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --limit 10
  
  # Process cases concurrently
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --parallel --workers 4
  
  # Retry failed cases
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --retry-failed
        """
    )

    parser.add_argument(
        "--benchmark-set",
        type=str,
        required=True,
        help="Public MedEinst test.jsonl (Benchmark Test Set), or a paired-case directory"
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML configuration file (default: config/settings.yaml)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides online_inference.output_dir)"
    )

    parser.add_argument(
        "--raw-graph-dir",
        type=str,
        default=None,
        help="Raw Illness Graph directory (overrides online_inference.golden_graph_dir)"
    )

    parser.add_argument(
        "--refined-graph-dir", type=str, default=None,
        help="Refined Illness Graph directory (overrides online_inference.refined_graph_dir)"
    )

    parser.add_argument(
        "--memory-bank-dir",
        type=str,
        default=None,
        help="Exemplar Base directory (overrides online_inference.memory_bank_dir)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of case pairs to process"
    )

    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Comma-separated case pair IDs to process"
    )

    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Skip control cases"
    )

    parser.add_argument(
        "--skip-trap",
        action="store_true",
        help="Skip trap cases"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process cases concurrently"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)"
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only previously failed cases"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the base model ID in the configuration"
    )

    return parser.parse_args()

def main():

    load_dotenv()

    args = parse_args()

    config_path = Path(args.config or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    config = load_config(config_path)
    online_config = config.get("online_inference", {})

    def runtime_path(cli_value: Optional[str], key: str, default: str) -> str:
        if cli_value:
            return str(Path(cli_value).expanduser().resolve())
        return config_relative_path(online_config.get(key, default), config_path)

    output_dir = runtime_path(args.output_dir, "output_dir", "../output_online")
    raw_graph_dir = runtime_path(args.raw_graph_dir, "golden_graph_dir", "../golden_graphs")
    refined_graph_dir = runtime_path(args.refined_graph_dir, "refined_graph_dir", "../golden_graphs_refined")
    memory_bank_dir = runtime_path(args.memory_bank_dir, "memory_bank_dir", "../memory_bank")

    if args.model:
        config["api"]["model_name"] = args.model

    output_manager = OnlineOutputManager(output_dir)

    # Load the Benchmark Test Set, preserving original pair IDs and narratives.
    input_path = Path(args.benchmark_set).expanduser()
    case_dirs = load_pairs(input_path)

    if args.case_ids:
        specified_ids = {item.strip() for item in args.case_ids.split(",")}
        missing_ids = specified_ids - {pair.case_id for pair in case_dirs}
        if missing_ids:
            raise ValueError(f"Pair IDs absent from Benchmark Test Set: {sorted(missing_ids)}")
        case_dirs = [pair for pair in case_dirs if pair.case_id in specified_ids]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        case_dirs = case_dirs[:args.limit]

    case_types = []
    if not args.skip_control:
        case_types.append("control_case")
    if not args.skip_trap:
        case_types.append("trap_case")

    if not case_types:
        print("❌ No case types to process")
        sys.exit(1)

    print("=" * 70)
    print("ECR-Agent Online DCI (Illness Graphs + Exemplar Base)")
    print("=" * 70)
    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_manager.output_dir}")
    print(f"📁 Raw Illness Graphs: {raw_graph_dir}")
    print(f"📁 Refined Illness Graphs: {refined_graph_dir}")
    print(f"📁 Exemplar Base: {memory_bank_dir}")
    print(f"📊 Benchmark pairs selected: {len(case_dirs)}")
    print(f"📋 Case types: {', '.join(case_types)}")
    print(f"🔧 Model: {config['api']['model_name']}")
    print("📊 Illness Graphs: Enabled")
    print("🧠 Exemplar Base: Enabled")
    print(f"⚡ Parallel mode: {'Yes' if args.parallel else 'No'}")
    if args.parallel:
        print(f"👷 Workers: {args.workers}")
    print("=" * 70)

    try:
        runner = OnlineInferenceRunner(
            config=config,
            output_manager=output_manager,
            golden_graph_dir=raw_graph_dir,
            refined_graph_dir=refined_graph_dir,
            memory_bank_dir=memory_bank_dir,
            parallel=args.parallel,
            max_workers=args.workers,
            config_path=str(config_path)
        )
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        traceback.print_exc()
        sys.exit(1)

    start_time = time.time()
    results = runner.run(
        case_dirs=case_dirs,
        case_types=case_types,
        retry_failed=args.retry_failed
    )
    total_time = time.time() - start_time

    success_count = sum(1 for r in results if r.status == "success")
    error_count = sum(1 for r in results if r.status == "error")
    correct_count = sum(1 for r in results if r.is_correct)

    control_results = [r for r in results if r.case_type == "control_case"]
    trap_results = [r for r in results if r.case_type == "trap_case"]

    control_correct = sum(1 for r in control_results if r.is_correct)
    trap_correct = sum(1 for r in trap_results if r.is_correct)

    overturn_count = sum(1 for r in results if r.verdict_status == "Overturn")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
        "success": success_count,
        "error": error_count,
        "correct": correct_count,
        "accuracy": correct_count / success_count if success_count > 0 else 0,
        "control_accuracy": control_correct / len(control_results) if control_results else 0,
        "trap_accuracy": trap_correct / len(trap_results) if trap_results else 0,
        "overturn_count": overturn_count,
        "total_time_seconds": total_time,
        "avg_time_per_case": total_time / len(results) if results else 0,
        "config": {
            "model": config["api"]["model_name"],
            "use_memory": True,
            "golden_graph_dir": raw_graph_dir,
            "refined_graph_dir": refined_graph_dir,
            "memory_bank_dir": memory_bank_dir,
            "parallel": args.parallel,
            "workers": args.workers if args.parallel else 1
        }
    }
    output_manager.save_summary(summary)

    print("\n" + "=" * 70)
    print("Online Inference Complete!")
    print("=" * 70)
    print(f"✅ Success: {success_count}")
    print(f"❌ Error: {error_count}")
    print(f"🎯 Correct: {correct_count}/{success_count} ({summary['accuracy']:.2%})")
    if control_results:
        print(f"  - Control: {control_correct}/{len(control_results)} ({summary['control_accuracy']:.2%})")
    if trap_results:
        print(f"  - Trap: {trap_correct}/{len(trap_results)} ({summary['trap_accuracy']:.2%})")
    print(f"🔄 Overturn: {overturn_count}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Results saved to: {output_manager.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

