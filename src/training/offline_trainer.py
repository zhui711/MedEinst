"""Run critic-guided CGME and persist successful experience."""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.graph.state import AgentState
from src.agents.phase1_manager import Phase1Manager
from src.agents.phase2_investigation import Phase2GraphReasoning
from src.agents.phase3_debate import JudgeAgent
from src.agents.critical_model import CriticalModelAgent
from src.training.aggregation_manager import AggregationManager
from src.memory.memory_bank import MemoryBankManager
from src.memory.embedding_client import EmbeddingClient
from src.utils.api_client import LLMClient
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.utils.graph_tools import (
    rebuild_graph_from_json,
    serialize_graph_to_summary,
    calculate_deterministic_score
)


class OfflineTrainer:
    """Run serial CGME over support cases."""

    def __init__(
        self,
        llm_client: LLMClient,
        critical_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        model_name: str = "qwen3-32b",
        critical_model_name: str = "gpt-5",
        max_retries: int = 3,
        aggregation_step: int = 5,
        golden_graph_dir: str = "golden_graphs",
        memory_bank_dir: str = "memory_bank",
        log_dir: str = "logs"
    ):

        self.phase1_manager = Phase1Manager(llm_client, model_name=model_name)
        # CGME bootstraps illness knowledge, so it reasons from live evidence before graphs exist.
        self.phase2_agent = Phase2GraphReasoning(llm_client, model_name=model_name)
        self.judge_agent = JudgeAgent(llm_client, model_name=model_name)

        critical_client = critical_client or llm_client
        self.critical_agent = CriticalModelAgent(
            critical_client,
            model_name=critical_model_name
        )

        self.aggregation_manager = AggregationManager(
            aggregation_step=aggregation_step,
            output_dir=golden_graph_dir
        )

        self.memory_bank = MemoryBankManager(
            output_dir=memory_bank_dir,
            embedding_client=embedding_client
        )

        self.max_retries = max_retries
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._log_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._processed_lock = threading.Lock()

        self._pending_aggregations: List[Dict[str, Any]] = []
        self._aggregation_lock = threading.Lock()

        self.stats = {
            "total_processed": 0,
            "success_first_try": 0,
            "success_after_retry": 0,
            "failed": 0,
            "errors": 0
        }

        self.processed_cases: set = set()

    def load_checkpoint(self, log_file: Optional[str] = None) -> None:

        log_path = Path(log_file) if log_file else self.log_dir / "offline_training_log.jsonl"

        if not log_path.exists():
            print(f"[Trainer] No checkpoint file found at {log_path}")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        case_id = record.get("case_id")
                        if case_id:
                            self.processed_cases.add(case_id)

            print(f"[Trainer] Loaded checkpoint: {len(self.processed_cases)} cases already processed")

        except Exception as e:
            print(f"[Trainer] Error loading checkpoint: {e}")

    def process_case(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> Dict[str, Any]:

        result = {
            "case_id": case_id,
            "type": case_type,
            "ground_truth_id": ground_truth_id,
            "ground_truth_name": ground_truth_name,
            "timestamp": datetime.now().isoformat(),
            "attempts": [],
            "final_status": "pending",
            "total_retries": 0,
            "aggregated": False
        }

        self.critical_agent.reset_history()

        state = self._create_initial_state(case_id, narrative)

        print(f"\n[Trainer] Processing {case_id} (GT: {ground_truth_name})...")

        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1
            print(f"[Trainer] Attempt {attempt_num}/{self.max_retries + 1}")

            try:

                state = self._run_pipeline(state)

                is_correct, error_type = self._check_result(
                    state, ground_truth_id
                )

                p1_result = state.get("phase1_result") or {}
                f_output = state.get("final_output") or {}
                attempt_record = {
                    "attempt": attempt_num,
                    "phase1_top5": p1_result.get("top_candidates", []) or [],
                    "initial_diagnosis_id": p1_result.get("final_diagnosis_id", "") or "",
                    "final_diagnosis_id": f_output.get("final_diagnosis_id", "") or "",
                    "correct": is_correct,
                    "error_type": error_type,
                    "critique": None
                }

                if is_correct:

                    print(f"[Trainer] ✓ Correct diagnosis on attempt {attempt_num}")

                    result["attempts"].append(attempt_record)
                    result["final_status"] = "success"
                    result["total_retries"] = attempt

                    self._on_success(state, case_id, case_type, ground_truth_id, ground_truth_name)
                    result["aggregated"] = True

                    if attempt == 0:
                        self.stats["success_first_try"] += 1
                    else:
                        self.stats["success_after_retry"] += 1

                    return result

                else:

                    print(f"[Trainer] ✗ Incorrect. Error type: {error_type}")

                    if attempt < self.max_retries:

                        critique = self.critical_agent.generate_critique(
                            narrative=narrative,
                            ground_truth_id=ground_truth_id,
                            phase1_result=state.get("phase1_result", {}),
                            final_output=state.get("final_output", {}),
                            graph_json=state.get("graph_json", {}),
                            previous_critiques=self.critical_agent.get_history()
                        )

                        critique_text = self.critical_agent.get_injectable_critique(critique)
                        self.critical_agent.add_to_history(critique_text)

                        attempt_record["critique"] = critique_text

                        state = self._inject_hint(state, critique_text)

                        print(f"[Trainer] Injected critique for retry")

                    result["attempts"].append(attempt_record)

            except Exception as e:
                print(f"[Trainer] Error in attempt {attempt_num}: {e}")
                import traceback
                traceback.print_exc()

                result["attempts"].append({
                    "attempt": attempt_num,
                    "error": str(e)
                })

                if attempt >= self.max_retries:
                    break

        print(f"[Trainer] ✗ Failed after {self.max_retries + 1} attempts")
        result["final_status"] = "failed"
        result["total_retries"] = self.max_retries
        self.stats["failed"] += 1

        return result

    def _create_initial_state(self, case_id: str, narrative: str) -> AgentState:

        return AgentState(
            case_id=case_id,
            input_case={"narrative": narrative},
            phase1_result=None,
            graph_json=None,
            graph_summary=None,
            naive_scores=None,
            final_output=None,
            status="pending",
            error_log=None,
            global_hint=None,
            retry_count=0,
            memory_context=None
        )

    def _inject_hint(self, state: AgentState, hint: str) -> AgentState:

        state["global_hint"] = hint
        state["retry_count"] = state.get("retry_count", 0) + 1

        state["phase1_result"] = None
        state["graph_json"] = None
        state["graph_summary"] = None
        state["naive_scores"] = None
        state["final_output"] = None
        state["status"] = "pending"

        return state

    def _run_pipeline(self, state: AgentState) -> AgentState:

        narrative = state["input_case"].get("narrative", "")
        global_hint = state.get("global_hint")

        # === Phase 1 ===
        print("[Trainer] Running Phase 1...")

        phase1_result = self.phase1_manager.process(
            narrative,
            global_hint=global_hint
        )
        state["phase1_result"] = phase1_result

        if phase1_result.get("error"):
            state["status"] = "failed"
            state["error_log"] = phase1_result.get("error")
            return state

        # === Phase 2 ===
        print("[Trainer] Running Phase 2...")
        state = self.phase2_agent.process(state)

        if state.get("status") == "failed":
            return state

        # === Graph Summary ===
        if state.get("graph_json"):
            try:
                graph = rebuild_graph_from_json(state["graph_json"])
                state["graph_summary"] = serialize_graph_to_summary(graph)
                state["naive_scores"] = calculate_deterministic_score(graph)
            except Exception as e:
                print(f"[Trainer] Graph summary error: {e}")

        # === Phase 3 ===
        print("[Trainer] Running Phase 3...")
        state = self.judge_agent.process(state)

        return state

    def _check_result(
        self,
        state: AgentState,
        ground_truth_id: str
    ) -> Tuple[bool, str]:

        gt_id = ground_truth_id.lstrip("d_")

        phase1_result = state.get("phase1_result") or {}
        top_candidates = phase1_result.get("top_candidates", []) or []
        top_candidates_clean = [c.lstrip("d_") for c in top_candidates if c]
        gt_in_top5 = gt_id in top_candidates_clean

        final_output = state.get("final_output") or {}
        final_id = final_output.get("final_diagnosis_id", "") or ""
        final_id_clean = final_id.lstrip("d_")
        is_correct = (final_id_clean == gt_id)

        if is_correct:
            return True, "none"
        elif not gt_in_top5:
            return False, "GT_NOT_IN_TOP5"
        else:
            return False, "FINAL_WRONG"

    def _on_success(
        self,
        state: AgentState,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str
    ) -> None:

        graph_json = state.get("graph_json", {})
        self.aggregation_manager.record_success(
            pathology_id=ground_truth_id,
            pathology_name=ground_truth_name,
            case_graph=graph_json,
            case_id=case_id
        )

        # The Exemplar Base stores a compact validated-case record, not a full transcript.
        phase1_result = state.get("phase1_result") or {}
        final_output = state.get("final_output") or {}
        track_b_output = phase1_result.get("track_b_output") or {}
        p_nodes = track_b_output.get("p_nodes", []) or []

        initial_diagnosis_id = phase1_result.get("final_diagnosis_id", "") or ""
        initial_diagnosis_name = DIAGNOSIS_ID_MAP.get(
            initial_diagnosis_id.lstrip("d_"), "Unknown"
        )

        final_diagnosis_id = final_output.get("final_diagnosis_id", "") or ""
        final_diagnosis_name = final_output.get("final_diagnosis_name", "") or "Unknown"

        self.memory_bank.add_case(
            case_id=case_id,
            case_type=case_type,
            ground_truth_id=ground_truth_id.lstrip("d_"),
            ground_truth_name=ground_truth_name,
            initial_diagnosis_id=initial_diagnosis_id.lstrip("d_"),
            initial_diagnosis_name=initial_diagnosis_name,
            final_diagnosis_id=final_diagnosis_id.lstrip("d_"),
            final_diagnosis_name=final_diagnosis_name,
            p_nodes=p_nodes,
            retry_count=state.get("retry_count", 0)
        )

    def log_result(self, result: Dict[str, Any]) -> None:

        log_path = self.log_dir / "offline_training_log.jsonl"

        log_record = {k: v for k, v in result.items() if not k.startswith("_")}

        with self._log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        with self._stats_lock:
            self.stats["total_processed"] += 1

        with self._processed_lock:
            self.processed_cases.add(result["case_id"])

    def finalize(self) -> Dict[str, Any]:

        print("\n[Trainer] Finalizing training...")

        aggregation_stats = self.aggregation_manager.finalize()

        self.memory_bank.save()

        final_stats = {
            **self.stats,
            "aggregation": aggregation_stats,
            "memory_bank": self.memory_bank.get_statistics()
        }

        stats_path = self.log_dir / "training_summary.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        print(f"[Trainer] Training complete!")
        print(f"    Total processed: {self.stats['total_processed']}")
        print(f"    Success (first try): {self.stats['success_first_try']}")
        print(f"    Success (after retry): {self.stats['success_after_retry']}")
        print(f"    Failed: {self.stats['failed']}")

        return final_stats

    def is_processed(self, case_id: str) -> bool:

        with self._processed_lock:
            return case_id in self.processed_cases

    def process_case_for_batch(
        self,
        case_id: str,
        case_type: str,
        ground_truth_id: str,
        ground_truth_name: str,
        narrative: str
    ) -> Dict[str, Any]:

        result = {
            "case_id": case_id,
            "type": case_type,
            "ground_truth_id": ground_truth_id,
            "ground_truth_name": ground_truth_name,
            "timestamp": datetime.now().isoformat(),
            "attempts": [],
            "final_status": "pending",
            "total_retries": 0,
            "aggregated": False,
            "_aggregation_data": None
        }

        local_critique_history = []

        state = self._create_initial_state(case_id, narrative)

        for attempt in range(self.max_retries + 1):
            attempt_num = attempt + 1

            try:

                state = self._run_pipeline(state)

                is_correct, error_type = self._check_result(state, ground_truth_id)

                p1_result = state.get("phase1_result") or {}
                f_output = state.get("final_output") or {}
                attempt_record = {
                    "attempt": attempt_num,
                    "phase1_top5": p1_result.get("top_candidates", []) or [],
                    "initial_diagnosis_id": p1_result.get("final_diagnosis_id", "") or "",
                    "final_diagnosis_id": f_output.get("final_diagnosis_id", "") or "",
                    "correct": is_correct,
                    "error_type": error_type,
                    "critique": None
                }

                if is_correct:
                    result["attempts"].append(attempt_record)
                    result["final_status"] = "success"
                    result["total_retries"] = attempt

                    result["_aggregation_data"] = {
                        "state": state,
                        "case_id": case_id,
                        "case_type": case_type,
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_name": ground_truth_name
                    }

                    with self._stats_lock:
                        if attempt == 0:
                            self.stats["success_first_try"] += 1
                        else:
                            self.stats["success_after_retry"] += 1

                    return result

                else:
                    if attempt < self.max_retries:

                        critique = self.critical_agent.generate_critique(
                            narrative=narrative,
                            ground_truth_id=ground_truth_id,
                            phase1_result=state.get("phase1_result", {}),
                            final_output=state.get("final_output", {}),
                            graph_json=state.get("graph_json", {}),
                            previous_critiques=local_critique_history
                        )

                        critique_text = self.critical_agent.get_injectable_critique(critique)
                        local_critique_history.append(critique_text)

                        attempt_record["critique"] = critique_text
                        state = self._inject_hint(state, critique_text)

                    result["attempts"].append(attempt_record)

            except Exception as e:
                result["attempts"].append({
                    "attempt": attempt_num,
                    "error": str(e)
                })
                if attempt >= self.max_retries:
                    break

        result["final_status"] = "failed"
        result["total_retries"] = self.max_retries

        with self._stats_lock:
            self.stats["failed"] += 1

        return result

    def batch_aggregate(self, results: List[Dict[str, Any]]) -> int:

        aggregated_count = 0

        for result in results:
            if result["final_status"] == "success" and result.get("_aggregation_data"):
                agg_data = result["_aggregation_data"]

                try:
                    self._on_success(
                        state=agg_data["state"],
                        case_id=agg_data["case_id"],
                        case_type=agg_data["case_type"],
                        ground_truth_id=agg_data["ground_truth_id"],
                        ground_truth_name=agg_data["ground_truth_name"]
                    )
                    result["aggregated"] = True
                    aggregated_count += 1

                except Exception as e:
                    print(f"[Trainer] Aggregation error for {result['case_id']}: {e}")

            if "_aggregation_data" in result:
                del result["_aggregation_data"]

        return aggregated_count

class ConcurrentOfflineTrainer(OfflineTrainer):
    """Run concurrent CGME over support cases."""

    def __init__(
        self,
        max_workers: int = 5,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.max_workers = max_workers

        print(f"[ConcurrentTrainer] Initialized with max_workers={max_workers}")

    def train_batch(
        self,
        cases: List[Dict[str, Any]],
        load_narrative_fn,
        progress_callback=None
    ) -> List[Dict[str, Any]]:

        all_results = []
        total = len(cases)
        completed = 0

        def process_single(case_record):

            case_id = case_record["case_id"]
            case_type = case_record["type"]
            ground_truth = case_record["ground_truth"]
            ground_truth_id = case_record["ground_truth_id"]
            case_path = case_record["path"]

            try:
                narrative = load_narrative_fn(case_path, case_type)

                if not narrative:
                    return {
                        "case_id": case_id,
                        "type": case_type,
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_name": ground_truth,
                        "timestamp": datetime.now().isoformat(),
                        "attempts": [],
                        "final_status": "error",
                        "error": "Empty narrative",
                        "total_retries": 0,
                        "aggregated": False,
                        "_aggregation_data": None
                    }

                return self.process_case_for_batch(
                    case_id=case_id,
                    case_type=case_type,
                    ground_truth_id=ground_truth_id,
                    ground_truth_name=ground_truth,
                    narrative=narrative
                )

            except Exception as e:
                return {
                    "case_id": case_id,
                    "type": case_type,
                    "ground_truth_id": ground_truth_id,
                    "ground_truth_name": ground_truth,
                    "timestamp": datetime.now().isoformat(),
                    "attempts": [],
                    "final_status": "error",
                    "error": str(e),
                    "total_retries": 0,
                    "aggregated": False,
                    "_aggregation_data": None
                }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_case = {
                executor.submit(process_single, case): case
                for case in cases
            }

            for future in as_completed(future_to_case):
                case = future_to_case[future]

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "case_id": case["case_id"],
                        "final_status": "error",
                        "error": str(e),
                        "_aggregation_data": None
                    }

                all_results.append(result)

                self.log_result(result)

                completed += 1
                if progress_callback:
                    progress_callback(completed, total, result)

        aggregated = self.batch_aggregate(all_results)

        self._incremental_save()

        print(f"[ConcurrentTrainer] Batch complete: {completed} processed, {aggregated} aggregated")

        return all_results

    def _incremental_save(self) -> None:

        try:

            self.memory_bank.save()

            current_stats = {
                **self.stats,
                "status": "in_progress",
                "aggregation": self.aggregation_manager.get_statistics() if hasattr(self.aggregation_manager, 'get_statistics') else {},
                "memory_bank": self.memory_bank.get_statistics()
            }

            stats_path = self.log_dir / "training_summary.json"
            with self._log_lock:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(current_stats, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[ConcurrentTrainer] Warning: Incremental save failed: {e}")

if __name__ == "__main__":
    print("OfflineTrainer module loaded successfully.")
    print("Classes available: OfflineTrainer, ConcurrentOfflineTrainer")
    print("To run training, use scripts/run_cgme.py")

