"""Final reranking and TREC evaluation."""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pytrec_eval

from .config import Config
from .data import load_qrels
from .utils import get_logger

LOGGER = get_logger(__name__)

BINARY_MEASURES = {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"}
GRADED_MEASURES = {"ndcg_cut.3"}


def rerank(cfg: Config, fused: Dict[str, Dict[str, float]],
           selections: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Promote the LLM-selected passage to rank 1, keep the RRF order below it."""
    runs: Dict[str, Dict[str, float]] = {}
    for qid, info in selections.items():
        ordered = sorted(fused.get(qid, {}).items(), key=lambda x: (-x[1], x[0]))
        ranked, seen = [], set()
        for pid in info["picked_pids"]:
            if pid not in seen:
                ranked.append(pid)
                seen.add(pid)
        for pid, _ in ordered:
            if pid not in seen:
                ranked.append(pid)
                seen.add(pid)
        ranked = ranked[:cfg.top_n]
        runs[qid] = {pid: float(cfg.top_n - i) for i, pid in enumerate(ranked)}
    return runs


def write_run(cfg: Config, runs: Dict[str, Dict[str, float]]) -> None:
    cfg.final_run_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"compcqr_{cfg.file_prefix}_topk{cfg.top_k}"
    with open(cfg.final_run_path, "w") as f:
        for qid, doc_scores in runs.items():
            for rank, (pid, score) in enumerate(
                    sorted(doc_scores.items(), key=lambda x: -x[1]), 1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.1f} {tag}\n")
    LOGGER.info(f"[Step6] final run written: {cfg.final_run_path}")


def evaluate(cfg: Config, runs: Dict[str, Dict[str, float]],
             selections: Dict[str, Dict]) -> Dict[str, float]:
    qrels_binary, qrels_graded = load_qrels(cfg.qrel_file, cfg.id_prefix_strip)

    binary_results = pytrec_eval.RelevanceEvaluator(qrels_binary, BINARY_MEASURES).evaluate(runs)
    graded_results = pytrec_eval.RelevanceEvaluator(qrels_graded, GRADED_MEASURES).evaluate(runs)

    def mean(key: str, results) -> float:
        values = [v[key] for v in results.values()]
        return float(np.mean(values)) if values else 0.0

    # Diagnostics: how often is a gold passage inside the candidate pool shown to
    # the selector, and how often does the selector actually pick it?
    gold_in_pool = gold_selected = evaluated = 0
    for qid, info in selections.items():
        gold = set(qrels_binary.get(qid, {}))
        gold = {pid for pid in gold if qrels_binary[qid][pid] > 0}
        if not gold:
            continue
        evaluated += 1
        if gold & set(info["top_docs"]):
            gold_in_pool += 1
            if gold & set(info["picked_pids"]):
                gold_selected += 1

    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "llm": cfg.llm_tag,
        "retriever": cfg.retriever_tag,
        "top_k": cfg.top_k,
        "MAP": mean("map", binary_results),
        "MRR": mean("recip_rank", binary_results),
        "NDCG@3": mean("ndcg_cut_3", graded_results),
        "Recall@5": mean("recall_5", binary_results),
        "Recall@10": mean("recall_10", binary_results),
        "Recall@20": mean("recall_20", binary_results),
        "Recall@100": mean("recall_100", binary_results),
        "gold_in_pool": gold_in_pool,
        "gold_selected": gold_selected,
        "selection_accuracy": gold_selected / gold_in_pool if gold_in_pool else 0.0,
        "pool_recall": gold_in_pool / evaluated if evaluated else 0.0,
    }

    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    LOGGER.info(f"[Step6] metrics written: {cfg.metrics_path}")
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  CompCQR — {metrics['dataset']}/{metrics['split']} | "
          f"LLM={metrics['llm']} | retriever={metrics['retriever']} | "
          f"top_k={metrics['top_k']}")
    print("=" * width)
    for key, value in metrics.items():
        if key in {"dataset", "split", "llm", "retriever", "top_k"}:
            continue
        print(f"  {key:22s}: {value:.4f}" if isinstance(value, float)
              else f"  {key:22s}: {value}")
    print("=" * width + "\n")
