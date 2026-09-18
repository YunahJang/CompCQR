"""Rank fusion over the query variations of one turn.

Each turn produced N variations, each with its own ranked list. We fuse them
with Reciprocal Rank Fusion::

    RRF(d) = sum_v 1 / (k + rank_v(d))

summed only over the variations that actually retrieved ``d`` within the top-N.
``k = 0`` (the default) is the aggressive setting: a document ranked first by
several variations dominates, which is exactly the consensus signal we want from
a large pool of paraphrases.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .config import Config
from .utils import get_logger

LOGGER = get_logger(__name__)

QID_RULE_SEP = "##"


def rrf_score(ranks: List[int], k: int = 0) -> float:
    return sum(1.0 / (k + r) for r in ranks if r > 0)


def fuse_run(cfg: Config) -> Dict[str, Dict[str, float]]:
    """Read the variation-level run file and return ``{qid: {pid: rrf_score}}``."""
    ranks_by_qid: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    with open(cfg.run_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or QID_RULE_SEP not in parts[0]:
                continue
            qid, rule = parts[0].split(QID_RULE_SEP, 1)
            pid, rank = parts[2], int(parts[3])
            if pid.startswith("doc"):
                pid = pid[3:]
            if rank <= cfg.top_n:
                ranks_by_qid[qid][rule][pid] = rank

    fused: Dict[str, Dict[str, float]] = {}
    for qid, per_rule in ranks_by_qid.items():
        rules = sorted(per_rule)
        documents = {pid for rule_ranks in per_rule.values() for pid in rule_ranks}
        fused[qid] = {
            pid: rrf_score([per_rule[rule].get(pid, 0) for rule in rules], cfg.rrf_k)
            for pid in documents
        }

    LOGGER.info(f"[Step4] RRF(k={cfg.rrf_k}) fused {len(fused)} turns")
    return fused


def top_documents(fused: Dict[str, float], top_k: int) -> List[str]:
    """Top-k documents of one turn; ties broken deterministically by pid."""
    return [pid for pid, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
