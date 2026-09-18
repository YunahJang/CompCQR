"""LLM passage selection: pick the single best passage out of the fused top-k."""
from __future__ import annotations

import json
from typing import Dict, List, Set

from .config import Config
from .data import ConversationAdapter, load_passages_by_ids, write_jsonl
from .fusion import top_documents
from .prompts import SYSTEM_SELECTOR, build_selection_prompt, parse_selected_passage
from .utils import get_logger

LOGGER = get_logger(__name__)


def _load_cached(path) -> Dict[str, Dict]:
    selections: Dict[str, Dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            top_docs = [str(d) for d in item["top_docs"]]
            picked = [str(p) for p in (item.get("picked_pids") or [])]
            if not picked:
                numbers = item.get("picked_nums") or [1]
                picked = [top_docs[n - 1] for n in numbers if 1 <= n <= len(top_docs)]
            selections[item["id"]] = {
                "picked_pids": picked or top_docs[:1],
                "top_docs": top_docs,
            }
    return selections


def select_passages(cfg: Config, fused: Dict[str, Dict[str, float]],
                    id2item: Dict[str, Dict], adapter: ConversationAdapter,
                    llm) -> Dict[str, Dict]:
    if cfg.selection_path.exists():
        LOGGER.info(f"[Step5] selection cache found: {cfg.selection_path}")
        return _load_cached(cfg.selection_path)

    top_k = cfg.top_k

    # Candidate pool per turn (padded when the pool is smaller than top_k, so the
    # prompt always lists exactly top_k passages).
    candidates: Dict[str, List[str]] = {}
    needed_docs: Set[int] = set()
    for qid, scores in fused.items():
        if qid not in id2item:
            continue
        docs = top_documents(scores, top_k)
        if not docs:
            continue
        while len(docs) < top_k:
            docs = docs + docs[:top_k - len(docs)]
        candidates[qid] = docs[:top_k]
        needed_docs.update(int(d) for d in candidates[qid])

    LOGGER.info(f"[Step5] loading {len(needed_docs):,} passages from the collection")
    passages = load_passages_by_ids(cfg.collection_h5, needed_docs)

    prompts, order = [], []
    for qid, docs in candidates.items():
        item = id2item[qid]
        texts = [passages.get(int(d), "[INVALID DOC ID]") for d in docs]
        prompts.append(build_selection_prompt(
            query=adapter.query(item),
            context=adapter.selection_context(item, cfg.selection_history_turns),
            passages=texts,
            top_k=top_k))
        order.append(qid)

    LOGGER.info(f"[Step5] asking the LLM to select 1 of {top_k} passages "
                f"for {len(prompts)} turns")
    outputs = llm.generate(prompts, system_prompt=SYSTEM_SELECTOR, temperature=0.7)["responses"]

    cfg.selection_raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.selection_raw_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(str(output) + "\n\n")

    records, selections, unparsed = [], {}, 0
    for i, raw in enumerate(outputs):
        qid = order[i]
        docs = candidates[qid]
        reply = (raw[0] if isinstance(raw, list) and raw else str(raw)).strip()

        number = parse_selected_passage(reply, 1, top_k)
        if number is None:  # fall back to the RRF top-1
            number = 1
            unparsed += 1
        picked = docs[number - 1]

        records.append({"id": qid, "picked_nums": [number], "picked_pids": [str(picked)],
                        "top_docs": [str(d) for d in docs], "llm_output_raw": reply})
        selections[qid] = {"picked_pids": [str(picked)], "top_docs": [str(d) for d in docs]}

    write_jsonl(cfg.selection_path, records)
    LOGGER.info(f"[Step5] {len(records)} selections written ({unparsed} unparsable "
                f"-> fell back to RRF rank 1): {cfg.selection_path}")
    return selections
