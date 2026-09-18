"""End-to-end CompCQR pipeline.

    Step 0  rewrite + hypothetical answer for the current turn      (LLM)
    Step 1  follow-up query + its answer                            (LLM)
    Step 2  compose the component texts into query variations
    Step 3  retrieve with every variation                           (dense | BM25)
    Step 4  fuse the variation rankings with RRF
    Step 5  select the single best passage out of the fused top-k   (LLM)
    Step 6  promote the selected passage to rank 1 and evaluate

Every step writes its output to disk and skips itself when that file already
exists, so a run can be resumed, and steps 0-2 (which depend only on the LLM)
are shared across retrievers.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .config import Config
from .data import (ConversationAdapter, load_json_or_jsonl, write_jsonl)
from .evaluation import evaluate, print_metrics, rerank, write_run
from .fusion import QID_RULE_SEP, fuse_run
from .prompts import (FOLLOWUP_SYSTEM_PROMPT, SYSTEM_ASSISTANT, build_followup_prompt,
                      build_rewrite_prompt, load_demonstrations, parse_rewrite_outputs,
                      repair_json)
from .retrieval import bm25_search, dense_search
from .selection import select_passages
from .utils import get_logger
from .variations import build_variations, num_variations

LOGGER = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════
#  Step 0 — rewrite + hypothetical answer
# ═════════════════════════════════════════════════════════════════════════
def step0_rewrites(cfg: Config, items: List[Dict], adapter: ConversationAdapter,
                   llm) -> List[Dict]:
    if cfg.rewrite_path.exists():
        LOGGER.info(f"[Step0] rewrite cache found: {cfg.rewrite_path}")
        return load_json_or_jsonl(cfg.rewrite_path)

    demo = load_demonstrations(cfg.demonstrations)
    prompts = [build_rewrite_prompt(adapter.rewrite_context(item), demo) for item in items]
    LOGGER.info(f"[Step0] generating rewrite + response for {len(prompts)} turns")
    outputs = llm.generate(prompts, system_prompt=SYSTEM_ASSISTANT,
                           max_new_tokens=256, temperature=0.7)["responses"]

    records, fallbacks = [], 0
    for item, output in zip(items, outputs):
        rewrites, responses, cots = parse_rewrite_outputs(output)
        record = {"sample_id": adapter.sample_id(item), "predicted_rewrite": rewrites,
                  "predicted_response": responses, "predicted_cot": cots, "output": output}
        if None in rewrites or None in responses:
            # Unparsable generation: keep the raw text as the rewrite so the turn
            # still contributes a usable component.
            fallbacks += 1
            record["predicted_rewrite"] = [output[0].strip() if output else ""]
            record["predicted_response"] = [""]
            record["predicted_cot"] = [""]
        records.append(record)

    write_jsonl(cfg.rewrite_path, records)
    LOGGER.info(f"[Step0] {len(records)} rewrites written ({fallbacks} fallbacks): "
                f"{cfg.rewrite_path}")
    return records


# ═════════════════════════════════════════════════════════════════════════
#  Step 1 — follow-up query + answer
# ═════════════════════════════════════════════════════════════════════════
def step1_followups(cfg: Config, items: List[Dict], adapter: ConversationAdapter,
                    llm) -> List[Dict]:
    if cfg.followup_path.exists():
        LOGGER.info(f"[Step1] follow-up cache found: {cfg.followup_path}")
        return load_json_or_jsonl(cfg.followup_path)

    prompts = [build_followup_prompt(adapter.followup_context(item), adapter.query(item))
               for item in items]
    LOGGER.info(f"[Step1] generating follow-up query + answer for {len(prompts)} turns")
    outputs = llm.generate(prompts, system_prompt=FOLLOWUP_SYSTEM_PROMPT,
                           temperature=0.7)["responses"]

    records = [{"sample_id": adapter.sample_id(item),
                "generated_next_query": output[0] if output else ""}
               for item, output in zip(items, outputs)]
    write_jsonl(cfg.followup_path, records)
    LOGGER.info(f"[Step1] follow-ups written: {cfg.followup_path}")
    return records


# ═════════════════════════════════════════════════════════════════════════
#  Step 2 — compositional variations
# ═════════════════════════════════════════════════════════════════════════
def step2_variations(cfg: Config, items: List[Dict], adapter: ConversationAdapter,
                     rewrites: List[Dict], followups: List[Dict]) -> None:
    if cfg.variation_path.exists() and cfg.retrieval_input_path.exists():
        LOGGER.info(f"[Step2] variation cache found: {cfg.variation_path}")
        return

    id2rewrite = {str(r["sample_id"]): r for r in rewrites}
    id2followup = {str(f["sample_id"]): f for f in followups}

    composed, malformed = [], 0
    for item in items:
        sid = adapter.sample_id(item)
        rewrite = id2rewrite.get(sid)
        if rewrite is None:
            continue
        raw_followup = id2followup.get(sid, {}).get("generated_next_query", "")
        try:
            followup = json.loads(repair_json(raw_followup)) or {}
        except json.JSONDecodeError:
            followup = {}
            malformed += 1

        parts = {
            "oq": adapter.query(item),
            "mq": next((v for v in rewrite["predicted_rewrite"] if v), ""),
            "ma": next((v for v in rewrite["predicted_response"] if v), ""),
            "mq2": followup.get("followup_query", ""),
            "ma2": followup.get("next_answer", ""),
        }
        variations = build_variations(parts, cfg.components, cfg.max_k, cfg.joiner)
        composed.append({"id": sid, **parts, "variations": variations})

    n_var = num_variations(cfg.components, cfg.max_k)
    LOGGER.info(f"[Step2] {len(composed)} turns x {n_var} variations "
                f"({malformed} malformed follow-up JSON)")

    write_jsonl(cfg.variation_path, composed)
    with open(cfg.retrieval_input_path, "w", encoding="utf-8") as f:
        for turn in composed:
            for variation in turn["variations"]:
                qid = f"{turn['id']}{QID_RULE_SEP}r{variation['rule_id']}"
                f.write(json.dumps({"qid": qid, "query": variation["text"]},
                                   ensure_ascii=False) + "\n")
    LOGGER.info(f"[Step2] retrieval queries written: {cfg.retrieval_input_path}")


# ═════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═════════════════════════════════════════════════════════════════════════
def run_pipeline(cfg: Config, adapter: ConversationAdapter, llm_factory,
                 retrieval_top_k: int = 100, num_subsets: int = 20,
                 batch_size: int = 128, skip_retrieval: bool = False,
                 defer_llm: bool = False, use_gpu_faiss: bool = True) -> Dict[str, float]:
    """Run steps 0-6. ``llm_factory`` is a zero-argument callable building the LLM.

    ``defer_llm=True`` postpones LLM construction until step 5, so a local model
    never shares the GPU with the FAISS shards (needed on smaller cards when
    steps 0-2 are already cached).
    """
    items = load_json_or_jsonl(cfg.input_file)
    id2item = {adapter.sample_id(item): item for item in items}
    LOGGER.info(f"Loaded {len(items)} turns from {cfg.input_file}")

    llm = None if defer_llm else llm_factory()

    need_generation = not (cfg.rewrite_path.exists() and cfg.followup_path.exists())
    if need_generation and llm is None:
        llm = llm_factory()

    rewrites = step0_rewrites(cfg, items, adapter, llm)
    followups = step1_followups(cfg, items, adapter, llm)
    step2_variations(cfg, items, adapter, rewrites, followups)

    if skip_retrieval:
        if not cfg.run_path.exists():
            raise FileNotFoundError(
                f"--skip_retrieval was given but no run file exists at {cfg.run_path}")
        LOGGER.info("[Step3] skipped (reusing the existing run file)")
    elif cfg.retriever.type == "bm25":
        bm25_search(cfg, top_k=retrieval_top_k)
    else:
        dense_search(cfg, top_k=retrieval_top_k, num_subsets=num_subsets,
                     batch_size=batch_size, use_gpu_faiss=use_gpu_faiss)

    fused = fuse_run(cfg)

    if llm is None:
        llm = llm_factory()
    selections = select_passages(cfg, fused, id2item, adapter, llm)

    runs = rerank(cfg, fused, selections)
    write_run(cfg, runs)
    metrics = evaluate(cfg, runs, selections)
    print_metrics(metrics)
    return metrics
