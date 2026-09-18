"""Offline smoke test: runs steps 0-2 and 4-6 on a 1-turn toy dataset.

No GPU, no index and no API key needed — the LLM is stubbed and the retrieval
run file is synthetic. Useful to check an installation::

    python tests/test_pipeline_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compcqr.config import Config
from compcqr.data import get_adapter
from compcqr.pipeline import (run_pipeline, step0_rewrites, step1_followups,
                              step2_variations)

CONFIG_TEMPLATE = """
dataset: smoke
split: dev
schema: topiocqa
paths:
  input_file: {tmp}/dev.json
  qrel_file: {tmp}/qrel.trec
  collection_h5: {tmp}/data.h5
  demonstrations: {tmp}/demos.json
  work_dir: {tmp}/work
  output_dir: {tmp}/out
variations: {{components: [oq, mq, ma, mq2, ma2], max_k: 2, joiner: "+"}}
fusion: {{rrf_k: 0, top_n: 100}}
selection: {{history_turns: 4}}
default_retriever: fake
retrievers:
  fake: {{type: dense, encoder: none, index_dir: {tmp}}}
"""


class StubLLM:
    """Answers each of the three prompt types in the format the parsers expect."""

    def generate(self, prompts, system_prompt=None, **_):
        responses = []
        for prompt in prompts:
            if "selected_passage_number" in prompt:
                responses.append(['{"selected_passage_number": "2"}'])
            elif "followup_query" in str(system_prompt):
                responses.append(['{"followup_query": "who wrote it?", '
                                  '"next_answer": "someone"}'])
            else:
                responses.append(["Rewrite: Reason. So the question should be rewritten "
                                  "as: who sang song X?\nResponse: Singer S."])
        return {"responses": responses}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        items = [{"id": "1-1", "query": "who sang it?",
                  "history_query": ["what is song X"], "history_answer": ["a 1975 song"]}]
        (tmp / "dev.json").write_text(json.dumps(items))
        (tmp / "qrel.trec").write_text("1-1 Q0 2 1\n")       # doc 2 is the gold passage
        (tmp / "demos.json").write_text("[]")
        with h5py.File(tmp / "data.h5", "w") as hf:
            hf.create_dataset("data",
                              data=[json.dumps({"id": i, "text": f"passage {i}"})
                                    for i in range(1, 6)],
                              dtype=h5py.string_dtype())
        (tmp / "cfg.yaml").write_text(CONFIG_TEMPLATE.format(tmp=tmp))

        cfg = Config.load(tmp / "cfg.yaml").bind("stub", "fake", top_k=3)
        adapter = get_adapter(cfg.schema)
        llm = StubLLM()

        rewrites = step0_rewrites(cfg, items, adapter, llm)
        followups = step1_followups(cfg, items, adapter, llm)
        step2_variations(cfg, items, adapter, rewrites, followups)

        queries = [json.loads(line) for line in
                   cfg.retrieval_input_path.read_text().splitlines()]
        assert len(queries) == 25, f"max_k=2 should give 25 variations, got {len(queries)}"

        # Synthetic retrieval: every variation ranks doc 3 first, gold doc 2 second.
        with open(cfg.run_path, "w") as f:
            for query in queries:
                for rank, pid in enumerate([3, 2, 1], 1):
                    f.write(f"{query['qid']} Q0 {pid} {rank} {1.0 / rank:.3f} fake\n")

        metrics = run_pipeline(cfg, adapter, lambda: llm, skip_retrieval=True)

        assert metrics["MRR"] == 1.0, metrics
        assert metrics["selection_accuracy"] == 1.0, metrics
        top_line = cfg.final_run_path.read_text().splitlines()[0]
        assert top_line.split()[2] == "2", top_line
        print("smoke test passed")


if __name__ == "__main__":
    main()
