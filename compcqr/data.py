"""Dataset adapters and small I/O helpers.

Preprocessed conversational-search files come in a few different shapes
depending on which prior work produced them, so every dataset gets a tiny
adapter that answers three questions:

* what is the current query of this turn?
* what does the dialogue history look like (for the rewrite / follow-up prompt)?
* what is the sample id used in the qrels?

Supported schemas
-----------------
``topiocqa``  {"id", "query", "history_query": [...], "history_answer": [...]}
``qrecc``     {"sample_id", "cur_utt_text", "ctx_utts_text": [q, a, q, a, ...]}
``cast19``    {"id", "input": [q1, ..., q_cur]}                (queries only)
``cast20``    {"id", "input": [...], "automatic_response": [...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import h5py


# ═════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═════════════════════════════════════════════════════════════════════════
def load_json_or_jsonl(path: Path) -> List[Dict]:
    """Read a file that is either a JSON list or one JSON object per line."""
    text = Path(path).read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_passages_by_ids(h5_path: Path, doc_ids: Set[int]) -> Dict[int, str]:
    """Fetch passage texts from the HDF5 collection built during preprocessing.

    The collection stores one JSON record per row and ``h5 index = doc_id - 1``.
    """
    passages: Dict[int, str] = {}
    with h5py.File(h5_path, "r") as hf:
        dataset = hf["data"]
        for doc_id in sorted(doc_ids):
            idx = doc_id - 1
            if not (0 <= idx < len(dataset)):
                continue
            raw = dataset[idx]
            text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            try:
                obj = json.loads(text)
                text = obj.get("text") or obj.get("body") or obj.get("content") or text
            except json.JSONDecodeError:
                pass
            passages[doc_id] = text
    return passages


def load_qrels(path: Path, strip_prefix: str | None = None
               ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Load TREC-style qrels; returns (binary, graded)."""
    binary: Dict[str, Dict[str, int]] = {}
    graded: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if strip_prefix:
                qid = qid.replace(strip_prefix, "")
            graded.setdefault(qid, {})[pid] = rel
            binary.setdefault(qid, {})[pid] = int(rel >= 1)
    return binary, graded


# ═════════════════════════════════════════════════════════════════════════
#  Dataset adapters
# ═════════════════════════════════════════════════════════════════════════
class ConversationAdapter:
    """Base adapter: a turn with an explicit question/answer history."""

    name = "base"

    def __init__(self, strip_prefix: str | None = None):
        self.strip_prefix = strip_prefix

    # ── identity ───────────────────────────────────────────────────────────
    def sample_id(self, item: Dict) -> str:
        sid = str(item.get("id", item.get("sample_id", "")))
        return sid.replace(self.strip_prefix, "") if self.strip_prefix else sid

    # ── turn content ───────────────────────────────────────────────────────
    def query(self, item: Dict) -> str:
        raise NotImplementedError

    def history(self, item: Dict) -> List[Tuple[str, str]]:
        """History as (question, answer) pairs; answer may be an empty string."""
        raise NotImplementedError

    # ── prompt contexts ────────────────────────────────────────────────────
    def rewrite_context(self, item: Dict) -> str:
        """LLM4CS-style context block for the rewrite/response prompt."""
        blocks = []
        for question, answer in self.history(item):
            blocks.append(f"Question: {question}\nResponse: {answer}"
                          if answer else f"Question: {question}")
        if blocks:
            blocks[0] = "Context:\n" + blocks[0]
        blocks.append(f"Current Question: {self.query(item)}")
        return "\n\n".join(blocks)

    def followup_context(self, item: Dict) -> str:
        parts = [f"Question: {q}\nResponse: {a}" if a else f"Question: {q}"
                 for q, a in self.history(item)]
        return " ".join(parts)

    def selection_context(self, item: Dict, max_turns: int = 4) -> str:
        parts = [f"Q: {q} A: {a}" if a else f"Q: {q}" for q, a in self.history(item)]
        return " ".join(parts[-max_turns:]) if max_turns > 0 else " ".join(parts)


class TopiOCQAAdapter(ConversationAdapter):
    name = "topiocqa"

    def query(self, item: Dict) -> str:
        return item["query"]

    def history(self, item: Dict) -> List[Tuple[str, str]]:
        questions = item.get("history_query", [])
        answers = item.get("history_answer", [])
        return [(q, answers[i] if i < len(answers) else "")
                for i, q in enumerate(questions)]


class QReCCAdapter(ConversationAdapter):
    """QReCC stores the history as a flat alternating [q, a, q, a, ...] list."""

    name = "qrecc"

    def query(self, item: Dict) -> str:
        return item["cur_utt_text"]

    def history(self, item: Dict) -> List[Tuple[str, str]]:
        utts = item.get("ctx_utts_text", [])
        return [(utts[i], utts[i + 1] if i + 1 < len(utts) else "")
                for i in range(0, len(utts), 2)]


class CAsT19Adapter(ConversationAdapter):
    """CAsT-19 has no system responses — history is questions only."""

    name = "cast19"

    def query(self, item: Dict) -> str:
        return item["input"][-1]

    def history(self, item: Dict) -> List[Tuple[str, str]]:
        return [(q, "") for q in item["input"][:-1]]


class CAsT20Adapter(ConversationAdapter):
    """CAsT-20 provides automatic canonical responses for previous turns."""

    name = "cast20"

    def query(self, item: Dict) -> str:
        return item["input"][-1]

    def history(self, item: Dict) -> List[Tuple[str, str]]:
        questions = item["input"][:-1]
        responses = item.get("automatic_response", [])
        return [(q, responses[i] if i < len(responses) else "")
                for i, q in enumerate(questions)]


_ADAPTERS = {
    "topiocqa": TopiOCQAAdapter,
    "qrecc": QReCCAdapter,
    "cast19": CAsT19Adapter,
    "cast20": CAsT20Adapter,
}


def get_adapter(schema: str, strip_prefix: str | None = None) -> ConversationAdapter:
    if schema not in _ADAPTERS:
        raise ValueError(f"Unknown dataset schema '{schema}'. "
                         f"Available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[schema](strip_prefix=strip_prefix)
