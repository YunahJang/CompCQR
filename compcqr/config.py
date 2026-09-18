"""Configuration loading.

Everything that is environment-specific (dataset files, passage collection,
pre-built indexes) lives in a YAML file under ``configs/``; everything that is
experiment-specific (LLM, retriever, top-k) is a command line flag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def _as_path(value: str, root: Path = REPO_ROOT) -> Path:
    """Resolve a config path: absolute stays, relative is anchored at the repo root."""
    p = Path(_expand(value))
    return p if p.is_absolute() else (root / p)


@dataclass
class RetrieverConfig:
    """One retriever entry of the ``retrievers:`` block."""

    name: str
    type: str  # "dense" | "bm25"
    # dense
    encoder: str = ""
    encoder_type: str = "sentence-transformers"  # or "nvembed"
    dim: int = 768
    index_dir: Optional[Path] = None
    index_splits: int = 1
    index_file_pattern: str = "index_test_{shard}.faiss"
    docids_file_pattern: str = "doc_ids_{shard}.json"
    query_prefix: str = ""
    normalize: bool = False
    max_seq_length: int = 0
    query_instruction: str = ""
    # bm25
    k1: float = 0.9
    b: float = 0.4

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "RetrieverConfig":
        raw = dict(raw)
        index_dir = raw.pop("index_dir", None)
        return cls(
            name=name,
            index_dir=_as_path(index_dir) if index_dir else None,
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
        )


@dataclass
class Config:
    dataset: str
    split: str
    schema: str
    id_prefix_strip: Optional[str]

    input_file: Path
    qrel_file: Path
    collection_h5: Path
    demonstrations: Path
    work_dir: Path
    output_dir: Path

    components: List[str]
    max_k: int
    joiner: str

    rrf_k: int
    top_n: int
    selection_history_turns: int

    retrievers: Dict[str, RetrieverConfig] = field(default_factory=dict)
    default_retriever: str = ""

    # ── filled in at runtime by ``bind`` ───────────────────────────────────
    llm_tag: str = ""
    retriever_tag: str = ""
    run_id: Optional[int] = None
    top_k: int = 10

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        paths = raw["paths"]
        var = raw.get("variations", {})
        fusion = raw.get("fusion", {})
        sel = raw.get("selection", {})
        retrievers = {
            name: RetrieverConfig.from_dict(name, cfg)
            for name, cfg in raw.get("retrievers", {}).items()
        }
        return cls(
            dataset=raw["dataset"],
            split=raw["split"],
            schema=raw.get("schema", raw["dataset"]),
            id_prefix_strip=raw.get("id_prefix_strip") or None,
            input_file=_as_path(paths["input_file"]),
            qrel_file=_as_path(paths["qrel_file"]),
            collection_h5=_as_path(paths["collection_h5"]),
            demonstrations=_as_path(paths["demonstrations"]),
            work_dir=_as_path(paths.get("work_dir", f"work/{raw['dataset']}")),
            output_dir=_as_path(paths.get("output_dir", f"output/{raw['dataset']}")),
            components=var.get("components", ["oq", "mq", "ma", "mq2", "ma2"]),
            max_k=int(var.get("max_k", 5)),
            joiner=var.get("joiner", "+"),
            rrf_k=int(fusion.get("rrf_k", 0)),
            top_n=int(fusion.get("top_n", 100)),
            selection_history_turns=int(sel.get("history_turns", 4)),
            retrievers=retrievers,
            default_retriever=raw.get("default_retriever", next(iter(retrievers), "")),
        )

    # ── runtime binding ────────────────────────────────────────────────────
    def bind(self, llm_tag: str, retriever_tag: str, top_k: int,
             run_id: Optional[int] = None) -> "Config":
        self.llm_tag = llm_tag
        self.retriever_tag = retriever_tag
        self.top_k = top_k
        self.run_id = run_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def retriever(self) -> RetrieverConfig:
        return self.retrievers[self.retriever_tag]

    # ── derived file names ─────────────────────────────────────────────────
    # Steps 0-2 only depend on the LLM, so their caches are shared by every
    # retriever; steps 3+ additionally depend on the retriever and on how many
    # components are composed.
    @property
    def llm_prefix(self) -> str:
        return f"{self.dataset}_{self.split}_{self.llm_tag}"

    @property
    def var_prefix(self) -> str:
        return f"{self.llm_prefix}_k{self.max_k}"

    @property
    def file_prefix(self) -> str:
        return f"{self.var_prefix}_{self.retriever_tag}"

    @property
    def run_suffix(self) -> str:
        return f"_run{self.run_id}" if self.run_id is not None else ""

    @property
    def rewrite_path(self) -> Path:
        return self.work_dir / f"rewrites_{self.llm_prefix}.jsonl"

    @property
    def followup_path(self) -> Path:
        return self.work_dir / f"followups_{self.llm_prefix}.jsonl"

    @property
    def variation_path(self) -> Path:
        return self.work_dir / f"variations_{self.var_prefix}.jsonl"

    @property
    def retrieval_input_path(self) -> Path:
        return self.work_dir / f"retrieval_queries_{self.var_prefix}.jsonl"

    @property
    def run_path(self) -> Path:
        return self.work_dir / f"run_{self.file_prefix}.trec"

    @property
    def merge_cache_path(self) -> Path:
        return self.work_dir / f"merged_{self.file_prefix}.pkl"

    def query_vector_path(self, subset_idx: int) -> Path:
        return self.work_dir / f"qvecs_{self.file_prefix}_subset{subset_idx}.npy"

    @property
    def selection_path(self) -> Path:
        return (self.output_dir /
                f"selection_{self.file_prefix}_topk{self.top_k}{self.run_suffix}.jsonl")

    @property
    def selection_raw_path(self) -> Path:
        return (self.output_dir /
                f"selection_raw_{self.file_prefix}_topk{self.top_k}{self.run_suffix}.txt")

    @property
    def final_run_path(self) -> Path:
        return (self.output_dir /
                f"final_{self.file_prefix}_topk{self.top_k}{self.run_suffix}.trec")

    @property
    def metrics_path(self) -> Path:
        return (self.output_dir /
                f"metrics_{self.file_prefix}_topk{self.top_k}{self.run_suffix}.json")
