"""Sparse (BM25) retrieval through Pyserini's Lucene searcher."""
from __future__ import annotations

import json

from tqdm import tqdm

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)

_BATCH = 5_000


def bm25_search(cfg: Config, top_k: int = 100, threads: int = 20) -> None:
    if cfg.run_path.exists():
        LOGGER.info(f"[Step3] run file already exists: {cfg.run_path}")
        return

    from pyserini.search.lucene import LuceneSearcher

    retriever = cfg.retriever
    records = [json.loads(line) for line in cfg.retrieval_input_path.read_text().splitlines()
               if line.strip()]
    qids = [r["qid"] for r in records]
    queries = [r["query"] for r in records]
    LOGGER.info(f"[Step3] BM25 over {len(queries):,} query variations "
                f"(k1={retriever.k1}, b={retriever.b})")

    searcher = LuceneSearcher(str(retriever.index_dir))
    searcher.set_bm25(retriever.k1, retriever.b)

    cfg.run_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.run_path, "w") as out:
        for start in tqdm(range(0, len(queries), _BATCH), desc="BM25"):
            end = min(start + _BATCH, len(queries))
            hits = searcher.batch_search(queries[start:end], qids[start:end],
                                         k=top_k, threads=threads)
            for qid in qids[start:end]:
                for rank, hit in enumerate(hits.get(qid, []), 1):
                    docid = hit.docid[3:] if hit.docid.startswith("doc") else hit.docid
                    out.write(f"{qid} Q0 {docid} {rank} {hit.score:.6f} bm25\n")
    LOGGER.info(f"[Step3] run file written: {cfg.run_path}")
