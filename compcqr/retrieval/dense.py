"""Dense retrieval over a sharded FAISS index.

The index is built once during preprocessing (see README, "Index building") and
is stored as ``index_splits`` shards so that a full collection (e.g. 25M
TopiOCQA passages) never has to fit in memory at once.

Two loops matter for speed:

* queries are encoded subset by subset and cached on disk, so a crashed run
  resumes without re-encoding;
* **shards are the outer loop** — each shard is read from disk exactly once and
  searched against every query subset. Since the per-shard results are merged by
  a commutative max, the ordering does not change the output.
"""
from __future__ import annotations

import gc
import json
import math
import pickle
from heapq import nlargest, nsmallest
from typing import Dict, List, Sequence

import numpy as np

from ..config import Config, RetrieverConfig
from ..utils import get_logger

LOGGER = get_logger(__name__)


def _build_encoder(retriever: RetrieverConfig, device: str):
    if retriever.encoder_type == "nvembed":
        from transformers import AutoModel
        import torch
        model = AutoModel.from_pretrained(retriever.encoder, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).to(device).eval()
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(retriever.encoder, device=device, trust_remote_code=True)
        if retriever.max_seq_length > 0:
            model.max_seq_length = retriever.max_seq_length
    LOGGER.info(f"[Step3] encoder loaded: {retriever.encoder} on {device}")
    return model


def _encode_fn(retriever: RetrieverConfig, get_model, batch_size: int):
    """Return a ``texts -> np.ndarray`` closure for the configured encoder."""
    if retriever.encoder_type == "nvembed":
        import torch
        import torch.nn.functional as F

        max_len = retriever.max_seq_length or 512

        def encode(texts: Sequence[str]) -> np.ndarray:
            model = get_model()
            chunks = []
            for i in range(0, len(texts), batch_size):
                with torch.no_grad():
                    emb = model.encode(list(texts[i:i + batch_size]),
                                       instruction=retriever.query_instruction,
                                       max_length=max_len)
                emb = F.normalize(emb, p=2, dim=1)
                chunks.append(emb.detach().to(torch.float32).cpu().numpy())
            return np.concatenate(chunks, axis=0)
    else:
        def encode(texts: Sequence[str]) -> np.ndarray:
            model = get_model()
            return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                                convert_to_numpy=True,
                                normalize_embeddings=retriever.normalize)
    return encode


def dense_search(cfg: Config, top_k: int = 100, num_subsets: int = 20,
                 batch_size: int = 128, use_gpu_faiss: bool = True) -> None:
    """Search every query variation and write a TREC run file."""
    if cfg.run_path.exists():
        LOGGER.info(f"[Step3] run file already exists: {cfg.run_path}")
        return

    import faiss
    import torch

    retriever = cfg.retriever
    records = [json.loads(line) for line in cfg.retrieval_input_path.read_text().splitlines()
               if line.strip()]
    qids = [r["qid"] for r in records]
    queries = [retriever.query_prefix + r["query"] for r in records]
    LOGGER.info(f"[Step3] {len(queries):,} query variations to search "
                f"({retriever.name}, {retriever.index_splits} shards)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Lazy: if every query-vector subset is already cached we never load the
    # encoder at all, leaving the whole GPU to FAISS.
    holder: Dict[str, object] = {}

    def get_model():
        if "model" not in holder:
            holder["model"] = _build_encoder(retriever, device)
        return holder["model"]

    encode = _encode_fn(retriever, get_model, batch_size)

    if cfg.merge_cache_path.exists():
        LOGGER.info(f"[Step3] loading cached merged results: {cfg.merge_cache_path}")
        with open(cfg.merge_cache_path, "rb") as f:
            merged = pickle.load(f)
    else:
        merged: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}
        subset_size = math.ceil(len(queries) / max(num_subsets, 1))

        # ── Phase A: encode (or load) query vectors ────────────────────────
        subsets: List[tuple] = []
        for subset_idx in range(num_subsets):
            start = subset_idx * subset_size
            end = min(start + subset_size, len(queries))
            if start >= end:
                break
            vec_path = cfg.query_vector_path(subset_idx)
            if vec_path.exists():
                vectors = np.load(vec_path)
            else:
                LOGGER.info(f"[Step3] encoding subset {subset_idx + 1}/{num_subsets} "
                            f"({end - start} queries)")
                vectors = encode(queries[start:end])
                np.save(vec_path, vectors)
            subsets.append((qids[start:end], vectors))

        holder.clear()  # free the encoder before FAISS takes the GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase B: one pass per shard ────────────────────────────────────
        gpu_faiss = (use_gpu_faiss and hasattr(faiss, "StandardGpuResources")
                     and faiss.get_num_gpus() > 0)
        LOGGER.info(f"[Step3] FAISS backend: {'GPU (fp16)' if gpu_faiss else 'CPU'}")
        resources = cloner = None
        if gpu_faiss:
            resources = faiss.StandardGpuResources()
            resources.setTempMemory(512 * 1024 * 1024)  # leave room for the shard itself
            cloner = faiss.GpuClonerOptions()
            cloner.useFloat16 = True

        for shard in range(retriever.index_splits):
            doc_ids = json.loads(
                (retriever.index_dir /
                 retriever.docids_file_pattern.format(shard=shard)).read_text())
            index = faiss.read_index(
                str(retriever.index_dir /
                    retriever.index_file_pattern.format(shard=shard)))
            if gpu_faiss:
                index = faiss.index_cpu_to_gpu(resources, 0, index, cloner)
            LOGGER.info(f"[Step3] shard {shard + 1}/{retriever.index_splits} loaded")

            for subset_qids, vectors in subsets:
                query_matrix = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
                scores, indices = index.search(query_matrix, top_k)
                for row_scores, row_indices, qid in zip(scores, indices, subset_qids):
                    store = merged[qid]
                    for score, idx in zip(row_scores, row_indices):
                        if idx < 0:
                            continue
                        pid = doc_ids[idx]
                        if score > store.get(pid, -1e30):
                            store[pid] = float(score)
                    # keep the store bounded: drop everything below the top-k
                    if len(store) > 2 * top_k:
                        for pid, _ in nsmallest(len(store) - top_k, store.items(),
                                                key=lambda x: x[1]):
                            del store[pid]

            del index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(cfg.merge_cache_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(f"[Step3] merged results cached: {cfg.merge_cache_path}")

    cfg.run_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.run_path, "w") as f:
        for qid, doc_scores in merged.items():
            for rank, (pid, score) in enumerate(
                    nlargest(top_k, doc_scores.items(), key=lambda x: x[1]), 1):
                f.write(f"{qid} Q0 {str(pid).replace('doc', '')} {rank} "
                        f"{score:.6f} {cfg.retriever_tag}\n")
    LOGGER.info(f"[Step3] run file written: {cfg.run_path}")

    del merged
    gc.collect()
