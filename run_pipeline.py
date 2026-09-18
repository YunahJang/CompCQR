#!/usr/bin/env python3
"""CompCQR end-to-end entry point.

Examples
--------
# TopiOCQA, Llama-3.1-8B served locally with vLLM, ANCE dense retrieval
python run_pipeline.py --config configs/topiocqa.yaml \
    --llm_backend vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_tag llama-3.1-8b --retriever ance --top_k 10

# QReCC with BM25 and a hosted model
DEEPINFRA_API_KEY=... python run_pipeline.py --config configs/qrecc.yaml \
    --llm_backend deepinfra --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_tag llama-3.1-8b --retriever bm25 --top_k 10

# Cheap variant of the composition step (5 single-component queries instead of 325)
python run_pipeline.py --config configs/topiocqa.yaml --max_k 1 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from compcqr.config import Config
from compcqr.data import get_adapter
from compcqr.llm import build_llm
from compcqr.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CompCQR end-to-end pipeline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="dataset YAML under configs/")

    # ── LLM ────────────────────────────────────────────────────────────────
    p.add_argument("--llm_backend", default="vllm",
                   choices=["vllm", "hf", "deepinfra", "openai"])
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help="model id passed to the backend")
    p.add_argument("--model_tag", default=None,
                   help="short tag used in output file names (default: derived from --model)")
    p.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM only")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM only")
    p.add_argument("--llm_batch_size", type=int, default=8, help="HuggingFace backend only")
    p.add_argument("--defer_llm", action="store_true",
                   help="build the LLM only at step 5 (keeps the GPU free for FAISS)")

    # ── retrieval ──────────────────────────────────────────────────────────
    p.add_argument("--retriever", default=None,
                   help="a key of the config's retrievers block (default: its default_retriever)")
    p.add_argument("--retrieval_top_k", type=int, default=100,
                   help="documents kept per query variation")
    p.add_argument("--num_subsets", type=int, default=20,
                   help="query subsets for dense encoding (memory/restartability)")
    p.add_argument("--batch_size", type=int, default=128, help="query encoding batch size")
    p.add_argument("--cpu_faiss", action="store_true", help="force CPU FAISS search")
    p.add_argument("--skip_retrieval", action="store_true",
                   help="reuse the existing run file for steps 4-6")

    # ── method knobs ───────────────────────────────────────────────────────
    p.add_argument("--top_k", type=int, default=10,
                   help="passages shown to the selector in step 5")
    p.add_argument("--max_k", type=int, default=None,
                   help="override variations.max_k from the config (5 -> 325 variations)")
    p.add_argument("--rrf_k", type=int, default=None, help="override fusion.rrf_k")
    p.add_argument("--run_id", type=int, default=None,
                   help="suffix for repeated runs of the same configuration")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.load(args.config)
    if args.max_k is not None:
        cfg.max_k = args.max_k
    if args.rrf_k is not None:
        cfg.rrf_k = args.rrf_k

    retriever_tag = args.retriever or cfg.default_retriever
    if retriever_tag not in cfg.retrievers:
        raise SystemExit(f"Unknown retriever '{retriever_tag}'. "
                         f"Available in {args.config}: {sorted(cfg.retrievers)}")

    model_tag = args.model_tag or args.model.split("/")[-1].lower()
    cfg.bind(llm_tag=model_tag, retriever_tag=retriever_tag,
             top_k=args.top_k, run_id=args.run_id)

    adapter = get_adapter(cfg.schema, cfg.id_prefix_strip)

    def llm_factory():
        return build_llm(args.llm_backend, args.model,
                         tensor_parallel_size=args.tensor_parallel_size,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         batch_size=args.llm_batch_size)

    print(f"\nCompCQR | dataset={cfg.dataset}/{cfg.split} | llm={model_tag} "
          f"({args.llm_backend}) | retriever={retriever_tag} | "
          f"max_k={cfg.max_k} | rrf_k={cfg.rrf_k} | top_k={cfg.top_k}\n")

    run_pipeline(cfg, adapter, llm_factory,
                 retrieval_top_k=args.retrieval_top_k,
                 num_subsets=args.num_subsets,
                 batch_size=args.batch_size,
                 skip_retrieval=args.skip_retrieval,
                 defer_llm=args.defer_llm,
                 use_gpu_faiss=not args.cpu_faiss)


if __name__ == "__main__":
    main()
