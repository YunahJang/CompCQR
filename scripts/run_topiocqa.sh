#!/usr/bin/env bash
# TopiOCQA (dev) — full pipeline with a locally served Llama-3.1-8B.
set -euo pipefail
cd "$(dirname "$0")/.."

python run_pipeline.py \
  --config configs/topiocqa.yaml \
  --llm_backend vllm \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model_tag llama-3.1-8b \
  --retriever "${RETRIEVER:-ance}" \
  --top_k "${TOP_K:-10}" \
  --retrieval_top_k 100 \
  --num_subsets 20 \
  --batch_size 128
