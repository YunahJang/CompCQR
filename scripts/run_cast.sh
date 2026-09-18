#!/usr/bin/env bash
# TREC CAsT-19 / CAsT-20 — both share one passage collection and index.
set -euo pipefail
cd "$(dirname "$0")/.."

for DATASET in cast19 cast20; do
  python run_pipeline.py \
    --config "configs/${DATASET}.yaml" \
    --llm_backend vllm \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_tag llama-3.1-8b \
    --retriever ance \
    --top_k "${TOP_K:-10}"
done
