#!/usr/bin/env bash
# QReCC (test) — full pipeline with a hosted model (DeepInfra).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DEEPINFRA_API_KEY:?export DEEPINFRA_API_KEY first}"

python run_pipeline.py \
  --config configs/qrecc.yaml \
  --llm_backend deepinfra \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model_tag llama-3.1-8b \
  --retriever "${RETRIEVER:-ance}" \
  --top_k "${TOP_K:-10}"
