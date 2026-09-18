<div align="center">

# CompCQR

**Compositional Query Generation for Training-Free Conversational Search**

Yunah Jang · Kang-il Lee · Joongbo Shin · Kyomin Jung

[![EMNLP 2026](https://img.shields.io/badge/Findings%20of%20EMNLP-2026-4b0082.svg)](https://arxiv.org/abs/2609.14646)
[![arXiv](https://img.shields.io/badge/arXiv-2609.14646-b31b1b.svg)](https://arxiv.org/abs/2609.14646)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation, accepted to **Findings of EMNLP 2026**.

*325 query rewrites from 3 LLM calls — no training, no retriever-specific tuning.*

</div>

> Multi-turn interactions with LLMs are becoming increasingly common in information-seeking
> scenarios. However, user queries are often ambiguous and context-dependent, making them
> ill-suited for direct use as retriever queries. Conversational query reformulation (CQR)
> addresses this issue by rewriting the current utterance into a stand-alone query grounded in the
> dialogue history. Recent LLM-based CQR approaches achieve strong performance; however, their
> repeated LLM invocations and misalignment with downstream retrievers remain challenges. In this
> work, we begin from the observation that **retrievers are highly sensitive to content ordering**:
> simply reordering the same content can lead to changes in retrieval coverage and performance.
> Based on this, we propose a novel training-free method that generates a very large number of
> queries with minimal LLM usage by **compositionally combining a small set of atomic components**.
> We further apply LLM reasoning to construct a high-quality document set that balances precision
> and recall while capturing the user's core intent. Our framework generalizes across both open-
> and closed-source LLMs as well as dense and sparse retrievers. It achieves strong performance on
> four widely used conversational benchmarks, with up to **22.5% relative MRR improvement** over
> the previous state-of-the-art baseline with far fewer LLM calls.

---

## Overview

A conversational query (*"who sang it?"*) is useless on its own. Instead of betting everything
on **one** rewrite, CompCQR decomposes the turn into five reusable **components**, composes them
into many query **variations**, and lets retrieval consensus and an LLM selector decide what the
user meant.

| Component | Symbol | Produced by |
|---|---|---|
| Original query | `oq` | the dataset turn |
| Machine query (rewrite) | `mq` | LLM, step 0 |
| Machine answer (hypothetical answer) | `ma` | LLM, step 0 |
| Machine follow-up query | `mq2` | LLM, step 1 |
| Machine follow-up answer | `ma2` | LLM, step 1 |

Every ordered subset of the components — of size 1 up to `max_k` — is one variation. With the
default `max_k=5` (i.e. a variation may use up to all five components) this gives
**325 queries per turn**: 5 of length 1, 20 of length 2, 60 of length 3, 120 of length 4 and 120
of length 5. Order matters, which is the whole point: `ma+oq` and `oq+ma` are two different
queries for the retriever.

```
                 ┌──────────────────────────────────────────────────────────┐
  conversation   │ Step 0  rewrite  mq  +  hypothetical answer  ma    (LLM)  │
  history + turn │ Step 1  follow-up query mq2 + answer ma2          (LLM)  │
                 └──────────────────────────────────────────────────────────┘
                                          │
                  Step 2  compose ordered subsets of {oq, mq, ma, mq2, ma2}
                                          │        → up to 325 variations/turn
                                          ▼
                  Step 3  retrieve with every variation        (ANCE dense / BM25 sparse)
                                          │
                  Step 4  RRF(k=0) fusion over the variation rankings
                                          │        → candidate pool per turn
                                          ▼
                  Step 5  LLM reads the top-k passages and selects one  (LLM)
                                          │
                  Step 6  selected passage → rank 1, rest by RRF → evaluate
```

Only **three LLM calls per turn** are ever made (steps 0, 1 and 5) — the 325 queries come from
composition, not from generation. Each step caches its output, so a crashed run resumes where it
stopped, and steps 0–2 (which depend only on the LLM) are reused across every retriever you
evaluate.

---

## Scope of this repository

This repository contains the **end-to-end CompCQR pipeline only** (steps 0–6 above).

**Data preprocessing and index building follow prior work and are *not* re-implemented here.**
Concretely, we reuse:

| Asset | Source |
|---|---|
| Preprocessed QReCC / TopiOCQA / CAsT conversational files | [ConvGQR](https://github.com/fengranMark/ConvGQR) (Mo et al., ACL 2023) preprocessing scripts |
| Passage collection as HDF5 (`data.h5`) and sharded FAISS dense indexes | [AdaCQR](https://github.com/init0xyz/AdaCQR) (COLING 2025) `preprocess/` scripts |
| BM25 (Pyserini/Lucene) indexes | standard Pyserini indexing, as used by ConvGQR |
| Few-shot demonstrations for the rewrite prompt | [LLM4CS](https://github.com/kyriemao/LLM4CS) (Mao et al., EMNLP 2023 Findings) — see `prompts/README.md` |

Build those artifacts with the upstream repositories, then point `configs/*.yaml` at them.
Nothing else from prior work is required at run time.

---

## Installation

```bash
git clone <this-repo> && cd compcqr
conda create -n compcqr python=3.10 -y && conda activate compcqr
pip install -r requirements.txt
```

Optional extras, install only what you use:

* `vllm` — fast local serving of open-weight LLMs (recommended for Llama-3.1-8B);
* `faiss-gpu` — GPU search over the dense shards (falls back to `faiss-cpu` automatically);
* `pyserini` — BM25 retrieval (requires a JVM).

Fetch the few-shot demonstrations used by the rewrite prompt (kept external, not vendored here —
see `prompts/README.md`):

```bash
wget -O prompts/demonstrations.json \
  https://raw.githubusercontent.com/kyriemao/LLM4CS/main/demonstrations.json
```

API-based back-ends read their key from the environment; nothing is ever hard-coded:

```bash
export DEEPINFRA_API_KEY=...   # --llm_backend deepinfra
export OPENAI_API_KEY=...      # --llm_backend openai
```

Verify the installation without a GPU, an index or an API key:

```bash
python tests/test_pipeline_smoke.py     # stubbed LLM + synthetic run file
```

---

## Expected layout

Point the config at wherever your artifacts live; the default layout is:

```
datasets/
  topiocqa/{dev_new.json, dev_gold.trec, collections/data.h5}
  qrecc/{test.json, qrecc_qrel.tsv, collections/data.h5}
  cast19/{test.jsonl, qrels.tsv}
  cast20/{test.jsonl, qrels.tsv}
  cast_shared/collections/data.h5
indexes/
  topiocqa/ance/{index_test_0.faiss … index_test_9.faiss, doc_ids_0.json … doc_ids_9.json}
  topiocqa/bm25/            # Lucene index directory
prompts/demonstrations.json
```

---

## Quick start

```bash
# TopiOCQA (dev) · Llama-3.1-8B served with vLLM · ANCE dense retrieval
python run_pipeline.py \
  --config configs/topiocqa.yaml \
  --llm_backend vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model_tag llama-3.1-8b \
  --retriever ance --top_k 10
```

More recipes:

```bash
# BM25 (Pyserini) — steps 0-2 are read from cache, only retrieval re-runs
python run_pipeline.py --config configs/qrecc.yaml --retriever bm25 ...

# hosted open-weight LLM
DEEPINFRA_API_KEY=... python run_pipeline.py --config configs/qrecc.yaml \
  --llm_backend deepinfra --model meta-llama/Meta-Llama-3.1-8B-Instruct ...

# closed-source LLM (the paper's GPT-4.1-mini setting)
OPENAI_API_KEY=... python run_pipeline.py --config configs/topiocqa.yaml \
  --llm_backend openai --model gpt-4.1-mini --model_tag gpt-4.1-mini --retriever ance

# re-score an existing retrieval run (steps 4-6 only)
python run_pipeline.py --config configs/topiocqa.yaml --skip_retrieval ...

# small GPU: keep the LLM off the device until FAISS is done
python run_pipeline.py --config configs/topiocqa.yaml --defer_llm ...
```

Ready-made wrappers live in `scripts/`:

```bash
scripts/run_topiocqa.sh          # RETRIEVER=bm25 TOP_K=10 scripts/run_topiocqa.sh
scripts/run_qrecc.sh
scripts/run_cast.sh
```

---

## Command line reference

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | dataset YAML under `configs/` |
| `--llm_backend` | `vllm` | `vllm` \| `hf` \| `deepinfra` \| `openai` |
| `--model` / `--model_tag` | Llama-3.1-8B | model id / short tag used in file names |
| `--retriever` | config default | a key of the config's `retrievers:` block |
| `--top_k` | `10` | passages shown to the selector (step 5) |
| `--max_k` | config (`5`) | max **components per variation**; `5` = every ordered subset = 325 queries/turn (paper setting) |
| `--rrf_k` | config (`0`) | RRF constant |
| `--retrieval_top_k` | `100` | documents kept per variation |
| `--num_subsets` / `--batch_size` | `20` / `128` | query-encoding chunking |
| `--skip_retrieval` | off | reuse the existing run file |
| `--defer_llm` | off | build the LLM only at step 5 |
| `--cpu_faiss` | off | force CPU FAISS |
| `--run_id` | — | suffix for repeated runs |

Anything environment-specific (paths, encoders, index shards, BM25 parameters) belongs in the
YAML config; anything experiment-specific is a flag.

---

## Outputs

```
work/<dataset>/      rewrites_*.jsonl            step 0  (shared across retrievers)
                     followups_*.jsonl           step 1  (shared across retrievers)
                     variations_*.jsonl          step 2
                     retrieval_queries_*.jsonl   step 2  (qid = "<turn id>##r<rule id>")
                     run_*.trec                  step 3  variation-level run
output/<dataset>/    selection_*.jsonl           step 5  picked passage + raw LLM output
                     final_*.trec                step 6  final ranking
                     metrics_*.json              step 6  MAP / MRR / NDCG@3 / Recall@k
```

`metrics_*.json` also reports two diagnostics that separate retrieval errors from selection errors:

* `pool_recall` — how often a gold passage reaches the top-k pool shown to the selector;
* `selection_accuracy` — how often the selector picks a gold passage when one is in the pool.

---

## Adding a dataset or a retriever

* **New retriever** — add an entry to `retrievers:` in the config (encoder id, index dir, shard
  count, query prefix, normalization). No code change for any SentenceTransformer-compatible
  encoder; NV-Embed-style encoders are covered by `encoder_type: nvembed`.
* **New dataset** — if its preprocessed files match one of the existing schemas, just write a
  config. Otherwise add a ~10-line adapter in `compcqr/data.py` that answers: current query,
  dialogue history, sample id.

---

## Repository structure

```
run_pipeline.py            CLI entry point
compcqr/
  config.py                YAML config + derived file names
  data.py                  dataset adapters, qrels, HDF5 collection access
  prompts.py               prompt templates and output parsers
  variations.py            component composition (the 325 rules)
  llm.py                   vLLM / HuggingFace / DeepInfra / OpenAI back-ends
  retrieval/dense.py       sharded FAISS search (GPU or CPU)
  retrieval/sparse.py      BM25 via Pyserini
  fusion.py                RRF over variation rankings
  selection.py             LLM passage selection
  evaluation.py            rerank + pytrec_eval metrics
  pipeline.py              steps 0-6 orchestration
configs/                   one YAML per dataset
scripts/                   ready-made run scripts
tests/                     offline smoke test
```

---

## Results

Numbers are from Tables 2–3 of the paper (ANCE = dense, BM25 = sparse; MRR / NDCG@3 / R@10 / R@100).

**QReCC (test) & TopiOCQA (dev)**

| Retriever | LLM | QReCC MRR | NDCG@3 | R@10 | R@100 | TopiOCQA MRR | NDCG@3 | R@10 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | Llama-3.1-8B | 55.7 | 54.0 | 75.8 | 94.1 | 29.3 | 28.4 | 49.3 | 73.7 |
| BM25 | GPT-4.1-mini | 67.0 | 64.4 | 83.4 | 96.1 | 41.3 | 40.4 | 60.3 | 80.2 |
| ANCE | Llama-3.1-8B | 50.8 | 48.3 | 67.8 | 86.2 | 42.3 | 41.5 | 62.2 | 80.2 |
| ANCE | GPT-4.1-mini | 61.5 | 58.6 | 74.6 | 88.0 | 48.7 | 47.7 | 67.5 | 83.5 |

**TREC CAsT-19 / CAsT-20 (ANCE)**

| LLM | CAsT-19 MRR | R@10 | CAsT-20 MRR | R@10 |
|---|---|---|---|---|
| Llama-3.1-8B | 66.6 | 12.1 | 71.0 | 16.2 |
| GPT-4.1-mini | 75.8 | 13.2 | 78.3 | 16.7 |

Baselines compared in the paper: T5QR, ConvGQR, IterCQR, LLM4CS, AdaCQR, CHIQ-Fusion,
AdaRewriter and ConvSearch-R1. Your reproduction numbers land in `output/<dataset>/metrics_*.json`.

The framework also generalizes to other dense retrievers (GTR, Contriever) and to other LLMs
(Qwen2.5-72B, Llama-3.3-70B); this release ships the main BM25 / ANCE setting, and any
SentenceTransformer-compatible encoder can be added from the config alone.

---

## License

Released under the [MIT License](LICENSE). The datasets, passage collections and pre-trained
models used by this pipeline keep their own licenses.

---

## Acknowledgements

This work builds directly on
[ConvGQR](https://github.com/fengranMark/ConvGQR),
[LLM4CS](https://github.com/kyriemao/LLM4CS) and
[AdaCQR](https://github.com/init0xyz/AdaCQR)
for dataset preprocessing, demonstrations and index construction. We thank the authors for
releasing their code and preprocessed resources.

---

## Contact

For questions or issues, please open an issue in this repository.
