# CompCQR

Official implementation of **"CompCQR: Compositional Query Generation for Training-Free Conversational Search"**, accepted to **Findings of EMNLP 2026**.

## Overview

**CompCQR** is a training-free conversational query reformulation (CQR) framework that efficiently explores a large query space with minimal LLM usage.

Instead of repeatedly invoking an LLM to generate multiple query rewrites, CompCQR extracts a small set of query-related components and compositionally combines them to construct diverse retrieval queries. The retrieved documents are then aggregated using Reciprocal Rank Fusion (RRF), followed by LLM-based best-document selection.

CompCQR generates up to **325 compositional queries with only three LLM calls** in the full framework and generalizes across both sparse and dense retrievers.

## Method

CompCQR consists of four main stages:

1. **Component Generation**  
   Given the dialogue history and the current query, an LLM generates:
   - rewritten query
   - rewritten answer
   - follow-up query
   - follow-up answer

   Together with the original query, these form five query components.

2. **Compositional Query Reformulation**  
   CompCQR enumerates ordered combinations of the five components, producing up to **325 query candidates** without additional LLM calls.

3. **Document Ranking**  
   Each compositional query retrieves candidate documents. The retrieved results are aggregated at the document level using **Reciprocal Rank Fusion (RRF)** and document frequency.

4. **Best Document Selection**  
   An LLM examines the top-ranked candidate documents and selects the document that best supports the user's information need. The selected document is promoted to the top of the final ranking.

## Key Results

We evaluate CompCQR on four conversational search benchmarks:

- **QReCC**
- **TopiOCQA**
- **TREC CAsT-19**
- **TREC CAsT-20**

with both sparse (**BM25**) and dense (**ANCE**) retrieval.

CompCQR achieves strong performance without task-specific training and consistently outperforms prior training-free LLM-based query reformulation approaches across the evaluated settings.

The framework also generalizes to other dense retrievers, including **GTR** and **Contriever**, and to open-source LLMs such as **Llama** and **Qwen**.

## Repository Structure

> The code will be released soon.

```text
CompCQR/
├── ...
└── README.md
```

## Installation

Installation instructions will be provided with the code release.

## Usage

Usage instructions and scripts for reproducing the experiments will be provided with the code release.

## Datasets

The experiments use the following conversational search benchmarks:

- QReCC
- TopiOCQA
- TREC CAsT-19
- TREC CAsT-20

Please refer to the original dataset releases for access and licensing information.

## Models and Retrievers

The main experiments use:

**LLMs**
- GPT-4.1-mini
- Llama-3.1-8B-Instruct

**Retrievers**
- BM25
- ANCE

Additional experiments include GTR, Contriever, Qwen2.5-72B, and Llama-3.3-70B.


## Contact

For questions or issues, please open an issue in this repository.
