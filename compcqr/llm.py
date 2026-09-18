"""LLM back-ends.

All back-ends expose the same call signature so the pipeline never needs to know
where the model runs::

    llm.generate(prompts, system_prompt=..., max_new_tokens=..., temperature=...)
        -> {"responses": [[str, ...], ...]}      # one list per prompt

Supported back-ends
-------------------
``vllm``       local batched inference (fastest for open-weight models)
``hf``         local ``transformers`` generation
``deepinfra``  OpenAI-compatible hosted API (env ``DEEPINFRA_API_KEY``)
``openai``     OpenAI API (env ``OPENAI_API_KEY``)

API keys are read from the environment only — never hard-code them.
"""
from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Dict, List, Sequence

from tqdm import tqdm

from .utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class _OpenAICompatibleLLM:
    """Shared implementation for any OpenAI-compatible chat endpoint."""

    def __init__(self, model_id: str, api_key: str, base_url: str | None = None,
                 max_workers: int = 16, max_retries: int = 8):
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                f"No API key for {self.__class__.__name__}. "
                "Export the corresponding environment variable first.")
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _one(self, prompt: str, system_prompt: str, max_new_tokens: int,
             temperature: float, n: int) -> List[str]:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n,
                )
                return [c.message.content or "" for c in completion.choices]
            except Exception as exc:  # transient API errors: exponential backoff
                if attempt == self.max_retries - 1:
                    LOGGER.error(f"LLM request failed permanently: {exc}")
                    return [""] * n
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return [""] * n

    def generate(self, prompts: Sequence[str] | str, system_prompt: str | None = None,
                 max_new_tokens: int = 1024, temperature: float = 0.7,
                 num_return_sequences: int = 1, **_) -> Dict[str, List[List[str]]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        results: List[List[str]] = [[] for _ in prompts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._one, p, system_prompt, max_new_tokens,
                            temperature, num_return_sequences): i
                for i, p in enumerate(prompts)
            }
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc="LLM"):
                results[futures[fut]] = fut.result()
        return {"responses": results}


class DeepInfraLLM(_OpenAICompatibleLLM):
    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, api_key=os.environ.get("DEEPINFRA_API_KEY", ""),
                         base_url="https://api.deepinfra.com/v1/openai", **kw)


class OpenAILLM(_OpenAICompatibleLLM):
    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, api_key=os.environ.get("OPENAI_API_KEY", ""), **kw)


class VLLMLLM:
    """Local vLLM. Prefix caching helps a lot: every prompt shares the demos."""

    def __init__(self, model_id: str, tensor_parallel_size: int = 1,
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.85,
                 max_num_seqs: int = 128):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel_size,
                       dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                       enable_prefix_caching=True, trust_remote_code=True)

    def generate(self, prompts, system_prompt=None, max_new_tokens=1024,
                 temperature=0.7, num_return_sequences=1, **_):
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        params = SamplingParams(n=num_return_sequences, temperature=temperature,
                                top_p=1.0, max_tokens=max_new_tokens)
        outputs = self.llm.generate(chat_prompts, params)  # vLLM preserves order
        return {"responses": [[o.outputs[i].text for i in range(len(o.outputs))]
                              for o in outputs]}


class HuggingFaceLLM:
    """Plain ``transformers`` generation (no extra serving dependency)."""

    def __init__(self, model_id: str, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.batch_size = batch_size
        kwargs = dict(device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="flash_attention_2", **kwargs).eval()
        except Exception:  # flash-attn is optional
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts, system_prompt=None, max_new_tokens=1024,
                 temperature=0.7, num_return_sequences=1, **_):
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        responses: List[List[str]] = []
        for start in tqdm(range(0, len(chat_prompts), self.batch_size), desc="LLM"):
            batch = chat_prompts[start:start + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(batch)):
                responses.append(texts[i * num_return_sequences:(i + 1) * num_return_sequences])
        return {"responses": responses}


def build_llm(backend: str, model_id: str, **kwargs):
    backend = backend.lower()
    LOGGER.info(f"LLM backend={backend} model={model_id}")
    if backend == "vllm":
        return VLLMLLM(model_id,
                       tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                       gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.85))
    if backend == "deepinfra":
        return DeepInfraLLM(model_id)
    if backend == "openai":
        return OpenAILLM(model_id)
    if backend in ("hf", "huggingface", "local"):
        return HuggingFaceLLM(model_id, batch_size=kwargs.get("batch_size", 8))
    raise ValueError(f"Unknown LLM backend: {backend}")
