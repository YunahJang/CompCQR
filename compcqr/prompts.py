"""Prompt templates and output parsers for the three LLM calls of the pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import get_logger

LOGGER = get_logger(__name__)

SYSTEM_ASSISTANT = "You are a helpful assistant."
SYSTEM_SELECTOR = ("You are an expert in question answering. Your task is to select the "
                   "most relevant passage from the given passages based on the query.")

# ═════════════════════════════════════════════════════════════════════════
#  Step 0 — rewrite + response (LLM4CS style, few-shot with demonstrations)
# ═════════════════════════════════════════════════════════════════════════
REWRITE_INSTRUCTION = (
    "For an information-seeking dialog, please help reformulate the question into rewrite "
    "that can fully express the user's information needs without the need of context, but also "
    "generate an informative response to answer the question. I will give you several example "
    "multi-turn dialogs, where each turn contains a question as well as a rewrite and a response "
    "that you need to generate. The rewrite part begins with a sentence explaining the reason for "
    "the generated rewrite."
)
REWRITE_TAIL_INSTRUCTION = (
    'Now, you should give me the rewrite and response of the **Current Question** under the '
    '**Context**. The output format should always be: '
    '"Rewrite: $Reason. So the question should be rewritten as: $Rewrite\\nResponse: $Response."'
)

# ═════════════════════════════════════════════════════════════════════════
#  Step 1 — follow-up query + its answer
# ═════════════════════════════════════════════════════════════════════════
FOLLOWUP_SYSTEM_PROMPT = (
    'Generate a follow-up query and its answer based on the previous query and answer. '
    'The follow-up query should be highly relevant to the topic and context of the conversation. '
    '(Closely related so that both questions are likely to be answered by a passage.) '
    'If no follow-up is needed, return an empty string. '
    'Please format your output exactly as follows: '
    '{{"followup_query": "Your highly related query here", '
    '"next_answer": "Your corresponding answer here"}}'
)

# ═════════════════════════════════════════════════════════════════════════
#  Step 5 — passage selection
# ═════════════════════════════════════════════════════════════════════════
SELECTION_PROMPT_TEMPLATE = """I will provide you with {top_k} passages, each with a passage number. Select the passage based on their relevance to the search query: {{question}}.
Dialogue History: {context}
Search Query: {query}
Passages:
{passage_text}

Please follow the steps below:
Step 1: List up the information required to answer the search query and rewrite the query into a standalone and decontextualized query.
Step 2: Analyze whether each passage has the information required to answer the search query.
Step 3: Choose the passage that mostly covers clear and diverse information to answer the query.
The format of the final output should be as follows:
{{
"query analysis": "<step 1 output, your analysis of the query and the rewritten query>",
"passage analysis": "<step 2 output, analysis>",
"reason for selection": "<step 3 output,reason for selecting the passage>",
"selected_passage_number": "<the number of the selected passage, chosen from 1 to {top_k}>"
}}
"""


def load_demonstrations(demo_file: Path) -> str:
    """Render the few-shot demonstrations used by the rewrite prompt.

    The file format is the one released by LLM4CS (Mao et al., 2023):
    a list of dialogues, each turn carrying ``cot``/``manual_rewrite``/``response``.
    """
    try:
        demos = json.loads(Path(demo_file).read_text())
    except Exception:
        LOGGER.warning(f"No demonstration file at {demo_file}; running zero-shot.")
        return ""
    examples = []
    for demo in demos:
        dialog = []
        for turn in demo["turns"]:
            rewrite = (turn["cot"] +
                       " So the question should be rewritten as: {}".format(turn["manual_rewrite"]))
            dialog.append("Question: {}\nRewrite: {}\nResponse: {}".format(
                turn["question"], rewrite, turn["response"]))
        examples.append(f"Example #{len(examples) + 1}:\n" + "\n\n".join(dialog))
    return "\n\n".join(examples)


def build_rewrite_prompt(context_block: str, demo: str) -> str:
    return "\n\n".join([REWRITE_INSTRUCTION, demo, context_block, REWRITE_TAIL_INSTRUCTION])


def build_followup_prompt(context: str, query: str) -> str:
    return f"Context: {context} Current Query: {query}"


def build_selection_prompt(query: str, context: str, passages: List[str], top_k: int) -> str:
    passage_text = "\n\n".join(f"Passage {i + 1}. {p}" for i, p in enumerate(passages))
    return SELECTION_PROMPT_TEMPLATE.format(
        top_k=top_k, context=context, query=query, passage_text=passage_text)


# ═════════════════════════════════════════════════════════════════════════
#  Parsers
# ═════════════════════════════════════════════════════════════════════════
def parse_rewrite_outputs(texts: List[str]
                          ) -> Tuple[List[str | None], List[str | None], List[str | None]]:
    """Parse "Rewrite: $reason. So the question should be rewritten as: $q\\nResponse: $a".

    Returns (rewrites, responses, chains-of-thought); entries are ``None`` when
    the sample could not be parsed, and the caller applies a fallback.
    """
    cots: List[str | None] = []
    rewrites: List[str | None] = []
    responses: List[str | None] = []

    def _fail():
        cots.append(None), rewrites.append(None), responses.append(None)

    for text in texts:
        text = (text or "").strip()
        lines = text.split("\n")
        if not lines or not lines[0].startswith("Rewrite: "):
            _fail()
            continue
        response_idx = next((i for i, line in enumerate(lines[1:], 1)
                             if line.startswith("Response: ")), None)
        if response_idx is None:
            _fail()
            continue
        rewrite_line = lines[0][len("Rewrite: "):]
        marker = "the question should be rewritten as: "
        pos = rewrite_line.find(marker)
        if pos == -1:
            _fail()
            continue
        cots.append(rewrite_line[:pos])
        rewrites.append(rewrite_line[pos + len(marker):])
        responses.append("\n".join(lines[response_idx:])[len("Response: "):])
    return rewrites, responses, cots


def repair_json(raw: str) -> str:
    """Best-effort repair of an LLM's JSON object (fences, doubled braces, trailing commas)."""
    text = str(raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    text = text[start:end + 1].strip()
    while text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1].strip()
    while text.count("{") > text.count("}"):
        text += "}"
    while text.count("{") < text.count("}"):
        text = text[:text.rfind("}")]
    return re.sub(r",\s*([}\]])", r"\1", text)


# Prose fallbacks, used when the selector answers in free text instead of JSON.
_PROSE_PATTERNS = [
    re.compile(r"selected\s+passage\s+number\D*?(\d+)", re.IGNORECASE),
    re.compile(r"passage\s+number\s+(?:is\s+)?\D*?(\d+)", re.IGNORECASE),
    re.compile(r"passage\s+(\d+)\s+is\s+(?:the\s+)?(?:most\s+)?relevant", re.IGNORECASE),
    re.compile(r"(?:select|choose|pick)(?:ed|ing|s)?\b\D{0,30}?passage\s+(\d+)", re.IGNORECASE),
]


def parse_selected_passage(text: str, lo: int = 1, hi: int = 10) -> int | None:
    """Extract the selected passage number; returns ``None`` if nothing parses.

    Three cascading strategies: repaired JSON -> key regex -> prose patterns.
    """
    repaired = repair_json(text)
    if repaired:
        try:
            value = json.loads(repaired).get("selected_passage_number", "")
            numbers = [int(n) for n in re.findall(r"\d+", str(value))]
            numbers = [n for n in numbers if lo <= n <= hi]
            if numbers:
                return numbers[0]
        except Exception:
            pass

    match = re.search(r'selected_passage_number["\s:]*(\d+)', str(text))
    if match and lo <= int(match.group(1)) <= hi:
        return int(match.group(1))

    for pattern in _PROSE_PATTERNS:
        numbers = [int(n) for n in pattern.findall(str(text))]
        numbers = [n for n in numbers if lo <= n <= hi]
        if numbers:
            return numbers[-1]
    return None
