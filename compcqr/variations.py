"""Compositional query variations.

A turn is decomposed into five components::

    oq   original (context-dependent) user query
    mq   machine query      — the LLM rewrite of the current turn
    ma   machine answer     — the LLM's hypothetical answer to the current turn
    mq2  machine follow-up query
    ma2  machine follow-up answer

Every ordered subset of the components (size 1..``max_k``) is one *rule*, and
concatenating the corresponding texts yields one query variation. With five
components and ``max_k=5`` this gives 5+20+60+120+120 = **325** variations per
turn; ``max_k=1`` degenerates to the five single-component queries, which is the
cheap configuration used in the efficiency ablation.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

COMPONENT_FULL_NAME = {
    "oq": "original query",
    "mq": "machine query",
    "ma": "machine answer",
    "mq2": "machine follow-up query",
    "ma2": "machine follow-up answer",
}


def generate_rules(components: Sequence[str], max_k: int, joiner: str = "+") -> List[Dict]:
    """All ordered subsets of ``components`` of size 1..``max_k``, in a stable order."""
    rules: List[Dict] = []
    rule_id = 1
    max_k = min(max_k, len(components))
    for k in range(1, max_k + 1):
        for perm in itertools.permutations(sorted(components), k):
            rules.append({
                "rule_id": rule_id,
                "components": list(perm),
                "rule": joiner.join(perm),
                "explanation": " + ".join(COMPONENT_FULL_NAME.get(c, c) for c in perm),
            })
            rule_id += 1
    return rules


def build_variations(parts: Dict[str, str], components: Sequence[str],
                     max_k: int, joiner: str = "+") -> List[Dict]:
    """Instantiate every rule with this turn's component texts."""
    return [
        {
            "rule_id": rule["rule_id"],
            "rule": rule["rule"],
            "explanation": rule["explanation"],
            "text": joiner.join(parts.get(c, "") for c in rule["components"]),
        }
        for rule in generate_rules(components, max_k, joiner)
    ]


def num_variations(components: Sequence[str], max_k: int) -> int:
    return len(generate_rules(components, max_k))
