"""CompCQR — Compositional Conversational Query Reformulation.

An end-to-end, training-free pipeline that (1) decomposes a conversational turn
into reusable query components, (2) composes them into many query variations,
(3) retrieves with every variation, (4) fuses the ranked lists with RRF, and
(5) lets an LLM pick the single best passage, which is promoted to rank 1.
"""

__version__ = "1.0.0"
