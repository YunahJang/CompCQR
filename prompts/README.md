# Prompt assets

`demonstrations.json` — the few-shot dialogues used by the rewrite prompt
(step 0). We reuse the demonstration file released by **LLM4CS**
(Mao et al., *Large Language Models Know Your Contextual Search Intent*,
EMNLP 2023 Findings) rather than redistributing it here:

```bash
wget -O prompts/demonstrations.json \
  https://raw.githubusercontent.com/kyriemao/LLM4CS/main/demonstrations.json
```

Expected format — a list of dialogues, each turn carrying `question`, `cot`,
`manual_rewrite` and `response`:

```json
[
  {"turns": [
      {"question": "...", "cot": "...", "manual_rewrite": "...", "response": "..."}
  ]}
]
```

If the file is missing the pipeline still runs, but step 0 falls back to
zero-shot rewriting and quality drops.
