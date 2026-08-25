# OWASP Top 10 for LLM Applications — Compliance

**Guardrail:** every query passes through `input_guard` (first node in the
LangGraph) before it reaches retrieval or generation. Two classifiers run in sequence, a query is blocked if either flags it:
1. **ProtectAI `deberta-v3-base-prompt-injection-v2`** — detects injection/jailbreak *patterns* ("ignore previous instructions").
2. **Llama Guard 3** (via Ollama) — detects harmful *content* (violence, hate, etc).

## Red-team results
20/20 attack prompts blocked (100%) across prompt injection, RAG poisoning,
jailbreak, and toxicity. Over-defense check: 5/5 benign domain questions
(containing trigger words like "ignore," "override") correctly allowed
through. Small hand-built dataset — a real signal, not a claim of full
coverage. Re-run: `python3 red_team/run_red_team.py`.

## Mapping

| Risk | Mitigation | Status |
|---|---|---|
| LLM01 Prompt Injection | `input_guard` blocks before retrieval/generation. 100% on red-team set. | ✅ |
| LLM02 Sensitive Info Disclosure | Generation prompt restricted to retrieved context only; no output PII scan. | ⚠️ Partial |
| LLM03 Excessive Agency | No tool-calling, no write access, retries capped at `MAX_RETRIES=3`. | ✅ |
| LLM04 Data/Model Poisoning | Vector DB built once from local, user-controlled PDFs — no untrusted ingestion. | Out of scope |
| LLM05 Improper Output Handling | Hallucination + relevance grading loop re-checks every answer before returning it. | ✅ |
| LLM06 Excessive Permissions | No DB writes, no shell access, no external calls beyond local Ollama/Chroma. | ✅ |
| LLM07 System Prompt Leakage | Caught at input by the injection classifier; no output-side scan. | ⚠️ Partial |
| LLM08 Vector/Embedding Weaknesses | Cross-encoder reranker + relevance threshold discard weak/adversarial chunks. | ⚠️ Partial |
| LLM09 Misinformation | Hallucination grader re-checks generated claims against retrieved context. | ✅ |
| LLM10 Unbounded Consumption | Hard retry cap; no rate limiting (not needed — single local user). | ⚠️ Partial |


## Evidence
`guardrails.py` · `rag_workflow.py` (`input_guard` node) · `red_team/attack_dataset.py` · `red_team/run_red_team.py` · `red_team/results.json` · `logs/guardrail_log.jsonl`
