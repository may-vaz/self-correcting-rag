# OWASP Top 10 for LLM Applications — Compliance Mapping

This document maps this project's architecture to the OWASP Top 10 for LLM
Applications and states honestly what is mitigated, partially mitigated, or
out of scope.

**Guardrail stack:** `input_guard`, the first node in the LangGraph state
machine, runs every user query through two local classifiers before it
touches the retriever or any LLM call:
1. **ProtectAI `deberta-v3-base-prompt-injection-v2`** — a dedicated
   classifier trained to detect prompt-injection / jailbreak *manipulation
   patterns* ("ignore previous instructions", "SYSTEM: override safety").
2. **Llama Guard 3** (via Ollama) — a content-safety classifier trained on
   harm categories (violence, weapons, hate speech, sexual content, etc).

Both run locally, no external API or key required. A query is blocked if
either layer flags it.

## Red-team results (last run: see terminal output / `red_team/results.json`)

| Category | Blocked | Rate | Threshold | Status |
|---|---|---|---|---|
| Prompt Injection | 5/5 | 100.0% | 95.0% | ✅ PASS |
| RAG Poisoning | 5/5 | 100.0% | 90.0% | ✅ PASS |
| Jailbreak | 5/5 | 100.0% | 90.0% | ✅ PASS |
| Toxicity | 5/5 | 100.0% | 90.0% | ✅ PASS |
| **Overall attack detection** | **20/20** | **100.0%** | — | ✅ |

**Over-defense check** (does the guardrail wrongly block legitimate,
domain-relevant queries containing trigger words like "ignore," "cancel,"
"override"?): 5/5 benign gardening questions allowed through (100%,
threshold 80%). See `red_team/attack_dataset.py` for the exact prompts and
`red_team/run_over_defense_only.py` to re-run this check in isolation.

These numbers are from a small, hand-built dataset (20 attack prompts, 5
benign) — enough to give an honest, reproducible signal, not a claim of
exhaustive coverage. Re-run `red_team/run_red_team.py` periodically and
after any change to `guardrails.py`.

## OWASP Top 10 mapping

| OWASP Risk | Mitigation | Status |
|---|---|---|
| **LLM01: Prompt Injection** | Every query passes through `input_guard` before reaching the retriever or any LLM. ProtectAI's classifier catches manipulation-pattern attacks (direct injection, "ignore previous instructions" style); Llama Guard catches harmful-content-based attacks. Flagged queries are blocked and never reach downstream nodes. Measured at 100% detection on the red-team set above. | ✅ Mitigated |
| **LLM02: Sensitive Information Disclosure** | The generation prompt restricts the LLM to answering only from retrieved context ("Use ONLY the context below"). No secrets or credentials are stored in the vector DB by design. There is no dedicated output-side PII scanner. | ⚠️ Partially Mitigated |
| **LLM03: Excessive Agency** | No tool-calling, no ability to take actions, no write access to external systems — read-only retrieval + generation. The retry loop is hard-capped at `MAX_RETRIES = 3` so the graph can't loop indefinitely. | ✅ Mitigated |
| **LLM04: Data and Model Poisoning** | Out of scope for this project's threat model — the vector DB is built once from a local, user-controlled `./data` folder of PDFs, not from untrusted or crowd-sourced input. If documents from external/untrusted sources are ever ingested, add a content scan at ingest time. | ⚠️ Out of scope (no untrusted ingestion pipeline exists) |
| **LLM05: Improper Output Handling** | The self-correction loop already covers this: `grade_generation` runs a hallucination check (is every claim grounded in retrieved context?) and a relevance check before an answer is returned. No separate output content filter was added — the design bet is that a grounded, context-only answer is the main output risk, and that's handled by the existing loop rather than duplicated. | ✅ Mitigated (via grounding checks, not a separate filter) |
| **LLM06: Excessive Permissions** | No database write access, no shell access, no network calls other than the local Ollama models and the local Chroma DB. Nothing external to reach. | ✅ Mitigated |
| **LLM07: System Prompt Leakage** | ProtectAI's classifier catches direct "reveal your system prompt" style attacks at the input stage (part of the prompt-injection category above). No secondary check scans *outputs* for leaked prompt text. | ⚠️ Partially Mitigated |
| **LLM08: Vector and Embedding Weaknesses** | Retrieval uses a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on top of raw vector similarity, plus a relevance threshold (`RELEVANCE_THRESHOLD = 0.3`) that discards weakly-related chunks before they reach the LLM. Reduces, doesn't eliminate, the chance of an adversarial or irrelevant chunk influencing the answer. | ✅ Partially Mitigated |
| **LLM09: Misinformation** | `grade_generation`'s hallucination stage re-checks every generated answer against the retrieved context before returning it, retrying generation if ungrounded claims are found. | ✅ Mitigated |
| **LLM10: Unbounded Consumption** | `MAX_RETRIES = 3` hard-caps the retrieve → grade → transform loop. No per-user rate limiting — judged unnecessary for a single-user local project; add one if this is ever exposed as a public service. | ⚠️ Partially Mitigated |

## What was deliberately left out, and why

- **No output guardrail / separate output content filter.** The
  self-correction loop (hallucination + relevance grading) already
  re-checks every answer against the source documents before it's
  returned. A second, separate output filter on top would be redundant for
  this project's scope — flagged as a known limitation (LLM07/LLM10)
  rather than silently ignored.
- **No CI/CD pipeline for red-team tests.** Run manually via
  `python3 red_team/run_red_team.py`. Enough for a single-developer
  project to produce credible, reproducible numbers; a CI step is a
  one-line addition later if ever needed, not a requirement now.
- **No data-poisoning defense (LLM04).** This project ingests only local,
  user-supplied PDFs, so there's no untrusted-input surface to defend yet.

## Where the evidence lives

- Guardrail implementation: [`guardrails.py`](../guardrails.py)
- Guardrail wired into the graph: [`rag_workflow.py`](../rag_workflow.py) (`input_guard` node)
- Attack + over-defense dataset: [`red_team/attack_dataset.py`](../red_team/attack_dataset.py)
- Red-team runner + latest results: [`red_team/run_red_team.py`](../red_team/run_red_team.py), `red_team/results.json`
- Over-defense-only test: [`red_team/run_over_defense_only.py`](../red_team/run_over_defense_only.py)