# Self-Correcting RAG System for Gardening Knowledge

A robust, production-grade **Retrieval-Augmented Generation (RAG)** system with self-correction capabilities, built for accurate question-answering over PDF documents (rose cultivation guides).

---

## Key Features

- **Self-Correcting Pipeline** — Automatically detects hallucinations and irrelevant answers, then retries or rewrites queries.
- **Intelligent Retrieval** — Hybrid retrieval with **Cross-Encoder reranking** for higher relevance.
- **LLM-as-Judge** — Multi-stage grading for document relevance, faithfulness, and answer quality.
- **Bounded Loops** — Strict `MAX_RETRIES` prevents infinite loops (fixed & hardened).
- **Source Citations** — Returns traceable sources with page numbers.
- **Local-first** — Fully runs on Ollama + local embeddings (no cloud costs).
- **Input Safety Guardrails** — Every query is screened for prompt injection and harmful content before it reaches the RAG pipeline.

---

## Architecture Overview

The system uses a **LangGraph** state machine with the following flow:

0. **Input Guard** → Screens the query for prompt injection / jailbreak patterns (ProtectAI classifier) and harmful content (Llama Guard). Blocked queries stop here — nothing else runs.
1. **Retrieve** → Fetch top chunks from ChromaDB + Cross-Encoder reranking
2. **Grade Documents** → LLM filters truly relevant chunks
3. **Generate** → LLM produces answer using only retrieved context
4. **Grade Generation** → Two-stage check:
   - Hallucination detection (faithfulness)
   - Answer relevancy to original question
5. **Self-Correction**:
   - If hallucination → **regenerate**
   - If irrelevant → **transform query** + loop
   - Max retries enforced → fallback answer

This creates a **reliable, self-healing, and safety-screened RAG loop**.

---

## AI Safety

This project follows the **OWASP Top 10 for LLM Applications**. Full mapping,
red-team test results, and design decisions: [`docs/owasp_compliance.md`](docs/owasp_compliance.md).

- Guardrail code: [`guardrails.py`](guardrails.py)
- Attack test suite: `python3 red_team/run_red_team.py`
- Over-defense check (does it wrongly block real users?): `python3 red_team/run_over_defense_only.py`
- Every guardrail decision is logged to `logs/guardrail_log.jsonl`

---

## Tech Stack

| Component              | Technology                              |
|------------------------|------------------------------------------|
| **LLM**                | Llama 3 / Llama 3.1 (via Ollama)         |
| **Embeddings**         | `nomic-embed-text` (Ollama)              |
| **Vector Store**       | ChromaDB                                 |
| **Framework**          | LangChain + **LangGraph**                |
| **Reranker**           | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| **Input Guardrails**   | ProtectAI injection classifier + Llama Guard 3 |
| **Document Loader**    | PyPDFDirectoryLoader                     |
| **Language**           | Python 3.12                              |

---

## Installation & Setup

### 1. Clone & Setup Environment

```bash
git clone <your-repo>
cd self-correcting-rag
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull required Ollama models

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama pull llama-guard3
```

### 3. Build the vector database

Put your PDFs in `./data`, then:

```bash
python3 ingest.py
```

### 4. Run it

```bash
python3 app.py
```
