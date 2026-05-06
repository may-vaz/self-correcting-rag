# Self-Correcting RAG System for Rose Gardening Knowledge

A robust, production-grade **Retrieval-Augmented Generation (RAG)** system with self-correction capabilities, built for accurate question-answering over PDF documents (rose cultivation guides).

---

## ✨ Key Features

- **Self-Correcting Pipeline** — Automatically detects hallucinations and irrelevant answers, then retries or rewrites queries.
- **Intelligent Retrieval** — Hybrid retrieval with **Cross-Encoder reranking** for higher relevance.
- **LLM-as-Judge** — Multi-stage grading for document relevance, faithfulness, and answer quality.
- **Bounded Loops** — Strict `MAX_RETRIES` prevents infinite loops (fixed & hardened).
- **Source Citations** — Returns traceable sources with page numbers.
- **Local-first** — Fully runs on Ollama + local embeddings (no cloud costs).

---

## 🏗️ Architecture Overview

The system uses a **LangGraph** state machine with the following flow:

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

This creates a **reliable, self-healing RAG loop**.

---

## 🛠️ Tech Stack

| Component              | Technology                              |
|------------------------|-----------------------------------------|
| **LLM**                | Llama 3 / Llama 3.1 (via Ollama)      |
| **Embeddings**         | `nomic-embed-text` (Ollama)            |
| **Vector Store**       | ChromaDB                               |
| **Framework**          | LangChain + **LangGraph**              |
| **Reranker**           | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Evaluation**         | Ragas + custom metrics                 |
| **Document Loader**    | PyPDFDirectoryLoader                   |
| **Language**           | Python 3.12                            |

---

## 📋 Installation & Setup

### 1. Clone & Setup Environment

```bash
git clone <your-repo>
cd self-correcting-rag
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
