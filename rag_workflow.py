from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph
from sentence_transformers import CrossEncoder
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_RETRIES      = 3          # hard cap on retrieve→grade→transform loops
RELEVANCE_THRESHOLD = 0.3    # cross-encoder score below this → discard chunk
TOP_K_RETRIEVAL  = 6         # fetch more chunks; reranker will prune them down
TOP_K_AFTER_RANK = 3         # keep only the top-N after reranking

# ─────────────────────────────────────────────
# 1. GRAPH STATE
# ─────────────────────────────────────────────
class GraphState(TypedDict):
    question:    str
    generation:  str
    documents:   List[Document]
    sources:     List[str]          # NEW – citations carried through graph
    iterations:  int                # NEW – loop counter


# ─────────────────────────────────────────────
# 2. MODELS & STORES
# ─────────────────────────────────────────────
llm     = ChatOllama(model="llama3", format="json", temperature=0)
llm_gen = ChatOllama(model="llama3", temperature=0.2)

embeddings  = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})

# Cross-encoder reranker (runs locally via sentence-transformers)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ─────────────────────────────────────────────
# 3. HELPER – build citation string
# ─────────────────────────────────────────────
def _extract_source(doc: Document) -> str:
    meta   = doc.metadata or {}
    source = meta.get("source", "unknown")
    page   = meta.get("page")
    return f"{source} (page {page + 1})" if page is not None else source


# ─────────────────────────────────────────────
# 4. NODE FUNCTIONS
# ─────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    print("---RETRIEVAL---")
    question  = state["question"]
    raw_docs  = retriever.invoke(question)

    # ── Cross-encoder reranking ──────────────────────────────────────────
    if raw_docs:
        pairs  = [(question, d.page_content) for d in raw_docs]
        scores = cross_encoder.predict(pairs)           # float32 array
        ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)

        # Apply threshold & keep top-K
        docs = [
            doc for score, doc in ranked
            if score >= RELEVANCE_THRESHOLD
        ][:TOP_K_AFTER_RANK]

        if not docs:
            # Nothing passed threshold – keep the single best anyway so the
            # grader can decide, rather than triggering an immediate rewrite
            docs = [ranked[0][1]]

        print(f"  Retrieved {len(raw_docs)} chunks → {len(docs)} kept after reranking")
    else:
        docs = []

    return {
        **state,
        "documents": docs,
        "question":  question,
        "iterations": state.get("iterations", 0),
    }


def grade_documents(state: GraphState) -> GraphState:
    print("---CHECKING DOCUMENT RELEVANCE (LLM grader)---")
    question  = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
Retrieved document:
{document}

User question: {question}

If the document contains keywords or concepts related to the question, grade it relevant.
Return JSON with a single key 'score' whose value is 'yes' or 'no'.""",
        input_variables=["question", "document"],
    )
    chain = prompt | llm | JsonOutputParser()

    filtered = []
    for doc in documents:
        try:
            result = chain.invoke({"question": question, "document": doc.page_content})
            grade  = result.get("score", "no")
        except Exception:
            grade = "no"

        if grade.lower() == "yes":
            print("  GRADE: RELEVANT")
            filtered.append(doc)
        else:
            print("  GRADE: IRRELEVANT – dropped")

    return {**state, "documents": filtered, "question": question}


def generate(state: GraphState) -> GraphState:
    print("---GENERATING ANSWER---")
    question  = state["question"]
    documents = state["documents"]

    context = "\n\n".join(doc.page_content for doc in documents)
    sources = list({_extract_source(d) for d in documents})   # deduplicated

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."
Be concise – three sentences maximum.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["question", "context"],
    )

    chain      = prompt | llm_gen | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {**state, "documents": documents, "question": question,
            "generation": generation, "sources": sources}


def transform_query(state: GraphState) -> GraphState:
    print("---TRANSFORMING QUERY---")
    question   = state["question"]
    iterations = state.get("iterations", 0) + 1

    prompt = PromptTemplate(
        template="""You are optimizing a retrieval query.
Analyse the semantic intent of the question below and rewrite it so that
a vector search will find the most relevant passages.

Original question:
{question}

Return JSON with a single key 'question' containing the rewritten question.""",
        input_variables=["question"],
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        response      = chain.invoke({"question": question})
        better_question = response.get("question", question)
    except Exception:
        better_question = question

    print(f"  NEW QUERY: {better_question}")
    return {**state, "question": better_question, "iterations": iterations}


# ─────────────────────────────────────────────
# 5. CONDITIONAL EDGES
# ─────────────────────────────────────────────

def decide_to_generate(state: GraphState) -> str:
    print("---ROUTING: enough relevant docs?---")
    iterations = state.get("iterations", 0)

    if iterations >= MAX_RETRIES:
        print(f"  MAX RETRIES ({MAX_RETRIES}) reached – forcing generate")
        return "generate"

    if not state["documents"]:
        print("  No relevant docs → transform query")
        return "transform_query"

    print("  Relevant docs found → generate")
    return "generate"


def grade_generation(state: GraphState) -> str:
    """
    Two-stage check:
      1. Hallucination: is the answer grounded in the retrieved context?
      2. Relevance: does the answer actually address the question?
    Returns one of: 'useful' | 'not_supported' | 'not_useful'
    """
    print("---GRADING GENERATION---")
    question   = state["question"]
    documents  = state["documents"]
    generation = state["generation"]
    iterations = state.get("iterations", 0)

    # Hard stop – don't loop forever
    if iterations >= MAX_RETRIES:
        print(f"  MAX RETRIES reached – accepting answer as-is")
        return "useful"

    context = "\n\n".join(doc.page_content for doc in documents)

    # ── Stage 1: hallucination grader ───────────────────────────────────
    hal_prompt = PromptTemplate(
        template="""You are checking whether an LLM answer is fully grounded in the
provided facts. Answer 'yes' only if EVERY claim in the answer can be traced
back to the facts. Answer 'no' if ANY part is invented or assumed.

Facts:
{documents}

LLM answer: {generation}

Return JSON with a single key 'score' whose value is 'yes' or 'no'.""",
        input_variables=["generation", "documents"],
    )
    try:
        hal_score = (hal_prompt | llm | JsonOutputParser()).invoke(
            {"documents": context, "generation": generation}
        )
        grounded = hal_score.get("score", "yes").lower() == "yes"
    except Exception:
        grounded = True   # default to pass on parse error

    if not grounded:
        print("  HALLUCINATION detected → retry generate")
        return "not_supported"

    print("  Answer is grounded – checking relevance …")

    # ── Stage 2: answer relevance grader ────────────────────────────────
    ans_prompt = PromptTemplate(
        template="""Does the answer below fully address the question?
Question: {question}
Answer:   {generation}

Return JSON with a single key 'score' whose value is 'yes' or 'no'.""",
        input_variables=["generation", "question"],
    )
    try:
        ans_score = (ans_prompt | llm | JsonOutputParser()).invoke(
            {"question": question, "generation": generation}
        )
        useful = ans_score.get("score", "yes").lower() == "yes"
    except Exception:
        useful = True

    if useful:
        print("  Answer is relevant → DONE")
        return "useful"
    else:
        print("  Answer doesn't resolve question → rewrite query")
        return "not_useful"


# ─────────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve",         retrieve)
workflow.add_node("grade_documents",  grade_documents)
workflow.add_node("generate",         generate)
workflow.add_node("transform_query",  transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not_supported": "generate",       # hallucination → retry
        "useful":        END,              # good answer → done
        "not_useful":    "transform_query",# wrong answer → rewrite query
    },
)

app_graph = workflow.compile()