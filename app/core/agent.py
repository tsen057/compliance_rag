"""
app/core/agent.py
──────────────────
LangGraph agent with query routing.

Graph structure:
  ┌─────────────┐
  │  route_query │  Classifies question as simple / complex
  └──────┬───────┘
         │
    ┌────▼────┐
    │ retrieve │  Fetches top-k chunks from FAISS
    └────┬─────┘
         │
  ┌──────▼──────┐      ┌──────────────────┐
  │   Simple?   │─yes─►│  generate_direct │  Single-step RAG answer
  └──────┬──────┘      └──────────────────┘
         │ no
  ┌──────▼──────────┐
  │ generate_reason │  Multi-step reasoning over context
  └─────────────────┘

The LLM is google/flan-t5-base — local, free, no API key needed.
"""

from __future__ import annotations

import sys
from typing import TypedDict

from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langgraph.graph import END, StateGraph
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from app.core.config import get_settings
from app.core.retriever import retrieve
from app.core.schemas import QueryType, SourceChunk

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)

# ── LLM (loaded once, cached at module level) ─────────────────────────────────

_llm: HuggingFacePipeline | None = None


def _get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is None:
        settings = get_settings()
        logger.info(f"Loading LLM: {settings.llm_model}")
        tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model)
        hf_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=False,
        )
        _llm = HuggingFacePipeline(pipeline=hf_pipe)
        logger.info("LLM ready")
    return _llm


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    top_k: int
    query_type: str           # "simple" | "complex"
    sources: list[SourceChunk]
    context: str
    answer: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_ROUTING_PROMPT = PromptTemplate.from_template(
    """Classify the compliance question below as either 'simple' or 'complex'.

- simple  : asks for a single fact, number, or definition
- complex : requires comparing concepts, multi-step reasoning, or synthesising multiple topics

Question: {question}

Answer with exactly one word — simple or complex:"""
)

_RAG_PROMPT = PromptTemplate.from_template(
    """You are a regulatory compliance assistant.
Answer the question using ONLY the context provided.
If the context does not contain enough information, say:
"I cannot find sufficient information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)

_REASONING_PROMPT = PromptTemplate.from_template(
    """You are a regulatory compliance expert.
The question requires careful analysis. Think step by step using ONLY the context provided.
If the context is insufficient, say so clearly.

Context:
{context}

Question: {question}

Step-by-step reasoning and answer:"""
)


# ── Node functions ────────────────────────────────────────────────────────────

def _route_query(state: AgentState) -> AgentState:
    llm = _get_llm()
    prompt = _ROUTING_PROMPT.format(question=state["question"])
    raw = llm.invoke(prompt).strip().lower()
    query_type = "complex" if "complex" in raw else "simple"
    logger.info(f"Routing decision: {query_type!r} for {state['question'][:60]!r}")
    return {**state, "query_type": query_type}


def _retrieve(state: AgentState) -> AgentState:
    sources = retrieve(state["question"], top_k=state["top_k"])
    context = "\n\n---\n\n".join(s.excerpt for s in sources)
    return {**state, "sources": sources, "context": context}


def _generate_direct(state: AgentState) -> AgentState:
    llm = _get_llm()
    prompt = _RAG_PROMPT.format(context=state["context"], question=state["question"])
    answer = llm.invoke(prompt).strip()
    logger.info("Answer generated (direct RAG)")
    return {**state, "answer": answer}


def _generate_reasoned(state: AgentState) -> AgentState:
    llm = _get_llm()
    prompt = _REASONING_PROMPT.format(context=state["context"], question=state["question"])
    answer = llm.invoke(prompt).strip()
    logger.info("Answer generated (multi-step reasoning)")
    return {**state, "answer": answer}


def _branch(state: AgentState) -> str:
    return state["query_type"]  # "simple" or "complex"


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("route_query", _route_query)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate_direct", _generate_direct)
    g.add_node("generate_reasoned", _generate_reasoned)

    g.set_entry_point("route_query")
    g.add_edge("route_query", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        _branch,
        {"simple": "generate_direct", "complex": "generate_reasoned"},
    )
    g.add_edge("generate_direct", END)
    g.add_edge("generate_reasoned", END)
    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

class ComplianceAgent:
    """
    High-level interface used by the API routes and the evaluation script.

    Usage:
        agent = ComplianceAgent()
        result = agent.query("What is the CET1 ratio under Basel III?")
        # result["answer"], result["sources"], result["query_type"]
    """

    def query(self, question: str, top_k: int | None = None) -> dict:
        settings = get_settings()
        top_k = top_k or settings.default_top_k

        initial: AgentState = {
            "question": question,
            "top_k": top_k,
            "query_type": "simple",
            "sources": [],
            "context": "",
            "answer": "",
        }

        result: AgentState = _get_graph().invoke(initial)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "query_type": result["query_type"],
        }
