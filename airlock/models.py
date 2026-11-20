"""Local model components: Ollama embeddings + chat via LangChain, cross-encoder reranker.

Heavy libraries are imported lazily inside the builders so the core pipeline
(and its tests) can be imported without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .config import AirlockConfig

# ------------------------------------------------------------------ prompt

SYSTEM_PROMPT = """You are Airlock, a careful analyst answering questions about ONE document.
You are given numbered source excerpts from that document.

Rules:
1. Use ONLY the sources. Never use outside knowledge and never guess.
2. End every sentence with the citation(s) that support it, like [S1] or [S2][S4].
3. Copy numbers, dates, amounts and names exactly as they appear in the sources.
4. If the sources do not contain the answer, reply with exactly: NOT_IN_DOCUMENT
5. Be concise: at most 5 sentences."""

HUMAN_PROMPT = """Sources:
{sources}

Question: {question}"""

Generator = Callable[[str, str], str]   # (question, formatted_sources) -> answer text


class PrefixedEmbeddings:
    """Wraps a LangChain Embeddings object, adding the model's task prefixes."""

    def __init__(self, base, doc_prefix: str = "", query_prefix: str = "") -> None:
        self.base = base
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


def build_embedder(cfg: AirlockConfig) -> PrefixedEmbeddings:
    from langchain_ollama import OllamaEmbeddings

    doc_prefix, query_prefix = cfg.embed_prefixes
    base = OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.ollama_url)
    return PrefixedEmbeddings(base, doc_prefix, query_prefix)


def build_generator(cfg: AirlockConfig) -> Generator:
    """LangChain LCEL chain: prompt | ChatOllama | string parser."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    llm = ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_url, temperature=0, num_ctx=cfg.num_ctx)
    chain = prompt | llm | StrOutputParser()

    def generate(question: str, sources: str) -> str:
        return chain.invoke({"question": question, "sources": sources})

    return generate


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker:
    """Stage-2 reranker. Reads (question, passage) *together*, which is slower
    than comparing two independent vectors but far more precise - so it only
    runs on the short list from stage 1.

    Returns calibrated 0..1 relevance probabilities (sigmoid of the raw logit).
    """

    def __init__(self, model_path: str, batch_size: int = 32, max_length: int = 512) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Reranker not found at {model_path}. Run `airlock fetch-models` once (needs internet), "
                "after which Airlock runs fully offline."
            )
        from sentence_transformers import CrossEncoder

        try:
            self.model = CrossEncoder(model_path, max_length=max_length, local_files_only=True)
        except TypeError:  # older sentence-transformers without these kwargs
            self.model = CrossEncoder(model_path)
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        pairs = [(query, p) for p in passages]
        kwargs = dict(batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        try:   # sentence-transformers >= 4
            logits = self.model.predict(pairs, activation_fn=torch.nn.Identity(), **kwargs)
        except TypeError:  # sentence-transformers 2.x / 3.x
            logits = self.model.predict(pairs, activation_fct=torch.nn.Identity(), **kwargs)
        return _sigmoid(np.asarray(logits, dtype=np.float64).reshape(-1)).tolist()


def build_reranker(cfg: AirlockConfig) -> CrossEncoderReranker:
    return CrossEncoderReranker(cfg.reranker_path)


def build_index():
    from .index import ChromaIndex

    return ChromaIndex()
