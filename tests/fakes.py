"""Deterministic stand-ins for the local models, so the full pipeline logic can be
tested without Ollama, ChromaDB or sentence-transformers installed.

They implement the same interfaces the real components expose:
  HashEmbedder        ~ LangChain Embeddings (embed_documents / embed_query)
  LexicalReranker     ~ CrossEncoderReranker.score -> 0..1 relevance
  ExtractiveGenerator ~ the LangChain prompt|ChatOllama chain (question, sources) -> text
  NumpyIndex          ~ ChromaIndex (add / query / count / close)
"""

from __future__ import annotations

import re
import zlib

import numpy as np

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to was what
when where which who will with within shall may must does do can after before than any each other such into
upon there their them they these those per much many long often quickly soon""".split())
GENERIC = {"agreem", "provid", "custom", "party", "partie"}   # appear everywhere in a contract
WORD = re.compile(r"[a-z0-9][a-z0-9.%$,'-]*")


SYNONYMS = {"uptime": "availa"}   # a pinch of "semantics" for the lexical stand-ins


def stems(text: str) -> list[str]:
    out = []
    for w in WORD.findall(text.lower()):
        w = w.strip(".,'-").replace(",", "")
        parts = w.split("-") if "-" in w and not w[:1].isdigit() else [w]
        for part in parts:
            if not part or part in STOP:
                continue
            if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
                part = part[:-1]
            stem = part[:6]  # crude stemming: terminate/termination -> "termin"
            out.append(SYNONYMS.get(stem, stem))
    return out


class HashEmbedder:
    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = stems(text)
        for t in toks:
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        for a, b in zip(toks, toks[1:]):
            v[zlib.crc32(f"{a}_{b}".encode()) % self.dim] += 0.5
        v = np.sqrt(v)
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls += 1
        return self._vec(text)


def coverage(question: str, passage: str) -> float:
    q = [s for s in dict.fromkeys(stems(question)) if s not in GENERIC]
    if not q:
        return 0.0
    p = set(stems(passage))
    return sum(s in p for s in q) / len(q)


class LexicalReranker:
    def score(self, query, passages):
        return [coverage(query, p) for p in passages]


_SOURCE = re.compile(r"\[(S\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[S\d+\] \(|\Z)", re.S)


class ExtractiveGenerator:
    """Answers with the single best-matching source sentence, cited."""

    def __init__(self, min_coverage: float = 0.5) -> None:
        self.min_coverage = min_coverage
        self.calls = 0

    def pick(self, question: str, sources: str) -> tuple[str, str, float]:
        from airlock.chunking import split_sentences

        best = None
        for label, text in _SOURCE.findall(sources):
            for sent in split_sentences(text):
                c = coverage(question, sent)
                if best is None or c > best[2] or (c == best[2] and len(sent) > len(best[1])):
                    best = (label, sent, c)
        return best or ("S1", "", 0.0)

    def __call__(self, question: str, sources: str) -> str:
        self.calls += 1
        label, sent, cov = self.pick(question, sources)
        if cov < self.min_coverage:
            return "NOT_IN_DOCUMENT"
        return f"{sent} [{label}]"


class LyingGenerator(ExtractiveGenerator):
    """Cites the right passage but changes the number - a classic hallucination."""

    def __call__(self, question: str, sources: str) -> str:
        label, sent, _ = self.pick(question, sources)
        return re.sub(r"\(90\)", "(60)", sent).replace("ninety", "sixty") + f" [{label}]"


class UncitedGenerator(ExtractiveGenerator):
    """Answers from 'memory': plausible text, no citation, not in the sources."""

    def __call__(self, question: str, sources: str) -> str:
        return "The agreement can be terminated with two weeks of notice by email."


class NumpyIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.m = np.zeros((0, 0), np.float32)
        self.closed = False

    def add(self, ids, embeddings, documents, metadatas):
        emb = np.asarray(embeddings, np.float32)
        self.m = emb if self.m.size == 0 else np.vstack([self.m, emb])
        self.ids += list(ids)

    def query(self, embedding, k):
        q = np.asarray(embedding, np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.m @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def count(self):
        return len(self.ids)

    def close(self):
        self.ids, self.m, self.closed = [], np.zeros((0, 0), np.float32), True


def fake_session(config=None, generator=None, reranker="default"):
    from airlock import Airlock, AirlockConfig

    cfg = config or AirlockConfig(min_rerank_prob=0.5, min_vector_sim=0.05, support_threshold=0.6,
                                  min_chunk_chars=200, max_chunk_chars=1200)
    rr = LexicalReranker() if reranker == "default" else reranker
    return Airlock(HashEmbedder(), generator or ExtractiveGenerator(), rr if cfg.use_reranker else None,
                   NumpyIndex, cfg, {"embeddings": "fake/hash", "llm": "fake/extractive", "reranker": "fake/lexical"})
