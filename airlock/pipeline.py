"""The Airlock session: ingest one PDF, answer questions about it, forget it."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import airgap
from .chunking import (Chunk, Embedder, ProgressFn, document_sentences, embed_in_batches,
                       fixed_chunks, semantic_chunks)
from .config import AirlockConfig
from .pdf import load_pdf
from .provenance import (NOT_FOUND_TOKEN, Answer, ProvenanceLog, ProvenanceRecord, SourceRef,
                         candidate_rows, format_sources, validate_answer)

Generator = Callable[[str, str], str]


@dataclass
class IngestReport:
    document: str
    pages: int
    empty_pages: list[int]
    sentences: int
    chunks: int
    avg_chunk_chars: float
    chunker: str
    seconds: float
    stripped_lines: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.pages and len(self.empty_pages) / self.pages > 0.2:
            out.append(f"{len(self.empty_pages)} of {self.pages} pages have no extractable text "
                       "(scanned images?). Run OCR first for full coverage.")
        return out


@dataclass
class Retrieval:
    question: str
    ranked: list[tuple[str, float, float | None]]   # (chunk_id, vector cosine, rerank prob)
    timings_ms: dict

    @property
    def best_vector_sim(self) -> float:
        return max((v for _, v, _ in self.ranked), default=0.0)

    @property
    def best_rerank_prob(self) -> float | None:
        probs = [p for _, _, p in self.ranked if p is not None]
        return max(probs) if probs else None


class Airlock:
    """One document, one session, zero retention.

    >>> with Airlock.local() as session:
    ...     session.ingest(pdf_bytes, "contract.pdf")
    ...     print(session.ask("What is the termination notice period?").text)
    """

    def __init__(self, embedder: Embedder, generator: Generator | None = None, reranker=None,
                 index_factory: Callable[[], object] | None = None,
                 config: AirlockConfig | None = None, model_info: dict | None = None) -> None:
        self.config = config or AirlockConfig()
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        if index_factory is None:
            from .models import build_index
            index_factory = build_index
        self._index_factory = index_factory
        self._index = None
        self._chunks: dict[str, Chunk] = {}
        self.document: dict | None = None
        self.provenance = ProvenanceLog()
        self.model_info = model_info or {}

    # ------------------------------------------------------------ construction

    @classmethod
    def local(cls, config: AirlockConfig | None = None, enforce_airgap: bool = True) -> "Airlock":
        """Real stack: Ollama (LangChain) + ChromaDB in RAM + local cross-encoder."""
        if enforce_airgap:
            airgap.install()
        from .models import build_embedder, build_generator, build_index, build_reranker

        cfg = config or AirlockConfig.from_env()
        return cls(
            embedder=build_embedder(cfg),
            generator=build_generator(cfg),
            reranker=build_reranker(cfg) if cfg.use_reranker else None,
            index_factory=build_index,
            config=cfg,
            model_info={"embeddings": f"ollama/{cfg.embed_model}", "llm": f"ollama/{cfg.llm_model}",
                        "reranker": cfg.reranker_hf_id if cfg.use_reranker else None},
        )

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "Airlock":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Purge the index, chunks and provenance log from memory."""
        if self._index is not None:
            self._index.close()
        self._index = None
        self._chunks.clear()
        self.provenance.clear()
        self.document = None
        gc.collect()

    @property
    def ready(self) -> bool:
        return self._index is not None and bool(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    # ------------------------------------------------------------ ingestion

    def ingest(self, data: bytes, name: str = "document.pdf",
               progress: ProgressFn | None = None) -> IngestReport:
        cfg = self.config
        t0 = time.perf_counter()
        self.close()  # one document per session; drop any previous one first

        if progress:
            progress("reading PDF", 0, 1)
        doc = load_pdf(data, name, strip_headers=cfg.strip_headers)
        if progress:
            progress("reading PDF", 1, 1)

        n_sentences = 0
        if cfg.chunking == "semantic":
            sentences = document_sentences(doc.pages)
            n_sentences = len(sentences)
            chunks = semantic_chunks(
                sentences, self.embedder, buffer=cfg.sentence_buffer,
                percentile=cfg.breakpoint_percentile, min_chars=cfg.min_chunk_chars,
                max_chars=cfg.max_chunk_chars, batch_size=cfg.embed_batch_size, progress=progress,
            )
            del sentences
        elif cfg.chunking == "fixed":
            chunks = fixed_chunks(doc.pages, cfg.fixed_chunk_chars, cfg.fixed_overlap_chars)
        else:
            raise ValueError(f"unknown chunking mode {cfg.chunking!r}")

        if not chunks:
            raise ValueError("No extractable text found. If this is a scanned PDF, OCR it first.")

        vectors = embed_in_batches(self.embedder, [c.text for c in chunks], cfg.embed_batch_size,
                                   progress, "indexing")
        index = self._index_factory()
        index.add(
            ids=[c.id for c in chunks], embeddings=vectors, documents=[c.text for c in chunks],
            metadatas=[{"page_start": c.page_start, "page_end": c.page_end} for c in chunks],
        )
        self._index = index
        self._chunks = {c.id: c for c in chunks}
        self.document = {"name": doc.name, "sha256": doc.sha256, "pages": doc.page_count}

        report = IngestReport(
            document=doc.name, pages=doc.page_count, empty_pages=doc.empty_pages,
            sentences=n_sentences, chunks=len(chunks),
            avg_chunk_chars=round(float(np.mean([len(c.text) for c in chunks])), 1),
            chunker=cfg.chunking, seconds=round(time.perf_counter() - t0, 2),
            stripped_lines=doc.stripped_lines,
        )
        del doc, vectors  # page text is no longer needed; chunks hold what we use
        return report

    # ------------------------------------------------------------ retrieval

    def retrieve(self, question: str) -> Retrieval:
        if not self.ready:
            raise RuntimeError("No document loaded. Call ingest() first.")
        cfg = self.config
        t0 = time.perf_counter()
        q = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        hits = self._index.query(q, cfg.vector_top_k)                    # stage 1
        t1 = time.perf_counter()
        if self.reranker is not None and cfg.use_reranker and hits:
            probs = self.reranker.score(question, [self._chunks[cid].text for cid, _ in hits])  # stage 2
            ranked = sorted(((cid, sim, float(p)) for (cid, sim), p in zip(hits, probs)),
                            key=lambda r: (-r[2], -r[1]))
        else:
            ranked = [(cid, sim, None) for cid, sim in hits]
        t2 = time.perf_counter()
        return Retrieval(question, ranked, {"vector_search": round((t1 - t0) * 1000, 1),
                                            "rerank": round((t2 - t1) * 1000, 1)})

    # ------------------------------------------------------------ answering

    def _source_refs(self, rows: list[tuple[str, float, float | None]]) -> list[SourceRef]:
        refs = []
        for i, (cid, vsim, rprob) in enumerate(rows, start=1):
            c = self._chunks[cid]
            refs.append(SourceRef(f"S{i}", cid, c.page_start, c.page_end, round(vsim, 4),
                                  None if rprob is None else round(rprob, 4), c.text))
        return refs

    def ask(self, question: str) -> Answer:
        if self.generator is None:
            raise RuntimeError("No generator configured.")
        cfg = self.config
        question = question.strip()
        retrieval = self.retrieve(question)
        timings = dict(retrieval.timings_ms)
        top = retrieval.ranked[: cfg.rerank_top_n]
        doc_name = (self.document or {}).get("name", "the document")

        def finish(status, text, reason, verification, sources, checks, raw=None) -> Answer:
            record = ProvenanceRecord(
                id=ProvenanceLog.new_id(), timestamp=ProvenanceLog.now(), document=dict(self.document or {}),
                question=question, status=status, reason=reason, verification=verification, answer=text,
                raw_model_output=raw, candidates=candidate_rows(self._chunks, retrieval.ranked),
                sources=[s.public() for s in sources], sentences=[c.__dict__ for c in checks],
                thresholds={"min_vector_sim": cfg.min_vector_sim, "min_rerank_prob": cfg.min_rerank_prob,
                            "support_threshold": cfg.support_threshold, "gate": cfg.use_gate,
                            "validation": cfg.validate_answers},
                models=self.model_info, timings_ms=timings,
            )
            self.provenance.add(record)
            return Answer(question, status, text, reason, verification, sources, checks, record.id)

        # ---- confidence gate: refuse before the LLM ever sees weak evidence
        if cfg.use_gate:
            best_v, best_r = retrieval.best_vector_sim, retrieval.best_rerank_prob
            low_vector = best_v < cfg.min_vector_sim
            low_rerank = best_r is not None and best_r < cfg.min_rerank_prob
            if not top or low_vector or low_rerank:
                detail = f"best vector similarity {best_v:.2f}" + (
                    f", best rerank relevance {best_r:.2f}" if best_r is not None else "")
                return finish(
                    "refused",
                    f"I couldn't find this in {doc_name}. The closest passages weren't a confident "
                    f"match ({detail}), so I won't guess.",
                    "low_confidence", None, self._source_refs(top[:3]), [],
                )
            if best_r is not None:
                top = [row for row in top if row[2] is not None and row[2] >= cfg.min_rerank_prob]

        sources = self._source_refs(top)
        t0 = time.perf_counter()
        raw = self.generator(question, format_sources(sources)).strip()
        timings["generate"] = round((time.perf_counter() - t0) * 1000, 1)

        bare = raw.strip(" .\"'`*").upper()
        if bare.startswith(NOT_FOUND_TOKEN) or bare == NOT_FOUND_TOKEN:
            return finish("refused", f"The most relevant passages in {doc_name} don't answer this question.",
                          "model_declined", None, sources, [], raw)

        if not cfg.validate_answers:
            return finish("answered", raw, None, "unchecked", sources, [], raw)

        t0 = time.perf_counter()
        checks = validate_answer(raw, sources, self.embedder, cfg.support_threshold)
        timings["validate"] = round((time.perf_counter() - t0) * 1000, 1)
        supported = sum(c.supported for c in checks)
        if not checks or supported == 0 or (cfg.strict_validation and supported < len(checks)):
            return finish(
                "refused",
                "I drafted an answer but couldn't verify it against the document's text, so I'm "
                "withholding it. The closest passages are listed below.",
                "failed_source_validation", None, sources, checks, raw,
            )
        verification = "verified" if supported == len(checks) else "partial"
        return finish("answered", raw, None, verification, sources, checks, raw)
