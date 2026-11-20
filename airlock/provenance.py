"""Source validation of generated answers + an in-memory provenance log.

Every answer is checked sentence-by-sentence against the exact source passages
it cites:
  * each sentence must cite at least one real source ([S1], [S2], ...),
  * it must be semantically close to a sentence in a cited source, and
  * every number in it (amounts, days, percentages) must literally appear in
    the cited sources - embeddings alone can't tell "60 days" from "90 days".

The log is kept in RAM only. It is written out solely when the user explicitly
exports it (download button / --log-out).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .chunking import Chunk, Embedder, normalise_rows, split_sentences

NOT_FOUND_TOKEN = "NOT_IN_DOCUMENT"
_CITE_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_ONLY_CITES = re.compile(r"^(\s*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]\s*)+[.!?]?\s*$", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@dataclass
class SourceRef:
    label: str          # "S1"
    chunk_id: str
    page_start: int
    page_end: int
    vector_sim: float
    rerank_prob: float | None
    text: str

    @property
    def pages_label(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    def public(self, excerpt_chars: int = 280) -> dict:
        d = asdict(self)
        d["pages"] = self.pages_label
        d["excerpt"] = self.text[:excerpt_chars] + ("..." if len(self.text) > excerpt_chars else "")
        del d["text"]
        return d


@dataclass
class SentenceCheck:
    text: str
    citations: list[str]
    invalid_citations: list[str]
    support: float
    numbers_missing: list[str]
    supported: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    status: str                 # "answered" | "refused"
    text: str
    reason: str | None          # why it refused, if it did
    verification: str | None    # "verified" | "partial" | "unchecked"
    sources: list[SourceRef]
    sentences: list[SentenceCheck]
    record_id: str

    @property
    def answered(self) -> bool:
        return self.status == "answered"


def format_sources(sources: list[SourceRef]) -> str:
    return "\n\n".join(f"[{s.label}] ({s.pages_label})\n{s.text}" for s in sources)


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}


def answer_sentences(answer: str) -> list[str]:
    """Split an LLM answer into sentences, re-attaching dangling '[S1].' fragments."""
    raw = split_sentences(answer.replace("\n- ", "\n").replace("\n* ", "\n"), max_chars=2000)
    out: list[str] = []
    for s in raw:
        if _ONLY_CITES.match(s) and out:
            out[-1] = f"{out[-1]} {s.strip()}"
        else:
            out.append(s)
    return out


def parse_citations(sentence: str) -> list[str]:
    found: list[str] = []
    for group in _CITE_GROUP.findall(sentence):
        for label in re.split(r"\s*[,;]\s*", group):
            label = label.strip().upper()
            if label and label not in found:
                found.append(label)
    return found


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+", " ", _CITE_GROUP.sub("", sentence)).strip()


def validate_answer(answer: str, sources: list[SourceRef], embedder: Embedder,
                    support_threshold: float) -> list[SentenceCheck]:
    by_label = {s.label: s for s in sources}
    source_sents = {s.label: split_sentences(s.text) or [s.text] for s in sources}
    flat = [(label, sent) for label, sents in source_sents.items() for sent in sents]
    if not flat:
        return []
    src_emb = normalise_rows(np.asarray(embedder.embed_documents([t for _, t in flat]), dtype=np.float32))
    src_labels = np.array([label for label, _ in flat])

    parsed = []
    for sentence in answer_sentences(answer):
        clean = strip_citations(sentence)
        if len(re.sub(r"\W", "", clean)) >= 3:
            parsed.append((parse_citations(sentence), clean))
    if not parsed:
        return []
    # Sentence-to-sentence similarity is a symmetric task, so both sides are
    # embedded as documents (no query prefix).
    ans_emb = normalise_rows(np.asarray(embedder.embed_documents([c for _, c in parsed]), dtype=np.float32))

    checks: list[SentenceCheck] = []
    for (cites, clean), q in zip(parsed, ans_emb):
        valid = [c for c in cites if c in by_label]
        invalid = [c for c in cites if c not in by_label]
        reasons: list[str] = []

        mask = np.isin(src_labels, valid) if valid else np.ones(len(flat), dtype=bool)
        support = float(np.max(src_emb[mask] @ q)) if mask.any() else 0.0

        cited_text = " ".join(by_label[c].text for c in valid)
        missing = sorted(_numbers(clean) - _numbers(cited_text)) if valid else sorted(_numbers(clean))

        if not valid:
            reasons.append("no valid citation")
        if invalid:
            reasons.append(f"cites non-existent source(s): {', '.join(invalid)}")
        if support < support_threshold:
            reasons.append(f"low similarity to cited source ({support:.2f} < {support_threshold:.2f})")
        if missing:
            reasons.append(f"number(s) not in cited source: {', '.join(missing)}")
        supported = bool(valid) and support >= support_threshold and not missing
        checks.append(SentenceCheck(clean, valid, invalid, round(support, 4), missing, supported, reasons))
    return checks


# ------------------------------------------------------------------ provenance log

@dataclass
class ProvenanceRecord:
    id: str
    timestamp: str
    document: dict
    question: str
    status: str
    reason: str | None
    verification: str | None
    answer: str
    raw_model_output: str | None
    candidates: list[dict]
    sources: list[dict]
    sentences: list[dict]
    thresholds: dict
    models: dict
    timings_ms: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ProvenanceLog:
    """Append-only, RAM-only audit trail for the current session."""

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add(self, record: ProvenanceRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def get(self, record_id: str) -> ProvenanceRecord | None:
        return next((r for r in self._records if r.id == record_id), None)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in self._records) + ("\n" if self._records else "")

    def clear(self) -> None:
        self._records.clear()


def candidate_rows(chunks: dict[str, Chunk], ranked: list[tuple[str, float, float | None]]) -> list[dict]:
    rows = []
    for rank, (cid, vsim, rprob) in enumerate(ranked, start=1):
        c = chunks[cid]
        rows.append({
            "rank": rank, "chunk_id": cid, "pages": c.pages_label,
            "vector_sim": round(vsim, 4), "rerank_prob": None if rprob is None else round(rprob, 4),
        })
    return rows
