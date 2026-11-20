"""Sentence splitting, semantic chunking and the fixed-size baseline chunker.

Every chunk keeps the page range it came from, so any answer built on it can be
cited back to the exact pages of the original PDF.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .pdf import Page

ProgressFn = Callable[[str, int, int], None]


class Embedder(Protocol):
    """LangChain `Embeddings` interface (duck-typed so tests can use stand-ins)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class Sentence:
    text: str
    page: int


@dataclass
class Chunk:
    id: str
    index: int
    text: str
    page_start: int
    page_end: int
    meta: dict = field(default_factory=dict)

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    def overlaps(self, pages: Sequence[int]) -> bool:
        return any(self.page_start <= p <= self.page_end for p in pages)


# ---------------------------------------------------------------- sentences

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp",
    "no", "nos", "sec", "secs", "art", "fig", "figs", "p", "pp", "para", "u.s", "u.k", "st",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx",
    "ch", "vol", "ed", "cf", "al", "jr", "sr", "dept", "est", "min", "max",
}
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z0-9§(\"'\[])")
_HEADING_MAX_CHARS = 70


def _is_heading(line: str, next_line: str | None) -> bool:
    s = line.strip()
    if not s or not next_line or len(s) > _HEADING_MAX_CHARS or len(s.split()) > 10:
        return False
    if s[-1] in ".,;:!?-":
        return False
    return next_line.strip()[:1].isupper() or next_line.strip()[:1].isdigit()


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    for sep in ("; ", ", "):
        if sep not in sentence:
            continue
        out, cur = [], ""
        for part in sentence.split(sep):
            candidate = f"{cur}{sep}{part}" if cur else part
            if len(candidate) > limit and cur:
                out.append(cur + sep.strip())
                cur = part
            else:
                cur = candidate
        if cur:
            out.append(cur)
        result: list[str] = []
        for piece in out:
            result.extend(_hard_wrap(piece, limit) if len(piece) > limit else [piece])
        return result
    return _hard_wrap(sentence, limit)


def _hard_wrap(text: str, limit: int) -> list[str]:
    out = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def split_sentences(text: str, max_chars: int = 500) -> list[str]:
    """Split page text into sentences; headings become their own 'sentence'."""
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        segments, current = [], []
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if _is_heading(line, nxt):
                if current:
                    segments.append(" ".join(current))
                    current = []
                segments.append(line.strip())
            else:
                current.append(line.strip())
        if current:
            segments.append(" ".join(current))
        for seg in segments:
            start = 0
            for m in _BOUNDARY.finditer(seg):
                before = seg[start:m.start()].split()
                last = before[-1].lower().rstrip(".") if before else ""
                if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                    continue
                piece = seg[start:m.end()].strip()
                if piece:
                    sentences.extend(_split_long(piece, max_chars))
                start = m.end()
            tail = seg[start:].strip()
            if tail:
                sentences.extend(_split_long(tail, max_chars))
    return [s for s in sentences if len(s) > 1]


def document_sentences(pages: Sequence[Page]) -> list[Sentence]:
    out: list[Sentence] = []
    for page in pages:
        out.extend(Sentence(text=s, page=page.number) for s in split_sentences(page.text))
    return out


# ---------------------------------------------------------------- embeddings helpers

def embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int = 64,
                     progress: ProgressFn | None = None, stage: str = "embedding") -> np.ndarray:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        vectors.extend(embedder.embed_documents(texts[start:start + batch_size]))
        if progress:
            progress(stage, min(start + batch_size, total), total)
    return normalise_rows(np.asarray(vectors, dtype=np.float32)) if vectors else np.zeros((0, 0), np.float32)


def normalise_rows(m: np.ndarray) -> np.ndarray:
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ---------------------------------------------------------------- semantic chunking

def _join(group: list[Sentence]) -> str:
    """Join sentences; keep headings (no closing punctuation) on their own line so
    they can be recognised again when a chunk is re-split for answer validation."""
    out = ""
    for s in group:
        if out:
            out += "\n" if out[-1] not in ".!?:;\"')]" else " "
        out += s.text
    return out


def semantic_chunks(sentences: list[Sentence], embedder: Embedder, *, buffer: int = 1,
                    percentile: float = 90.0, min_chars: int = 300, max_chars: int = 1500,
                    batch_size: int = 64, progress: ProgressFn | None = None) -> list[Chunk]:
    """Split where meaning shifts, not every N characters.

    1. Embed each sentence together with `buffer` neighbours on each side
       (smooths out noise from very short sentences).
    2. Measure cosine distance between consecutive windows.
    3. Cut where the distance is in the top (100 - percentile)% for this
       document - i.e. where the topic changes most sharply.
    4. Enforce min/max chunk sizes so chunks stay useful for the LLM.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        s = sentences[0]
        return [Chunk("c00000", 0, s.text, s.page, s.page, {"chunker": "semantic"})]

    windows = [
        " ".join(s.text for s in sentences[max(0, i - buffer): i + buffer + 1])
        for i in range(n)
    ]
    emb = embed_in_batches(embedder, windows, batch_size, progress, "semantic chunking")
    distances = 1.0 - np.einsum("ij,ij->i", emb[:-1], emb[1:])
    threshold = float(np.percentile(distances, percentile))

    groups: list[list[Sentence]] = []
    current: list[Sentence] = [sentences[0]]
    current_len = len(sentences[0].text)
    for i in range(1, n):
        s = sentences[i]
        topic_shift = distances[i - 1] > threshold
        too_long = current_len + 1 + len(s.text) > max_chars
        if (topic_shift and current_len >= min_chars) or too_long:
            groups.append(current)
            current, current_len = [], 0
        current.append(s)
        current_len += len(s.text) + 1
    groups.append(current)

    # fold a runt final chunk into its predecessor
    if len(groups) > 1 and sum(len(s.text) for s in groups[-1]) < min_chars // 2:
        tail = groups.pop()
        groups[-1].extend(tail)

    chunks = []
    for idx, group in enumerate(groups):
        chunks.append(Chunk(
            id=f"c{idx:05d}", index=idx, text=_join(group),
            page_start=group[0].page, page_end=group[-1].page,
            meta={"chunker": "semantic", "sentences": len(group)},
        ))
    return chunks


# ---------------------------------------------------------------- fixed-size baseline

def fixed_chunks(pages: Sequence[Page], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Classic character-window chunking (the 'before' in the evaluation)."""
    texts, starts, pos = [], [], 0
    for page in pages:
        starts.append(pos)
        flat = re.sub(r"\s+", " ", page.text).strip()
        texts.append(flat)
        pos += len(flat) + 1
    full = " ".join(texts)
    numbers = [p.number for p in pages]

    def page_at(offset: int) -> int:
        return numbers[max(0, bisect.bisect_right(starts, offset) - 1)] if numbers else 1

    chunks, start, idx = [], 0, 0
    while start < len(full):
        end = min(len(full), start + size)
        if end < len(full):
            cut = full.rfind(" ", start + size // 2, end)
            end = cut if cut != -1 else end
        text = full[start:end].strip()
        if text:
            chunks.append(Chunk(
                id=f"c{idx:05d}", index=idx, text=text,
                page_start=page_at(start), page_end=page_at(max(start, end - 1)),
                meta={"chunker": "fixed"},
            ))
            idx += 1
        if end >= len(full):
            break
        start = max(end - overlap, start + 1)
    return chunks
