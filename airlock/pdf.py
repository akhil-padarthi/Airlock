"""In-memory PDF text extraction. The PDF is read from bytes and never touches disk."""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Page:
    number: int   # 1-based, as a human would cite it
    text: str


@dataclass
class LoadedDocument:
    name: str
    sha256: str
    pages: list[Page]
    empty_pages: list[int]
    stripped_lines: list[str]

    @property
    def page_count(self) -> int:
        return len(self.pages)


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_DIGITS = re.compile(r"\d+")


def _normalise(text: str) -> str:
    text = text.replace("\r", "\n").replace("­", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)            # "termi-\nnation" -> "termination"
    text = re.sub(r"[ \t ]+", " ", text)
    return text


def _line_key(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())


def _strip_repeating_lines(raw_pages: list[str], min_share: float = 0.5) -> tuple[list[str], list[str]]:
    """Remove running headers/footers: lines (digits ignored) that appear on most pages."""
    if len(raw_pages) < 4:
        return raw_pages, []
    counts: Counter[str] = Counter()
    for text in raw_pages:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        edge = lines[:3] + lines[-3:]                     # headers/footers live at the edges
        counts.update({_line_key(ln) for ln in edge})
    threshold = max(3, int(len(raw_pages) * min_share))
    repeating = {k for k, c in counts.items() if c >= threshold and k}
    if not repeating:
        return raw_pages, []
    cleaned = []
    for text in raw_pages:
        cleaned.append("\n".join(ln for ln in text.split("\n") if _line_key(ln) not in repeating))
    return cleaned, sorted(repeating)


def load_pdf(data: bytes, name: str = "document.pdf", strip_headers: bool = True) -> LoadedDocument:
    from pypdf import PdfReader  # local import keeps module import cheap

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{name} doesn't look like a readable PDF ({e})") from e
    if reader.is_encrypted:
        # Lots of PDFs are "encrypted" with an empty password just to block editing.
        try:
            opened = reader.decrypt("")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{name} is encrypted and couldn't be opened ({e})") from e
        if not opened:
            raise ValueError(f"{name} is password-protected. Remove the password and try again.")
    raw = []
    for page in reader.pages:
        try:
            raw.append(_normalise(page.extract_text() or ""))
        except Exception:  # a single broken page shouldn't sink a 1,000-page document
            raw.append("")
    stripped: list[str] = []
    if strip_headers:
        raw, stripped = _strip_repeating_lines(raw)
    pages = [Page(number=i + 1, text=t.strip()) for i, t in enumerate(raw)]
    empty = [p.number for p in pages if len(p.text) < 20]
    return LoadedDocument(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        pages=pages,
        empty_pages=empty,
        stripped_lines=stripped,
    )
