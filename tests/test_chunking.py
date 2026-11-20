from airlock.chunking import Sentence, fixed_chunks, semantic_chunks, split_sentences
from airlock.pdf import Page, _strip_repeating_lines
from fakes import HashEmbedder


def test_sentence_splitter_respects_abbreviations_and_decimals():
    text = "See Sec. 4 for details. The fee is $5.00 per unit, e.g. for Inc. members. Availability is 99.95% monthly."
    assert split_sentences(text) == [
        "See Sec. 4 for details.",
        "The fee is $5.00 per unit, e.g. for Inc. members.",
        "Availability is 99.95% monthly.",
    ]


def test_heading_becomes_its_own_sentence():
    text = "Section 44.14 Termination for Convenience\nEither party may terminate this Agreement.\nIt ends then."
    assert split_sentences(text)[0] == "Section 44.14 Termination for Convenience"


def test_long_run_on_text_is_split():
    long = "; ".join(["clause about deliverables and acceptance criteria"] * 40) + "."
    parts = split_sentences(long, max_chars=200)
    assert len(parts) > 5 and all(len(p) <= 200 for p in parts)


def test_semantic_chunker_cuts_at_topic_shift():
    a = [Sentence(f"Invoices are payable net forty five days with late charges on overdue invoices item {i}.", 1)
         for i in range(6)]
    b = [Sentence(f"Volcanic basalt cools into hexagonal columns along the rugged coastline formation {i}.", 2)
         for i in range(6)]
    chunks = semantic_chunks(a + b, HashEmbedder(), buffer=0, percentile=90, min_chars=50, max_chars=5000)
    assert len(chunks) == 2
    assert "Invoices" in chunks[0].text and "basalt" not in chunks[0].text
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)


def test_semantic_chunker_enforces_max_size_and_keeps_every_sentence():
    sents = [Sentence(f"Sentence number {i} about the same topic of reporting obligations.", 1 + i // 10)
             for i in range(60)]
    chunks = semantic_chunks(sents, HashEmbedder(), min_chars=100, max_chars=400)
    assert all(len(c.text) <= 400 for c in chunks)
    assert " ".join(c.text for c in chunks).replace("\n", " ") == " ".join(s.text for s in sents)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 6


def test_fixed_chunker_tracks_pages():
    pages = [Page(1, "alpha " * 200), Page(2, "beta " * 200), Page(3, "gamma " * 200)]
    chunks = fixed_chunks(pages, size=500, overlap=50)
    first_beta = next(c for c in chunks if "beta" in c.text)
    assert first_beta.page_end == 2
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3


def test_running_headers_and_page_numbers_are_stripped():
    topics = ["payment", "audit", "insurance", "governing law", "term", "notices", "liability", "data",
              "personnel", "training"]
    raw = [f"ACME Corp - Confidential\nThis page covers {t} obligations.\nPage {i} of 10"
           for i, t in enumerate(topics, start=1)]
    cleaned, stripped = _strip_repeating_lines(raw)
    assert all("Confidential" not in t and "of 10" not in t for t in cleaned)
    assert all("This page covers" in t for t in cleaned)
    assert len(stripped) == 2
