"""1,000+ page document, end to end (with stand-in models)."""

import time

import pytest

from fakes import fake_session

pytestmark = pytest.mark.slow


def test_1040_page_contract(full_sample):
    data, questions = full_sample
    with fake_session() as s:
        t0 = time.perf_counter()
        report = s.ingest(data, "helios_msa_1040p.pdf")
        ingest_s = time.perf_counter() - t0
        assert report.pages == 1040 and report.empty_pages == []
        assert report.chunks > 500

        answerable = [q for q in questions if q["answerable"]]
        unanswerable = [q for q in questions if not q["answerable"]]
        correct = 0
        for q in answerable:
            ans = s.ask(q["question"])
            cited = {c for sent in ans.sentences for c in sent.citations}
            on_page = any(src.page_start <= q["pages"][0] <= src.page_end for src in ans.sources if src.label in cited)
            correct += ans.answered and ans.verification == "verified" and on_page
        refused = sum(not s.ask(q["question"]).answered for q in unanswerable)

        print(f"\n1,040 pages -> {report.chunks} chunks in {ingest_s:.1f}s; "
              f"{correct}/{len(answerable)} answered correctly with page citations; "
              f"{refused}/{len(unanswerable)} unanswerable refused")
        assert correct >= len(answerable) - 3     # stand-in models are lexical; see test_pipeline
        assert refused >= len(unanswerable) - 1
