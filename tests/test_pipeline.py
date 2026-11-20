import json

import pytest

from airlock import AirlockConfig
from fakes import LyingGenerator, NumpyIndex, UncitedGenerator, fake_session


@pytest.fixture()
def session(small_sample):
    data, _ = small_sample
    s = fake_session()
    s.ingest(data, "helios_msa.pdf")
    yield s
    s.close()


def _norm(t):
    return t.lower().replace(",", "").replace("$", "")


def test_ingest_report(small_sample):
    data, _ = small_sample
    with fake_session() as s:
        report = s.ingest(data, "helios_msa.pdf")
        assert report.pages == 120
        assert report.chunker == "semantic"
        assert report.chunks > 50 and report.empty_pages == []
        assert any("confidential" in line for line in report.stripped_lines)
        assert s.document["pages"] == 120 and len(s.document["sha256"]) == 64


# The stand-in models are lexical (no real semantics), so a few questions that need
# paraphrase understanding ("uptime" vs "available", "law governs" vs "governed by the
# laws") are out of their reach. Quality with the real models is measured by `airlock eval`.
STANDIN_MISSES_ALLOWED = 3


def test_answerable_questions_are_answered_verified_and_cited_to_the_right_page(session, small_sample):
    _, questions = small_sample
    answerable = [q for q in questions if q["answerable"]]
    correct = 0
    for q in answerable:
        ans = session.ask(q["question"])
        if not ans.answered:
            continue
        assert ans.verification == "verified"
        cited = {c for s in ans.sentences for c in s.citations}
        on_page = any(s.page_start <= q["pages"][0] <= s.page_end for s in ans.sources if s.label in cited)
        has_value = any(_norm(a) in _norm(ans.text) for a in q["answer_contains"])
        correct += on_page and has_value
    assert correct >= len(answerable) - STANDIN_MISSES_ALLOWED, f"{correct}/{len(answerable)}"


def test_confidence_gate_refuses_before_the_llm_sees_weak_evidence(small_sample):
    data, questions = small_sample
    with fake_session() as s:
        s.ingest(data, "helios_msa.pdf")
        gated = 0
        for q in (q for q in questions if not q["answerable"]):
            calls = s.generator.calls
            ans = s.ask(q["question"])
            if ans.reason == "low_confidence":
                gated += 1
                assert s.generator.calls == calls          # the LLM was never invoked
                assert "couldn't find" in ans.text
                assert ans.sources                          # closest passages are still shown
                record = s.provenance.get(ans.record_id)
                assert record.raw_model_output is None
        assert gated >= 7, gated


def test_model_saying_not_in_document_is_a_refusal(small_sample):
    data, _ = small_sample
    with fake_session(generator=lambda q, src: "NOT_IN_DOCUMENT.") as s:
        s.ingest(data, "helios_msa.pdf")
        ans = s.ask("How much notice is required to terminate the agreement for convenience?")
        assert not ans.answered and ans.reason == "model_declined"


def test_threshold_is_respected_exactly(small_sample):
    data, _ = small_sample
    strict = AirlockConfig(min_rerank_prob=1.01, min_vector_sim=0.0, min_chunk_chars=200, max_chunk_chars=1200)
    with fake_session(strict) as s:
        s.ingest(data, "x.pdf")
        assert s.ask("How much notice is required to terminate the agreement for convenience?").reason == "low_confidence"
    with fake_session(strict.with_(use_gate=False)) as s:
        s.ingest(data, "x.pdf")
        assert s.ask("How much notice is required to terminate the agreement for convenience?").answered


def test_wrong_number_is_caught_by_source_validation(small_sample):
    data, _ = small_sample
    with fake_session(generator=LyingGenerator()) as s:
        s.ingest(data, "helios_msa.pdf")
        ans = s.ask("How much notice is required to terminate the agreement for convenience?")
        assert not ans.answered and ans.reason == "failed_source_validation"
        assert ans.sentences[0].numbers_missing == ["60"]
        record = s.provenance.get(ans.record_id)
        assert "(60)" in record.raw_model_output  # the hallucinated draft is kept for audit, not shown


def test_uncited_answer_from_model_memory_is_withheld(small_sample):
    data, _ = small_sample
    with fake_session(generator=UncitedGenerator()) as s:
        s.ingest(data, "helios_msa.pdf")
        ans = s.ask("How much notice is required to terminate the agreement for convenience?")
        assert not ans.answered and ans.reason == "failed_source_validation"
        assert "no valid citation" in ans.sentences[0].reasons


def test_reranker_improves_ranking_over_vector_only(small_sample):
    data, questions = small_sample
    answerable = [q for q in questions if q["answerable"]]

    def first_hit_ranks(cfg):
        with fake_session(cfg) as s:
            s.ingest(data, "x.pdf")
            chunks = {c.id: c for c in s.chunks}
            ranks = []
            for q in answerable:
                ranked = s.retrieve(q["question"]).ranked
                ranks.append(next((i for i, (cid, _, _) in enumerate(ranked) if chunks[cid].overlaps(q["pages"])), 99))
            return ranks

    base = AirlockConfig(min_rerank_prob=0.5, min_vector_sim=0.05, min_chunk_chars=200, max_chunk_chars=1200)
    with_rr = first_hit_ranks(base)
    without = first_hit_ranks(base.with_(use_reranker=False))
    assert sum(r == 0 for r in with_rr) >= sum(r == 0 for r in without)
    assert sum(with_rr) <= sum(without)


def test_provenance_log_is_complete_and_exportable(session):
    a1 = session.ask("Which state's law governs the agreement?")
    a2 = session.ask("What is the annual salary of the Provider's chief executive officer?")
    assert len(session.provenance) == 2
    lines = [json.loads(line) for line in session.provenance.to_jsonl().splitlines()]
    assert [r["id"] for r in lines] == [a1.record_id, a2.record_id]
    r = lines[0]
    assert r["status"] == "answered" and r["verification"] == "verified"
    assert r["document"]["name"] == "helios_msa.pdf"
    assert r["candidates"] and {"vector_sim", "rerank_prob", "pages"} <= set(r["candidates"][0])
    assert r["sources"][0]["pages"].startswith("p")
    assert r["thresholds"]["min_rerank_prob"] == 0.5
    assert r["sentences"][0]["supported"] is True
    assert lines[1]["status"] == "refused" and lines[1]["reason"] == "low_confidence"


def test_close_purges_everything(small_sample):
    data, _ = small_sample
    s = fake_session()
    s.ingest(data, "x.pdf")
    s.ask("Which state's law governs the agreement?")
    index = s._index
    assert isinstance(index, NumpyIndex) and index.count() > 0
    s.close()
    assert index.closed and index.count() == 0
    assert not s.chunks and len(s.provenance) == 0 and s.document is None and not s.ready
    with pytest.raises(RuntimeError):
        s.ask("anything")


def test_reingest_replaces_previous_document(small_sample):
    data, _ = small_sample
    from airlock.sample import build_sample_pdf

    other, _ = build_sample_pdf(pages=30, seed=1)
    with fake_session() as s:
        s.ingest(data, "a.pdf")
        first_index = s._index
        s.ingest(other, "b.pdf")
        assert first_index.closed and s.document["name"] == "b.pdf" and s.document["pages"] == 30
