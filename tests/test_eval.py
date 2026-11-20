from airlock import Airlock, AirlockConfig
from airlock.evaluate import load_questions, render_report, run_evaluation
from fakes import ExtractiveGenerator, HashEmbedder, LexicalReranker, NumpyIndex


def test_evaluation_harness_end_to_end(small_sample):
    data, rows = small_sample
    questions = load_questions(rows)
    embedder, reranker = HashEmbedder(), LexicalReranker()
    eager = ExtractiveGenerator(min_coverage=0.0)  # a model that always answers, like many real ones

    def make_session(cfg):
        return Airlock(embedder, eager, reranker if cfg.use_reranker else None, NumpyIndex, cfg)

    cfg = AirlockConfig(min_rerank_prob=0.5, min_vector_sim=0.05, min_chunk_chars=200, max_chunk_chars=1200)
    results = run_evaluation(data, "helios.pdf", questions, make_session, cfg, k=5)

    b, a = results["baseline"], results["airlock"]
    assert b["config"]["chunking"] == "fixed" and a["config"]["chunking"] == "semantic"
    assert a["retrieval"]["mrr"] >= b["retrieval"]["mrr"]
    # the baseline answers every unanswerable question; Airlock's gate refuses (almost) all of them
    assert b["answers"]["correct_refusal"] == 0 and b["answers"]["unsupported"] >= 8
    assert a["answers"]["correct_refusal"] >= 7
    assert a["answers"]["unsupported"] < b["answers"]["unsupported"]
    assert results["summary"]["unsupported_change"] < -0.5
    assert 0.05 <= a["calibration"]["recommended"] <= 0.95

    report = render_report(results)
    assert "| Precision@5 |" in report and "Unsupported answers" in report and "Recommended confidence" in report
