"""Measure the claims instead of asserting them.

Runs the same questions through two pipelines over the same PDF:

* baseline - fixed 1,000-char chunks, vector search only, no confidence gate,
             no answer validation (a typical first RAG prototype)
* airlock  - semantic chunks, vector search -> cross-encoder rerank,
             confidence gate, source-validated answers

and reports retrieval precision and the unsupported-answer ("hallucination")
rate for both, plus a threshold calibration sweep.

Question file (JSONL), one object per line:
  {"question": "...", "answerable": true, "pages": [873], "answer_contains": ["90", "ninety"]}
  {"question": "...", "answerable": false}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import AirlockConfig
from .pipeline import Airlock
from .provenance import parse_citations


@dataclass
class EvalQuestion:
    question: str
    answerable: bool = True
    pages: list[int] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)


def load_questions(source: str | Path | Iterable[dict]) -> list[EvalQuestion]:
    if isinstance(source, (str, Path)):
        rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    else:
        rows = list(source)
    out = []
    for r in rows:
        answerable = bool(r.get("answerable", True))
        pages = [int(p) for p in r.get("pages", [])]
        if answerable and not pages:
            raise ValueError(f"answerable question needs gold 'pages': {r.get('question')!r}")
        out.append(EvalQuestion(r["question"], answerable, pages, list(r.get("answer_contains", []))))
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", "").replace("$", "")).strip()


# ------------------------------------------------------------------ retrieval

def retrieval_metrics(session: Airlock, questions: list[EvalQuestion], k: int = 5) -> dict:
    chunks = {c.id: c for c in session.chunks}
    precisions, hits_any, hit1, rr, rows = [], [], [], [], []
    for q in (q for q in questions if q.answerable):
        ranked = session.retrieve(q.question).ranked[:k]
        hits = [chunks[cid].overlaps(q.pages) for cid, _, _ in ranked]
        first = next((i for i, h in enumerate(hits) if h), None)
        precisions.append(sum(hits) / k)
        hits_any.append(first is not None)
        hit1.append(bool(hits and hits[0]))
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        rows.append({"question": q.question, "gold_pages": q.pages, "first_hit_rank": None if first is None else first + 1,
                     "top_pages": [chunks[cid].pages_label for cid, _, _ in ranked]})
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {"k": k, "precision_at_k": mean(precisions), "hit_rate_at_k": mean(hits_any),
            "hit_at_1": mean(hit1), "mrr": mean(rr), "per_question": rows}


def calibration(session: Airlock, questions: list[EvalQuestion]) -> dict:
    """Sweep the confidence threshold: how many answerable questions pass vs.
    how many unanswerable ones get refused."""
    use_rerank = session.reranker is not None and session.config.use_reranker
    signal = "rerank_prob" if use_rerank else "vector_sim"
    scored = []
    for q in questions:
        r = session.retrieve(q.question)
        best = r.best_rerank_prob if use_rerank else r.best_vector_sim
        scored.append((q.answerable, float(best or 0.0)))
    pos = [s for a, s in scored if a]
    neg = [s for a, s in scored if not a]
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tpr = float(np.mean([s >= t for s in pos])) if pos else 0.0
        tnr = float(np.mean([s < t for s in neg])) if neg else 0.0
        rows.append({"threshold": float(t), "answerable_pass": tpr, "unanswerable_refused": tnr,
                     "balanced": (tpr + tnr) / 2 if pos and neg else (tpr or tnr)})
    best = max(rows, key=lambda r: (round(r["balanced"], 6), -r["threshold"]))
    return {"signal": signal, "rows": rows, "recommended": best["threshold"],
            "scores": {"answerable": pos, "unanswerable": neg}}


# ------------------------------------------------------------------ answers

def answer_metrics(session: Airlock, questions: list[EvalQuestion]) -> dict:
    counts = {"correct": 0, "unsupported": 0, "false_refusal": 0, "correct_refusal": 0}
    rows = []
    for q in questions:
        ans = session.ask(q.question)
        if not ans.answered:
            outcome = "false_refusal" if q.answerable else "correct_refusal"
        elif not q.answerable:
            outcome = "unsupported"
        else:
            by_label = {s.label: s for s in ans.sources}
            cited = [by_label[c] for c in parse_citations(ans.text) if c in by_label]
            grounded = any(any(s.page_start <= p <= s.page_end for p in q.pages) for s in cited)
            content_ok = not q.answer_contains or any(_norm(a) in _norm(ans.text) for a in q.answer_contains)
            outcome = "correct" if grounded and content_ok else "unsupported"
        counts[outcome] += 1
        rows.append({"question": q.question, "answerable": q.answerable, "outcome": outcome,
                     "status": ans.status, "reason": ans.reason, "verification": ans.verification,
                     "answer": ans.text})
    n = max(1, len(questions))
    rates = {f"{k}_rate": v / n for k, v in counts.items()}
    return {"n": len(questions), **counts, **rates, "per_question": rows}


# ------------------------------------------------------------------ orchestration

SessionFactory = Callable[[AirlockConfig], Airlock]


def _relative(new: float, old: float) -> float | None:
    return None if old == 0 else (new - old) / old


def run_evaluation(pdf: bytes, name: str, questions: list[EvalQuestion], make_session: SessionFactory,
                   config: AirlockConfig, *, k: int = 5, with_answers: bool = True,
                   log: Callable[[str], None] = lambda _m: None) -> dict:
    results: dict = {"document": name, "questions": len(questions),
                     "answerable": sum(q.answerable for q in questions), "k": k}
    for label, cfg in (("baseline", AirlockConfig.baseline(config)), ("airlock", config)):
        log(f"[{label}] indexing {name} ({cfg.chunking} chunks)...")
        with make_session(cfg) as session:
            report = session.ingest(pdf, name)
            part = {"config": {"chunking": cfg.chunking, "reranker": cfg.use_reranker, "gate": cfg.use_gate,
                               "validation": cfg.validate_answers},
                    "ingest": {"chunks": report.chunks, "avg_chunk_chars": report.avg_chunk_chars,
                               "seconds": report.seconds}}
            log(f"[{label}] retrieval metrics...")
            part["retrieval"] = retrieval_metrics(session, questions, k)
            if label == "airlock":
                log("[airlock] calibrating confidence threshold...")
                part["calibration"] = calibration(session, questions)
            if with_answers:
                log(f"[{label}] answering {len(questions)} questions...")
                part["answers"] = answer_metrics(session, questions)
            results[label] = part

    b, a = results["baseline"], results["airlock"]
    summary = {"precision_change": _relative(a["retrieval"]["precision_at_k"], b["retrieval"]["precision_at_k"])}
    if with_answers:
        summary["unsupported_change"] = _relative(a["answers"]["unsupported_rate"], b["answers"]["unsupported_rate"])
    results["summary"] = summary
    return results


def _pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.1f}%"


def render_report(results: dict) -> str:
    b, a, s, k = results["baseline"], results["airlock"], results["summary"], results["k"]
    lines = [
        f"# Airlock evaluation - {results['document']}",
        "",
        f"{results['questions']} questions ({results['answerable']} answerable, "
        f"{results['questions'] - results['answerable']} not answerable from the document).",
        "",
        "| Metric | Baseline | Airlock | Relative change |",
        "|---|---|---|---|",
        f"| Precision@{k} | {_pct(b['retrieval']['precision_at_k'])} | {_pct(a['retrieval']['precision_at_k'])} | {_pct(s['precision_change'], True)} |",
        f"| Hit rate@{k} | {_pct(b['retrieval']['hit_rate_at_k'])} | {_pct(a['retrieval']['hit_rate_at_k'])} | |",
        f"| Hit@1 | {_pct(b['retrieval']['hit_at_1'])} | {_pct(a['retrieval']['hit_at_1'])} | |",
        f"| MRR | {b['retrieval']['mrr']:.3f} | {a['retrieval']['mrr']:.3f} | |",
    ]
    if "answers" in a:
        ba, aa = b["answers"], a["answers"]
        lines += [
            f"| Unsupported answers | {ba['unsupported']} ({_pct(ba['unsupported_rate'])}) | "
            f"{aa['unsupported']} ({_pct(aa['unsupported_rate'])}) | {_pct(s['unsupported_change'], True)} |",
            f"| Correct, grounded answers | {ba['correct']} | {aa['correct']} | |",
            f"| Correct refusals (unanswerable) | {ba['correct_refusal']} | {aa['correct_refusal']} | |",
            f"| False refusals (answerable) | {ba['false_refusal']} | {aa['false_refusal']} | |",
        ]
    cal = a.get("calibration")
    if cal:
        lines += ["", f"Recommended confidence threshold ({cal['signal']}): **{cal['recommended']:.2f}** "
                      f"(maximises balanced accuracy between answering and refusing).", "",
                  "| Threshold | Answerable passed | Unanswerable refused |", "|---|---|---|"]
        lines += [f"| {r['threshold']:.2f} | {_pct(r['answerable_pass'])} | {_pct(r['unanswerable_refused'])} |"
                  for r in cal["rows"] if round(r["threshold"] * 100) % 10 == 0]
    lines += [
        "", "## Definitions", "",
        f"- **Baseline**: {b['ingest']['chunks']} fixed 1,000-char chunks, top-{k} by vector similarity, "
        "no confidence gate, no answer validation.",
        f"- **Airlock**: {a['ingest']['chunks']} semantic chunks, top-{k} after cross-encoder reranking of the "
        "vector-search shortlist, confidence gate, source-validated answers.",
        f"- **Precision@{k}**: share of the top-{k} retrieved chunks that overlap a gold page.",
        "- **Unsupported answer**: the system answered, but the question is not answerable from the document, "
        "or none of the cited passages is on a gold page, or the expected value is missing from the answer.",
        "- Relative change = (Airlock - Baseline) / Baseline.",
    ]
    return "\n".join(lines) + "\n"
