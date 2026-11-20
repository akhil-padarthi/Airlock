"""`airlock` command line.

  airlock fetch-models                 one-time download (the only step that uses the internet)
  airlock doctor                       check every component + prove the air-gap is live
  airlock make-sample                  write a 1,040-page demo contract + gold questions
  airlock ask FILE.pdf "question"      one question
  airlock chat FILE.pdf                interactive session
  airlock eval FILE.pdf questions.jsonl  baseline vs. Airlock, measured
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import airgap
from .config import AirlockConfig


# ------------------------------------------------------------------ helpers

def _config(args) -> AirlockConfig:
    overrides = {}
    if getattr(args, "llm", None):
        overrides["llm_model"] = args.llm
    if getattr(args, "embed", None):
        overrides["embed_model"] = args.embed
    if getattr(args, "no_rerank", False):
        overrides["use_reranker"] = False
    if getattr(args, "min_rerank", None) is not None:
        overrides["min_rerank_prob"] = args.min_rerank
    if getattr(args, "min_sim", None) is not None:
        overrides["min_vector_sim"] = args.min_sim
    if getattr(args, "chunking", None):
        overrides["chunking"] = args.chunking
    return AirlockConfig.from_env(**overrides)


class _Progress:
    def __init__(self) -> None:
        self.last = 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and now - self.last < 0.25:
            return
        self.last = now
        pct = 100 * done / max(1, total)
        end = "\n" if done >= total else ""
        print(f"\r  {stage:<18} {done:>6}/{total:<6} {pct:5.1f}%", end=end, file=sys.stderr, flush=True)


def _open_session(args):
    from .pipeline import Airlock

    cfg = _config(args)
    pdf = _pdf_path(args.pdf)
    print(f"Air-gap: {'ON' if airgap.self_test() else 'NOT VERIFIED'} | LLM {cfg.llm_model} | "
          f"embeddings {cfg.embed_model} | reranker {'on' if cfg.use_reranker else 'off'}", file=sys.stderr)
    session = Airlock.local(cfg)
    report = session.ingest(pdf.read_bytes(), pdf.name, progress=_Progress())
    print(f"Indexed {report.pages} pages -> {report.chunks} {report.chunker} chunks "
          f"in {report.seconds:.1f}s (in memory only)", file=sys.stderr)
    for w in report.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    return session


def _pdf_path(path: str) -> Path:
    pdf = Path(path)
    if not pdf.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return pdf


def _print_answer(ans, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps({
            "question": ans.question, "status": ans.status, "reason": ans.reason,
            "verification": ans.verification, "answer": ans.text,
            "sources": [s.public() for s in ans.sources],
            "sentences": [c.__dict__ for c in ans.sentences], "record_id": ans.record_id,
        }, indent=2, ensure_ascii=False))
        return
    tag = {"verified": "VERIFIED", "partial": "PARTIALLY VERIFIED", "unchecked": "UNCHECKED"}.get(
        ans.verification or "", "NOT ANSWERED")
    print(f"\n[{tag}] {ans.text}\n")
    for c in ans.sentences:
        if not c.supported:
            print(f"  ! unverified: \"{c.text[:90]}\" - {'; '.join(c.reasons)}")
    if ans.sources:
        print("  Sources:" if ans.answered else "  Closest passages:")
        for s in ans.sources:
            rr = f" rerank {s.rerank_prob:.2f}" if s.rerank_prob is not None else ""
            print(f"   [{s.label}] {s.pages_label:<12} cosine {s.vector_sim:.2f}{rr}")
    print()


def _export_log(session, path: str | None) -> None:
    if path and len(session.provenance):
        Path(path).write_text(session.provenance.to_jsonl())
        print(f"Provenance log written to {path} (you asked for it; nothing else was saved).", file=sys.stderr)


# ------------------------------------------------------------------ commands

def cmd_ask(args) -> int:
    airgap.install()
    with _open_session(args) as session:
        _print_answer(session.ask(args.question), args.json)
        _export_log(session, args.log_out)
    return 0


def cmd_chat(args) -> int:
    airgap.install()
    with _open_session(args) as session:
        print("Ask anything about the document. Commands: :log (show provenance), :quit", file=sys.stderr)
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q in (":q", ":quit", ":exit"):
                break
            if q == ":log":
                print(session.provenance.to_jsonl() or "(empty)")
                continue
            _print_answer(session.ask(q))
        _export_log(session, args.log_out)
    print("Session closed: index, chunks and log purged from memory.", file=sys.stderr)
    return 0


def cmd_eval(args) -> int:
    airgap.install()
    from .evaluate import load_questions, render_report, run_evaluation
    from .models import build_embedder, build_generator, build_index, build_reranker
    from .pipeline import Airlock

    cfg = _config(args)
    pdf = _pdf_path(args.pdf)
    embedder = build_embedder(cfg)
    generator = build_generator(cfg) if not args.no_answers else None
    reranker = build_reranker(cfg) if cfg.use_reranker else None
    info = {"embeddings": f"ollama/{cfg.embed_model}", "llm": f"ollama/{cfg.llm_model}",
            "reranker": cfg.reranker_hf_id}

    def make_session(c: AirlockConfig) -> Airlock:
        return Airlock(embedder, generator, reranker if c.use_reranker else None, build_index, c, info)

    results = run_evaluation(pdf.read_bytes(), pdf.name, load_questions(args.questions), make_session, cfg,
                             k=args.k, with_answers=not args.no_answers,
                             log=lambda m: print(m, file=sys.stderr))
    report = render_report(results)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))
    return 0


def cmd_make_sample(args) -> int:
    airgap.install()
    from .sample import build_sample_pdf

    data, questions = build_sample_pdf(args.pages)
    Path(args.out).write_bytes(data)
    Path(args.questions).write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    print(f"Wrote {args.out} ({args.pages} pages) and {args.questions} ({len(questions)} questions).")
    return 0


def cmd_fetch_models(args) -> int:
    """The one step that needs the internet. Run it before going offline."""
    cfg = _config(args)
    target = Path(cfg.reranker_path)
    if target.exists() and any(target.iterdir()):
        print(f"Reranker already present at {target}")
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {cfg.reranker_hf_id} -> {target}")
        target.mkdir(parents=True, exist_ok=True)
        # PyTorch weights + tokenizer only (the repo also hosts ONNX/OpenVINO variants we don't need)
        snapshot_download(repo_id=cfg.reranker_hf_id, local_dir=str(target),
                          allow_patterns=["*.json", "*.txt", "model.safetensors"])
        if not (target / "model.safetensors").exists():
            snapshot_download(repo_id=cfg.reranker_hf_id, local_dir=str(target),
                              allow_patterns=["pytorch_model.bin"])
    if shutil.which("ollama"):
        for model in (cfg.embed_model, cfg.llm_model):
            print(f"ollama pull {model}")
            subprocess.run(["ollama", "pull", model], check=False)
    else:
        print("Ollama CLI not found. Install it from https://ollama.com, then run:\n"
              f"  ollama pull {cfg.embed_model}\n  ollama pull {cfg.llm_model}")
    print("Done. From here on Airlock never needs the network.")
    return 0


def cmd_doctor(args) -> int:
    import importlib
    import urllib.request

    cfg = _config(args)
    ok = True

    def line(passed: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

    print("Airlock doctor")
    for mod in ("pypdf", "numpy", "langchain_core", "langchain_ollama", "chromadb", "sentence_transformers"):
        try:
            importlib.import_module(mod)
            line(True, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            line(False, f"import {mod}", str(e).splitlines()[0])

    airgap.install()
    line(airgap.self_test(), "air-gap guard blocks outbound connections")
    host = cfg.ollama_url.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    line(airgap.is_local(host), "Ollama endpoint is on this machine", cfg.ollama_url)

    try:
        with urllib.request.urlopen(f"{cfg.ollama_url}/api/tags", timeout=3) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
        line(True, "Ollama server reachable")
        for model in (cfg.embed_model, cfg.llm_model):
            present = model in names or f"{model}:latest" in names
            line(present, f"Ollama model {model}", "" if present else f"run: ollama pull {model}")
    except Exception as e:  # noqa: BLE001
        line(False, "Ollama server reachable", f"{e} (start it with `ollama serve`)")

    rr = Path(cfg.reranker_path)
    line(rr.exists() and any(rr.iterdir()), "cross-encoder reranker on disk", str(rr))

    try:
        import numpy as np

        from .index import ChromaIndex

        idx = ChromaIndex()
        idx.add(["a"], np.ones((1, 4), np.float32), ["probe"], [{"page_start": 1, "page_end": 1}])
        hit = idx.query(np.ones(4, np.float32), 1)
        idx.close()
        line(bool(hit) and abs(hit[0][1] - 1.0) < 1e-4, "ChromaDB in-memory index")
    except Exception as e:  # noqa: BLE001
        line(False, "ChromaDB in-memory index", str(e).splitlines()[0])

    blocked = airgap.status()["blocked_attempts"]
    if blocked:
        print(f"  note: blocked {len(blocked)} outbound attempt(s) by libraries during checks: "
              + ", ".join(sorted({b['host'] for b in blocked})))
    print("All good." if ok else "Some checks failed - see above.")
    return 0 if ok else 1


# ------------------------------------------------------------------ entry point

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="airlock", description="Air-gapped Q&A over massive PDFs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_opts(sp):
        sp.add_argument("--llm", help="Ollama chat model (default llama3.1:8b)")
        sp.add_argument("--embed", help="Ollama embedding model (default nomic-embed-text)")
        sp.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder stage")
        sp.add_argument("--min-rerank", type=float, help="confidence gate: min rerank relevance (0-1)")
        sp.add_argument("--min-sim", type=float, help="confidence gate: min vector cosine similarity")
        sp.add_argument("--chunking", choices=["semantic", "fixed"])

    sp = sub.add_parser("ask", help="ask one question")
    sp.add_argument("pdf"), sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--log-out", help="export the provenance log (JSONL) to this file")
    model_opts(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("chat", help="interactive session")
    sp.add_argument("pdf")
    sp.add_argument("--log-out", help="export the provenance log (JSONL) on exit")
    model_opts(sp)
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("eval", help="measure baseline vs. Airlock")
    sp.add_argument("pdf"), sp.add_argument("questions")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--no-answers", action="store_true", help="retrieval metrics only (no LLM)")
    sp.add_argument("--out", help="write the markdown report here")
    sp.add_argument("--json-out", help="write full results JSON here")
    model_opts(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("make-sample", help="generate the 1,040-page demo contract")
    sp.add_argument("--pages", type=int, default=1040)
    sp.add_argument("--out", default="sample_contract.pdf")
    sp.add_argument("--questions", default="sample_questions.jsonl")
    sp.set_defaults(func=cmd_make_sample)

    sp = sub.add_parser("fetch-models", help="one-time model download (needs internet)")
    model_opts(sp)
    sp.set_defaults(func=cmd_fetch_models)

    sp = sub.add_parser("doctor", help="check the installation and the air-gap")
    model_opts(sp)
    sp.set_defaults(func=cmd_doctor)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        if os.environ.get("AIRLOCK_DEBUG"):
            raise
        print(f"error: {e or e.__class__.__name__}", file=sys.stderr)
        if isinstance(e, airgap.AirgapViolation):
            print("Something tried to reach the internet and the air-gap stopped it.", file=sys.stderr)
        elif any(s in repr(e).lower() for s in ("connecterror", "connection refused", "connect call failed")):
            print("Is Ollama running? Start it with `ollama serve`, then check with `airlock doctor`.",
                  file=sys.stderr)
        print("(set AIRLOCK_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
