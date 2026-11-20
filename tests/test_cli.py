"""CLI plumbing, with the model builders swapped for stand-ins (each run in a fresh process,
because the CLI air-gaps the interpreter it runs in)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCH = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]
import airlock.models as m
import fakes
m.build_embedder = lambda cfg: fakes.HashEmbedder()
m.build_generator = lambda cfg: fakes.ExtractiveGenerator()
m.build_reranker = lambda cfg: fakes.LexicalReranker()
m.build_index = fakes.NumpyIndex
from airlock.cli import main
"""


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    code = PATCH + f"\nraise SystemExit(main({args!r}))\n"
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, capture_output=True,
                          text=True, timeout=300)


def test_make_sample_ask_and_eval(tmp_path):
    r = run_cli(["make-sample", "--pages", "80", "--out", "c.pdf", "--questions", "q.jsonl"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "c.pdf").stat().st_size > 10_000
    assert len((tmp_path / "q.jsonl").read_text().splitlines()) == 25

    r = run_cli(["ask", "c.pdf", "How much notice is required to terminate the agreement for convenience?",
                 "--json", "--min-rerank", "0.5", "--min-sim", "0.05", "--log-out", "prov.jsonl"], tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["status"] == "answered" and out["verification"] == "verified" and "(90)" in out["answer"]
    assert "Air-gap: ON" in r.stderr and "in memory only" in r.stderr
    log = [json.loads(x) for x in (tmp_path / "prov.jsonl").read_text().splitlines()]
    assert log[0]["id"] == out["record_id"]

    r = run_cli(["ask", "c.pdf", "On which stock exchange is the Provider listed?",
                 "--min-rerank", "0.5", "--min-sim", "0.05"], tmp_path)
    assert r.returncode == 0 and "[NOT ANSWERED]" in r.stdout and "Closest passages" in r.stdout

    r = run_cli(["eval", "c.pdf", "q.jsonl", "--min-rerank", "0.5", "--min-sim", "0.05",
                 "--out", "report.md", "--json-out", "results.json"], tmp_path)
    assert r.returncode == 0, r.stderr
    report = (tmp_path / "report.md").read_text()
    assert "| Precision@5 |" in report and "Unsupported answers" in report
    assert json.loads((tmp_path / "results.json").read_text())["summary"]["precision_change"] is not None


def test_friendly_errors_instead_of_tracebacks(tmp_path):
    r = run_cli(["ask", "missing.pdf", "anything"], tmp_path)
    assert r.returncode == 1 and "error: no such file: missing.pdf" in r.stderr
    assert "Traceback" not in r.stderr

    (tmp_path / "fake.pdf").write_text("not really a pdf")
    r = run_cli(["ask", "fake.pdf", "anything"], tmp_path)
    assert r.returncode == 1 and "doesn't look like a readable PDF" in r.stderr
    assert "Traceback" not in r.stderr


def test_chat_session_purges_on_exit(tmp_path):
    assert run_cli(["make-sample", "--pages", "40", "--out", "c.pdf", "--questions", "q.jsonl"], tmp_path).returncode == 0
    code = PATCH + textwrap.dedent("""
        import builtins
        answers = iter(["Which state's law governs the agreement?", ":log", ":quit"])
        builtins.input = lambda prompt="": next(answers)
        raise SystemExit(main(["chat", "c.pdf", "--min-rerank", "0.5", "--min-sim", "0.05"]))
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert '"question": "Which state\'s law governs the agreement?"' in r.stdout   # :log output
    assert "purged from memory" in r.stderr
