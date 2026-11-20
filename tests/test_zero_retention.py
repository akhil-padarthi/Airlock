"""Zero retention: ingesting a PDF and answering questions must not write a single byte to disk.

Uses a CPython audit hook to observe every file open/create/rename/delete in a fresh process.
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ingest_and_ask_write_nothing_to_disk():
    code = textwrap.dedent(f"""
        import json, os, sys
        sys.dont_write_bytecode = True
        sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]
        from airlock.sample import build_sample_pdf
        from airlock.pdf import load_pdf
        from fakes import fake_session
        data, questions = build_sample_pdf(pages=60)
        load_pdf(data)   # warm up lazy imports so only pipeline work is observed

        WRITE = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        events = []
        def hook(event, args):
            if event == "open":
                path, mode, flags = (list(args) + [None, None])[:3]
                if (isinstance(mode, str) and any(c in mode for c in "wax+")) or \\
                   (isinstance(flags, int) and flags & WRITE):
                    events.append([event, str(path), str(mode)])
            elif event in ("os.rename", "os.remove", "os.mkdir", "os.truncate", "shutil.copyfile",
                           "tempfile.mkstemp", "tempfile.mkdtemp", "sqlite3.connect"):
                events.append([event, repr(args)[:120]])
        sys.addaudithook(hook)

        s = fake_session()
        report = s.ingest(data, "secret.pdf")
        answered = sum(s.ask(q["question"]).answered for q in questions)
        s.provenance.to_jsonl()
        s.close()
        print(json.dumps({{"events": events, "chunks": report.chunks, "answered": answered}}))
    """)
    proc = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["chunks"] > 0 and out["answered"] > 0
    assert out["events"] == [], out["events"]


def test_vector_store_is_ephemeral_by_construction():
    src = (ROOT / "airlock" / "index.py").read_text()
    code_only = re.sub(r'""".*?"""', "", src, flags=re.S)
    assert "chromadb.EphemeralClient(" in code_only
    for persistent in ("PersistentClient", "persist_directory", "HttpClient"):
        assert persistent not in code_only
    for module in (ROOT / "airlock").glob("*.py"):
        text = module.read_text()
        assert "PersistentClient" not in text and "persist_directory" not in text, module.name
