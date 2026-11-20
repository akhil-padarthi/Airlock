# Airlock

Chat with huge PDFs without sending them anywhere.

I wanted to be able to ask questions about really long documents (think 1,000+ page contracts and manuals) without uploading them to OpenAI or anyone else. Airlock runs the whole thing on your own machine: the models, the search index, all of it. When you close the session the document is gone. Nothing gets saved to disk.

It's built with Ollama, LangChain, ChromaDB and a small cross-encoder model from sentence-transformers.

## How it works

When you load a PDF:

1. The PDF is read straight from memory, never written to a temp file. Running headers and page numbers get stripped out, since they show up on every page and only add noise.
2. The text is split into sentences, and each sentence is embedded together with its neighbors. A chunk ends wherever the meaning jumps the most, so chunks follow the topics in the document instead of being cut every 1,000 characters.
3. The chunks are embedded with Ollama and stored in a ChromaDB collection that only lives in RAM.

When you ask a question:

1. Vector search grabs the 30 closest chunks.
2. A cross-encoder rereads each of those together with your question and scores how well it actually answers it. The top 5 are kept. This step is slower but a lot more accurate, which is why it only runs on the shortlist.
3. If even the best chunk scores too low, Airlock tells you it couldn't find the answer. The LLM never gets called, so it never gets the chance to make something up.
4. Otherwise the local LLM writes an answer using only those passages, and it has to cite them like [S1], [S2].
5. The answer is then checked sentence by sentence. Each sentence needs a real citation, has to be close in meaning to something in the cited passage, and every number in it has to appear in that passage. The number check matters more than you'd think, because embeddings can't really tell "60 days" from "90 days".
6. All of it (candidates, scores, which sources were used, which sentences passed) goes into a log that you can export if you want a record.

## Keeping it offline

I didn't want to rely on every library respecting an "offline" setting, so Airlock installs a Python audit hook when it starts. The hook blocks every network connection and DNS lookup that isn't going to localhost, for everything running in the process: LangChain, Chroma's telemetry, Hugging Face, Streamlit. It also removes proxy environment variables, because a proxy running on localhost would otherwise be an easy way out.

If you want to lock it down further, setting `AIRLOCK_LOOPBACK_PORTS=11434` only allows the Ollama port.

The one time you need internet is `airlock fetch-models`, which downloads the models. After that it never touches the network.

## Setup

You'll need Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
cd airlock
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e ".[ui,demo]"

airlock fetch-models   # downloads the reranker (~90 MB) and pulls the Ollama models
airlock doctor         # checks everything, every line should say OK
```

If doctor can't reach Ollama, make sure it's running (`ollama serve`).

## Using it

The web UI:

```bash
streamlit run app.py
```

It runs at http://127.0.0.1:8501 and only listens on localhost.

From the terminal:

```bash
airlock ask contract.pdf "What's the termination notice period?"
airlock chat contract.pdf
airlock chat contract.pdf --log-out audit.jsonl   # saves the log when you quit
```

The first time you index something big, it takes a while. A 1,000 page PDF has around 15,000 sentences, and every one of them gets embedded to find the topic breaks. On a laptop CPU that's a few minutes; on a GPU or an M-series Mac it's much faster. If you're in a hurry, `--chunking fixed` skips that step, at some cost to accuracy.

### No big PDF to test with?

```bash
airlock make-sample
```

This generates a fake 1,040 page services agreement with 15 specific facts buried on known pages (the liability cap on one page, the termination notice on another, and so on), along with a question file. 17 of the questions can be answered from the document and 8 can't, so you can watch it answer some and refuse others.

## Measuring it

Rather than putting made-up numbers in a README, there's an eval command. It runs the same questions through Airlock and through a plain RAG setup (fixed-size chunks, vector search only, no confidence check, no answer checking) on the same PDF:

```bash
airlock eval sample_contract.pdf sample_questions.jsonl --out report.md
```

The report includes:

- precision@5, hit rate and MRR
- how many answers weren't actually backed by the document
- how many questions it refused correctly vs. wrongly
- a table to help you pick the confidence threshold

Improvements are shown relative to the plain setup.

The sample document is fine as a sanity check. For numbers that mean something, write 30 or so questions about a real document. The format is one JSON object per line:

```json
{"question": "What's the notice period for termination?", "answerable": true, "pages": [873], "answer_contains": ["90"]}
{"question": "Who is the CEO?", "answerable": false}
```

## Settings

Defaults live in `airlock/config.py`, and you can override any of them with an environment variable, e.g. `AIRLOCK_LLM_MODEL=qwen2.5:7b`. The ones you'll most likely touch:

| Setting | Default | What it does |
|---|---|---|
| `llm_model` | `llama3.1:8b` | Ollama model that writes the answers |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `chunking` | `semantic` | or `fixed` |
| `vector_top_k` | `30` | how many chunks the first search pulls |
| `rerank_top_n` | `5` | how many are kept after reranking |
| `min_rerank_prob` | `0.30` | below this score it refuses instead of answering |
| `min_vector_sim` | `0.35` | same idea, for the vector search score |
| `support_threshold` | `0.60` | how close an answer sentence must be to its source |
| `strict_validation` | `false` | if true, one unsupported sentence means no answer at all |

Treat the thresholds as a starting point. Run the eval on your own documents and use what it recommends.

## Tests

```bash
pip install numpy pypdf reportlab pytest
pytest
```

You don't need the full install for this. There are 34 tests and they take about 30 seconds. They swap the real models for simple stand-ins with the same interfaces, so they run anywhere without a GPU or Ollama. GitHub Actions runs them on Python 3.10 to 3.13 on every push. They cover:

- chunking and page tracking
- the refusal logic and the answer checks
- the network block
- a check that nothing is written to disk during a session
- a full 1,040 page run
- the CLI and the eval

What they can't tell you is how good the answers are with the real models. That's what `airlock eval` is for.

## Things to know

- The network block covers the Airlock process only. Ollama runs as its own process, so keep it on localhost, and block it at the firewall too if you want to be extra careful.
- The block works at the Python level. A compiled library that opened its own sockets without going through Python wouldn't be caught. None of the libraries Airlock uses do that as far as I know, but the firewall is the real backstop.
- Encrypted PDFs with an empty password open fine. Ones that need a real password don't; remove the password first.
- Airlock doesn't write anything to disk, but your OS can still swap memory out. Use disk encryption if that worries you. Also avoid running Ollama with `OLLAMA_DEBUG=1`, since it logs a lot more.
- Scanned PDFs don't have a text layer, so they need OCR first. Airlock warns you if a lot of pages come back empty.
- Tables and charts come through as plain text, so questions about them are hit or miss.
- It handles one document at a time. Loading a new one wipes the previous one.

## TODO

- OCR for scanned PDFs
- More than one document per session
- Better table handling
- Streaming answers in the UI

## Project layout

```
airlock/
  airgap.py       blocks network access
  config.py       settings
  pdf.py          reads the PDF in memory
  chunking.py     sentence splitting and chunking
  index.py        in-memory ChromaDB
  models.py       Ollama + LangChain, and the reranker
  provenance.py   answer checks and the log
  pipeline.py     ties it all together
  evaluate.py     the eval
  sample.py       generates the demo contract
  cli.py          the airlock command
app.py            Streamlit UI
tests/
.github/workflows/tests.yml   runs the tests on GitHub
```
