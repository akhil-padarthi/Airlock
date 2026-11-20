"""Runtime configuration. Every field can be overridden with an AIRLOCK_<FIELD> env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = Path(os.environ.get("AIRLOCK_MODELS_DIR", PROJECT_ROOT / "models"))

# Some embedding models are trained with task prefixes; using them noticeably
# improves retrieval. (document_prefix, query_prefix)
EMBED_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed2": ("", "query: "),
}


@dataclass(frozen=True)
class AirlockConfig:
    # --- local model endpoints (loopback only; the air-gap blocks anything else) ---
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"
    num_ctx: int = 8192
    reranker_hf_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_path: str = str(DEFAULT_MODELS_DIR / "ms-marco-MiniLM-L6-v2")

    # --- chunking ---
    chunking: str = "semantic"            # "semantic" | "fixed"
    sentence_buffer: int = 1              # neighbours on each side when embedding a sentence
    breakpoint_percentile: float = 90.0   # split where topic shift is in the top 10%
    min_chunk_chars: int = 300
    max_chunk_chars: int = 1500
    fixed_chunk_chars: int = 1000         # baseline chunker only
    fixed_overlap_chars: int = 150
    strip_headers: bool = True            # drop running headers/footers repeated on most pages

    # --- retrieval ---
    vector_top_k: int = 30                # stage 1: candidates from ChromaDB
    rerank_top_n: int = 5                 # stage 2: kept after cross-encoder
    use_reranker: bool = True

    # --- confidence gate + source validation ---
    use_gate: bool = True
    min_rerank_prob: float = 0.30         # best cross-encoder relevance needed to answer
    min_vector_sim: float = 0.35          # best cosine similarity needed to answer
    validate_answers: bool = True
    support_threshold: float = 0.60       # answer sentence <-> source sentence similarity
    strict_validation: bool = False       # True: refuse if *any* sentence is unsupported

    embed_batch_size: int = 64

    @classmethod
    def from_env(cls, **overrides) -> "AirlockConfig":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"AIRLOCK_{f.name.upper()}")
            if raw is None:
                continue
            if f.type in ("bool", bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                values[f.name] = int(raw)
            elif f.type in ("float", float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)

    def with_(self, **changes) -> "AirlockConfig":
        return replace(self, **changes)

    @property
    def embed_prefixes(self) -> tuple[str, str]:
        base = self.embed_model.split(":")[0]
        return EMBED_PREFIXES.get(base, ("", ""))

    @classmethod
    def baseline(cls, base: "AirlockConfig | None" = None) -> "AirlockConfig":
        """Naive RAG for comparison: fixed-size chunks, vector search only, no gate, no validation."""
        base = base or cls()
        return replace(base, chunking="fixed", use_reranker=False, use_gate=False,
                       validate_answers=False)
