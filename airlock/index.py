"""ChromaDB vector index that lives only in RAM (EphemeralClient, no persist dir)."""

from __future__ import annotations

import uuid
from typing import Protocol, Sequence

import numpy as np


class VectorIndex(Protocol):
    def add(self, ids: list[str], embeddings: np.ndarray, documents: list[str],
            metadatas: list[dict]) -> None: ...
    def query(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def close(self) -> None: ...


def _cosine(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(rows, axis=1)
    norms[norms == 0] = 1.0
    return (rows @ q) / norms


class ChromaIndex:
    """One ephemeral Chroma collection per document session.

    * `chromadb.EphemeralClient` keeps everything in process memory; nothing is
      written to disk, so closing the session (or the process) erases the index.
    * Telemetry is disabled in settings; the air-gap guard would block it anyway.
    * Similarity scores are recomputed as exact cosine similarity from the
      returned vectors, so they mean the same thing whatever distance space the
      installed Chroma version defaults to.
    """

    def __init__(self, name_prefix: str = "airlock") -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False, allow_reset=False)
        )
        self.name = f"{name_prefix}-{uuid.uuid4().hex[:16]}"
        self._col = self._client.create_collection(name=self.name, metadata={"hnsw:space": "cosine"})
        try:
            self._max_batch = int(self._client.get_max_batch_size())
        except Exception:
            self._max_batch = 1000

    def add(self, ids, embeddings, documents, metadatas) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        step = max(1, min(self._max_batch, 1000))
        for s in range(0, len(ids), step):
            self._col.add(
                ids=list(ids[s:s + step]),
                embeddings=emb[s:s + step].tolist(),
                documents=list(documents[s:s + step]),
                metadatas=list(metadatas[s:s + step]),
            )

    def query(self, embedding, k: int) -> list[tuple[str, float]]:
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(k, n),
            include=["embeddings"],
        )
        ids: Sequence[str] = res["ids"][0]
        rows = np.asarray(res["embeddings"][0], dtype=np.float32)
        sims = _cosine(np.asarray(embedding, dtype=np.float32), rows)
        order = np.argsort(-sims)
        return [(ids[i], float(sims[i])) for i in order]

    def count(self) -> int:
        return int(self._col.count())

    def close(self) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:
            pass
        self._col = None
