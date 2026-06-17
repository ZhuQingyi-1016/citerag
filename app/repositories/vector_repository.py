from __future__ import annotations

import math
from threading import RLock
from typing import Any


class InMemoryVectorRepository:
    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._lock = RLock()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        file_ids = {chunk["file_id"] for chunk in chunks}
        if len(file_ids) != 1:
            raise ValueError("upsert_chunks expects chunks from exactly one file_id")

        file_id = next(iter(file_ids))
        with self._lock:
            self._chunks = [c for c in self._chunks if c["file_id"] != file_id]
            self._chunks.extend(chunks)

    def clear_file(self, file_id: str) -> None:
        with self._lock:
            self._chunks = [c for c in self._chunks if c["file_id"] != file_id]

    def clear_all(self) -> None:
        with self._lock:
            self._chunks.clear()

    def has_any_vectors(self) -> bool:
        with self._lock:
            return bool(self._chunks)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not query_vec:
            return []

        def cosine(a: list[float], b: list[float]) -> float:
            if len(a) != len(b):
                return 0.0

            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        with self._lock:
            if file_id is not None:
                candidates = [c.copy() for c in self._chunks if c["file_id"] == file_id]
            else:
                candidates = [c.copy() for c in self._chunks]

        scored = []
        for c in candidates:
            embedding = c.get("embedding", [])
            score = cosine(query_vec, embedding)
            if score <= 0:
                continue

            scored.append(
                {
                    "file_id": c["file_id"],
                    "filename": c["filename"],
                    "chunk_id": c["chunk_id"],
                    "score": float(score),
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["text"],
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]