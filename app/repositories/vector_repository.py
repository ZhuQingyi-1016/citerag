from __future__ import annotations

import math
from typing import Any


class InMemoryVectorRepository:
    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        file_id = chunks[0]["file_id"]
        self.clear_file(file_id)
        self._chunks.extend(chunks)

    def clear_file(self, file_id: str) -> None:
        self._chunks = [c for c in self._chunks if c["file_id"] != file_id]

    def has_any_vectors(self) -> bool:
        return len(self._chunks) > 0

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        candidates = self._chunks
        if file_id is not None:
            candidates = [c for c in candidates if c["file_id"] == file_id]

        scored = []
        for c in candidates:
            score = cosine(query_vec, c["embedding"])
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