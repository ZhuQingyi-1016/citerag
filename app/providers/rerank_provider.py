from __future__ import annotations

import os
from typing import Any

import requests


class BaseRerankProvider:
    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class NoopRerankProvider(BaseRerankProvider):
    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        return hits[:top_n]


class CohereRerankProvider(BaseRerankProvider):
    def __init__(self) -> None:
        self.api_key = os.getenv("COHERE_API_KEY")
        self.model = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
        self.base_url = os.getenv("COHERE_BASE_URL", "https://api.cohere.com/v1/rerank")

        if not self.api_key:
            raise ValueError("Missing COHERE_API_KEY for rerank provider")

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []

        documents = [h["text"] for h in hits]

        resp = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        reranked_hits: list[dict[str, Any]] = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item.get("relevance_score")
            hit = dict(hits[idx])
            if score is not None:
                hit["rerank_score"] = float(score)
            reranked_hits.append(hit)

        return reranked_hits