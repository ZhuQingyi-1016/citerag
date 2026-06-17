from __future__ import annotations

import re
from typing import Any

from app.providers.rerank_provider import CohereRerankProvider


def _normalize_for_diversity(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text)
    return [t for t in tokens if t.strip()]


def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(_normalize_for_diversity(a))
    sb = set(_normalize_for_diversity(b))

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def rrf_fuse(
    bm25_hits: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    fused: dict[tuple[str, int], dict[str, Any]] = {}

    def add_hits(hits: list[dict[str, Any]], source: str) -> None:
        for rank, hit in enumerate(hits, start=1):
            key = (hit["file_id"], hit["chunk_id"])
            if key not in fused:
                fused[key] = {
                    **hit,
                    "hybrid_score": 0.0,
                    "sources": [],
                }
            fused[key]["hybrid_score"] += 1.0 / (k + rank)
            if source not in fused[key]["sources"]:
                fused[key]["sources"].append(source)

    add_hits(bm25_hits, "bm25")
    add_hits(vector_hits, "vector")

    results = list(fused.values())
    results.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    return results[:top_k]


class RetrievalService:
    def __init__(
        self,
        index_service,
        embedding_provider,
        vector_repo,
        sqlite_repo,
        rerank_provider=None,
    ):
        self.index_service = index_service
        self.embedding_provider = embedding_provider
        self.vector_repo = vector_repo
        self.sqlite_repo = sqlite_repo
        self.rerank_provider = rerank_provider

    def build_query_variants(self, question: str, question_type: str = "general") -> list[str]:
        variants = [question]
        if question_type == "process":
            variants.append(
                question + " 发展 历程 演进 阶段 早期 后来 近年来"
            )
            variants.append(
                question + " development history evolution early later recent"
            )
        return _dedupe_keep_order(variants)

    def _merge_hits_by_score(
        self,
        hit_lists: list[list[dict[str, Any]]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for hits in hit_lists:
            for hit in hits:
                key = (hit["file_id"], hit["chunk_id"])
                curr_score = float(hit.get("score", hit.get("hybrid_score", 0.0)))

                if key not in merged:
                    merged[key] = hit
                    continue

                prev_score = float(
                    merged[key].get("score", merged[key].get("hybrid_score", 0.0))
                )
                if curr_score > prev_score:
                    merged[key] = hit

        results = list(merged.values())
        results.sort(
            key=lambda x: float(x.get("score", x.get("hybrid_score", 0.0))),
            reverse=True,
        )
        return results[:top_k]

    def _apply_process_bonus(
        self,
        hits: list[dict[str, Any]],
        question_type: str,
    ) -> list[dict[str, Any]]:
        if question_type != "process":
            return hits

        timeline_terms = [
            "早期",
            "后来",
            "随后",
            "近年来",
            "最初",
            "进一步",
            "发展",
            "演进",
            "阶段",
            "early",
            "later",
            "recent",
            "initially",
            "subsequently",
            "development",
            "evolution",
            "stage",
        ]
        boosted: list[dict[str, Any]] = []
        for hit in hits:
            text = hit.get("text", "").lower()
            bonus = 0.15 if any(term in text for term in timeline_terms) else 0.0
            base_score = float(hit.get("score", hit.get("hybrid_score", 0.0)))
            boosted.append({**hit, "score": base_score + bonus})

        boosted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return boosted

    def index_file_embeddings(self, file_id: str) -> None:
        bm25_index = self.index_service.index_repo.get_index(file_id)
        if bm25_index is None:
            raise ValueError("file not indexed for BM25 yet")

        chunks = bm25_index.chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)

        vector_chunks = []
        for chunk, emb in zip(chunks, embeddings):
            vector_chunks.append(
                {
                    "file_id": bm25_index.file_id,
                    "filename": bm25_index.filename,
                    "chunk_id": chunk.chunk_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "embedding": emb,
                }
            )

        self.vector_repo.upsert_chunks(vector_chunks)

        self.sqlite_repo.upsert_index_status(
            file_id=file_id,
            bm25_indexed=True,
            vector_indexed=True,
            indexed_at=bm25_index.indexed_at,
        )

    def search_bm25(
        self,
        question: str,
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if file_id is not None:
            return self.index_service.search_in_file(file_id, question, top_k=top_k)
        return self.index_service.search_all_files(question, top_k=top_k)

    def search_vector(
        self,
        question: str,
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedding_provider.embed_query(question)
        return self.vector_repo.search(query_vec, top_k=top_k, file_id=file_id)

    def search_hybrid(
        self,
        question: str,
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bm25_hits = self.search_bm25(question, top_k=top_k * 2, file_id=file_id)
        vector_hits = self.search_vector(question, top_k=top_k * 2, file_id=file_id)
        return rrf_fuse(bm25_hits, vector_hits, top_k=top_k)
    
    def search(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        file_id: str | None = None,
        question_type: str = "general",
    ) -> list[dict]:
        if top_k <= 0:
            return []

        retrieval_mode = retrieval_mode.lower().strip()
        variants = self.build_query_variants(
            question=question,
            question_type=question_type,
        )

        if retrieval_mode == "bm25":
            hit_lists = [
                self.search_bm25(q, top_k=top_k, file_id=file_id)
                for q in variants
            ]
            merged = self._merge_hits_by_score(hit_lists, top_k=top_k)
            return self._apply_process_bonus(merged, question_type=question_type)

        if retrieval_mode == "vector":
            hit_lists = [
                self.search_vector(q, top_k=top_k, file_id=file_id)
                for q in variants
            ]
            merged = self._merge_hits_by_score(hit_lists, top_k=top_k)
            return self._apply_process_bonus(merged, question_type=question_type)

        if retrieval_mode == "hybrid":
            hit_lists = [
                self.search_hybrid(q, top_k=top_k, file_id=file_id)
                for q in variants
            ]
            merged = self._merge_hits_by_score(hit_lists, top_k=top_k)
            return self._apply_process_bonus(merged, question_type=question_type)

        raise ValueError(f"unsupported retrieval_mode: {retrieval_mode}")
        

    def rebuild_vectors_from_sqlite(self) -> dict:
        rows = self.sqlite_repo.list_indexed_files()

        rebuilt = []
        skipped = []

        for row in rows:
            file_id = row["file_id"]

            if not row["vector_indexed"]:
                skipped.append(
                    {
                        "file_id": file_id,
                        "reason": "vector_indexed is false in sqlite",
                    }
                )
                continue

            try:
                self.index_file_embeddings(file_id)
                rebuilt.append(file_id)
            except Exception as e:
                skipped.append(
                    {
                        "file_id": file_id,
                        "reason": str(e),
                    }
                )

        return {
            "rebuilt_vector_files": rebuilt,
            "skipped": skipped,
        }
    
    def rerank_hits(
        self,
        question: str,
        hits: list[dict],
        final_top_k: int,
        rerank_mode: str = "none",
    ) -> list[dict]:
        if final_top_k <= 0 or not hits:
            return []

        rerank_mode = rerank_mode.lower().strip()

        if rerank_mode == "none":
            return self._select_diverse_hits(hits, final_top_k)

        if rerank_mode == "cohere":
            try:
                provider = self.rerank_provider or CohereRerankProvider()
                reranked = provider.rerank(
                    query=question,
                    hits=hits,
                    top_n=len(hits),
                )
                return self._select_diverse_hits(
                    reranked,
                    final_top_k,
                    similarity_threshold=0.75,
                )
            except Exception as e:
                # Graceful fallback to baseline top_k when external rerank fails.
                print(f"[rerank_fallback] mode=cohere error={e}")
                return self._select_diverse_hits(hits, final_top_k)

        return self._select_diverse_hits(hits, final_top_k)

    def _select_diverse_hits(
        self,
        hits: list[dict],
        final_top_k: int,
        similarity_threshold: float = 0.8,
    ) -> list[dict]:
        if final_top_k <= 0 or not hits:
            return []

        selected: list[dict] = []
        selected_keys: set[tuple[str | None, int | None]] = set()

        for hit in hits:
            hit_text = hit.get("text", "").strip()
            key = (hit.get("file_id"), hit.get("chunk_id"))

            if not hit_text or key in selected_keys:
                continue

            too_similar = any(
                _jaccard_similarity(hit_text, chosen.get("text", ""))
                >= similarity_threshold
                for chosen in selected
            )
            if too_similar:
                continue

            selected.append(hit)
            selected_keys.add(key)
            if len(selected) >= final_top_k:
                return selected

        # Backfill to keep output size stable if diversity filtering is strict.
        for hit in hits:
            key = (hit.get("file_id"), hit.get("chunk_id"))
            if key in selected_keys:
                continue
            selected.append(hit)
            selected_keys.add(key)
            if len(selected) >= final_top_k:
                break

        return selected[:final_top_k]
