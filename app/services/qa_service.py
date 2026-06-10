from __future__ import annotations

import re

from fastapi import HTTPException
from opentelemetry import trace

from app.core.observability import (
    ASK_COUNT,
    ASK_LATENCY,
    CITATION_COUNT,
    RETRIEVAL_HITS,
    elapsed_ms,
    log_event,
    now_perf,
)
from app.schemas import AskDebugResponse, AskRequest, AskResponse, CitationItem, HitItem
from app.services.index_service import IndexService
from app.services.retrieval_service import RetrievalService

tracer = trace.get_tracer(__name__)


class QAService:
    def __init__(
        self,
        index_service: IndexService,
        retrieval_service: RetrievalService,
        generator,
    ):
        self.index_service = index_service
        self.retrieval_service = retrieval_service
        self.generator = generator

    def _detect_question_type(self, question: str) -> str:
        q = question.lower()
        process_patterns = [
            r"发展过程", r"历程", r"演进", r"历史", r"阶段", r"早期", r"后来", r"近年来",
            r"development process", r"history", r"evolution", r"early", r"later", r"recent",
        ]
        definition_patterns = [r"是什么", r"定义", r"\bwhat is\b", r"\bdefinition\b"]
        reason_patterns = [r"为什么", r"原因", r"机理", r"\bwhy\b", r"\breason\b", r"\bmechanism\b"]
        compare_patterns = [r"区别", r"比较", r"优缺点", r"\bcompare\b", r"\bdifference\b"]

        if any(re.search(p, q) for p in process_patterns):
            return "process"
        if any(re.search(p, q) for p in definition_patterns):
            return "definition"
        if any(re.search(p, q) for p in reason_patterns):
            return "reason"
        if any(re.search(p, q) for p in compare_patterns):
            return "compare"
        return "general"

    def _run_ask(self, req: AskRequest):
        start = now_perf()
        retrieval_mode = req.retrieval_mode

        try:
            with tracer.start_as_current_span("qa.run_ask") as span:
                span.set_attribute("citerag.retrieval_mode", retrieval_mode)
                span.set_attribute("citerag.top_k", req.top_k)
                span.set_attribute("citerag.retrieve_top_k", req.retrieve_top_k)
                span.set_attribute("citerag.has_file_id", req.file_id is not None)
                if req.file_id:
                    span.set_attribute("citerag.file_id", req.file_id)

                if not req.question.strip():
                    ASK_COUNT.labels(status="failed", retrieval_mode=retrieval_mode).inc()
                    raise HTTPException(status_code=400, detail="question cannot be empty")
                question_type = self._detect_question_type(req.question)
                effective_retrieve_top_k = req.retrieve_top_k
                if question_type == "process":
                    effective_retrieve_top_k = max(req.retrieve_top_k, 15)
                span.set_attribute("citerag.question_type", question_type)
                span.set_attribute("citerag.effective_retrieve_top_k", effective_retrieve_top_k)

                retrieval_start = now_perf()
                with tracer.start_as_current_span("qa.retrieval") as retrieval_span:
                    retrieval_span.set_attribute("citerag.retrieval_mode", retrieval_mode)
                    retrieval_span.set_attribute("citerag.top_k", effective_retrieve_top_k)
                    retrieval_span.set_attribute("citerag.question_type", question_type)
                    retrieval_span.set_attribute(
                        "citerag.has_file_id", req.file_id is not None
                    )
                    if req.file_id:
                        retrieval_span.set_attribute("citerag.file_id", req.file_id)

                    if req.file_id is not None:
                        if retrieval_mode == "bm25":
                            if not self.index_service.has_index(req.file_id):
                                ASK_COUNT.labels(
                                    status="failed",
                                    retrieval_mode=retrieval_mode,
                                ).inc()
                                raise HTTPException(
                                    status_code=400,
                                    detail="file not indexed yet, call POST /index/{file_id} first",
                                )
                            hits = self.retrieval_service.search(
                                question=req.question,
                                top_k=effective_retrieve_top_k,
                                retrieval_mode="bm25",
                                file_id=req.file_id,
                                question_type=question_type,
                            )
                        else:
                            hits = self.retrieval_service.search(
                                question=req.question,
                                top_k=effective_retrieve_top_k,
                                retrieval_mode=retrieval_mode,
                                file_id=req.file_id,
                                question_type=question_type,
                            )
                    else:
                        if retrieval_mode == "bm25":
                            if not self.index_service.has_any_index():
                                ASK_COUNT.labels(
                                    status="failed",
                                    retrieval_mode=retrieval_mode,
                                ).inc()
                                raise HTTPException(
                                    status_code=400,
                                    detail=(
                                        "no indexed files available, call POST /index_all "
                                        "or upload with auto_index=true first"
                                    ),
                                )
                            hits = self.retrieval_service.search(
                                question=req.question,
                                top_k=effective_retrieve_top_k,
                                retrieval_mode="bm25",
                                file_id=None,
                                question_type=question_type,
                            )
                        else:
                            hits = self.retrieval_service.search(
                                question=req.question,
                                top_k=effective_retrieve_top_k,
                                retrieval_mode=retrieval_mode,
                                file_id=None,
                                question_type=question_type,
                            )

                    retrieval_ms = elapsed_ms(retrieval_start)
                    initial_hit_chunk_ids = [h["chunk_id"] for h in hits]

                    retrieval_span.set_attribute("citerag.hits_count", len(hits))
                    retrieval_span.set_attribute("citerag.retrieval_ms", retrieval_ms)
                    retrieval_span.set_attribute(
                        "citerag.top_hit_chunk_ids",
                        ",".join(str(cid) for cid in initial_hit_chunk_ids),
                    )

                if not hits:
                    answer = "我在已上传的文档中没有检索到相关内容，因此无法回答。"
                    citations = []
                    hit_items = []

                    span.set_attribute("citerag.hits_count", 0)
                    span.set_attribute("citerag.citations_count", 0)
                    span.set_attribute("citerag.used_chunk_ids_count", 0)
                    span.set_attribute("citerag.retrieval_ms", retrieval_ms)
                    span.set_attribute("citerag.generation_ms", 0.0)
                    span.set_attribute("citerag.rerank_mode", req.rerank_mode)
                    span.set_attribute("citerag.retrieve_top_k", effective_retrieve_top_k)
                    span.set_attribute("citerag.rerank_ms", 0.0)
                    span.set_attribute("citerag.top_hit_chunk_ids", "")
                    span.set_attribute("citerag.cited_chunk_ids", "")
                    span.set_attribute("citerag.prompt_tokens", 0)
                    span.set_attribute("citerag.completion_tokens", 0)
                    span.set_attribute("citerag.total_tokens", 0)
                    span.set_attribute("citerag.estimated_cost_usd", 0.0)

                    duration_ms = elapsed_ms(start)
                    ASK_COUNT.labels(status="ok", retrieval_mode=retrieval_mode).inc()
                    ASK_LATENCY.labels(retrieval_mode=retrieval_mode).observe(
                        duration_ms / 1000
                    )
                    RETRIEVAL_HITS.labels(retrieval_mode=retrieval_mode).observe(0)
                    CITATION_COUNT.labels(retrieval_mode=retrieval_mode).observe(0)

                    log_event(
                        "ask",
                        question=req.question,
                        file_id=req.file_id,
                        retrieval_mode=retrieval_mode,
                        top_k=req.top_k,
                        retrieve_top_k=req.retrieve_top_k,
                        effective_retrieve_top_k=effective_retrieve_top_k,
                        question_type=question_type,
                        rerank_mode=req.rerank_mode,
                        rerank_ms=0.0,
                        hits_count=0,
                        citations_count=0,
                        used_chunk_ids_count=0,
                        top_hit_chunk_ids=[],
                        cited_chunk_ids=[],
                        retrieval_ms=retrieval_ms,
                        generation_ms=0.0,
                        duration_ms=duration_ms,
                        generator_mode=self.generator.__class__.__name__,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=0.0,
                    )

                    return (
                        answer,
                        citations,
                        hit_items,
                        question_type,
                        effective_retrieve_top_k,
                    )

                with tracer.start_as_current_span("qa.rerank") as rerank_span:
                    rerank_span.set_attribute("citerag.rerank_mode", req.rerank_mode)
                    rerank_span.set_attribute("citerag.rerank_input_count", len(hits))
                    rerank_span.set_attribute("citerag.top_k", req.top_k)

                    log_event(
                        "rerank_debug",
                        rerank_mode=req.rerank_mode,
                        provider_class=(
                            self.retrieval_service.rerank_provider.__class__.__name__
                            if self.retrieval_service.rerank_provider is not None
                            else "None"
                        ),
                        rerank_input_count=len(hits),
                    )

                    rerank_start = now_perf()
                    reranked_hits = self.retrieval_service.rerank_hits(
                        question=req.question,
                        hits=hits,
                        final_top_k=req.top_k,
                        rerank_mode=req.rerank_mode,
                    )
                    rerank_ms = elapsed_ms(rerank_start)

                    hits = reranked_hits
                    reranked_hit_chunk_ids = [h["chunk_id"] for h in hits]

                    rerank_span.set_attribute("citerag.rerank_ms", rerank_ms)
                    rerank_span.set_attribute("citerag.rerank_output_count", len(hits))
                    rerank_span.set_attribute(
                        "citerag.reranked_chunk_ids",
                        ",".join(str(cid) for cid in reranked_hit_chunk_ids),
                    )

                generation_start = now_perf()
                with tracer.start_as_current_span("qa.generation") as generation_span:
                    generation_span.set_attribute(
                        "citerag.retrieval_mode", retrieval_mode
                    )
                    generation_span.set_attribute("citerag.hits_count", len(hits))

                    llm_result = self.generator.generate(
                        req.question,
                        hits,
                        question_type=question_type,
                    )
                    usage = llm_result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    estimated_cost_usd = usage.get("estimated_cost_usd", 0.0)

                    used_chunk_ids = llm_result.get("used_chunk_ids", [])
                    used_chunk_refs = llm_result.get("used_chunk_refs", [])
                    if not used_chunk_refs and used_chunk_ids:
                        # Backward-compatible fallback: old generators only return chunk_id.
                        used_chunk_refs = [
                            {
                                "file_id": h["file_id"],
                                "chunk_id": h["chunk_id"],
                            }
                            for h in hits
                            if h["chunk_id"] in used_chunk_ids
                        ]
                    used_ref_set = {
                        (ref["file_id"], int(ref["chunk_id"]))
                        for ref in used_chunk_refs
                        if "file_id" in ref and "chunk_id" in ref
                    }
                    generation_ms = elapsed_ms(generation_start)

                    generation_span.set_attribute(
                        "citerag.used_chunk_ids_count",
                        len(used_chunk_ids),
                    )
                    generation_span.set_attribute(
                        "citerag.used_chunk_refs_count",
                        len(used_ref_set),
                    )
                    generation_span.set_attribute(
                        "citerag.used_chunk_ids",
                        ",".join(str(cid) for cid in used_chunk_ids),
                    )
                    generation_span.set_attribute(
                        "citerag.generation_ms", generation_ms
                    )
                    generation_span.set_attribute("citerag.prompt_tokens", prompt_tokens)
                    generation_span.set_attribute("citerag.completion_tokens", completion_tokens)
                    generation_span.set_attribute("citerag.total_tokens", total_tokens)
                    generation_span.set_attribute("citerag.estimated_cost_usd", estimated_cost_usd)

                display_name_cache: dict[str, str] = {}

                def _display_filename(hit: dict) -> str:
                    file_id = hit.get("file_id")
                    if not file_id:
                        return hit.get("filename", "")
                    if file_id in display_name_cache:
                        return display_name_cache[file_id]
                    sqlite_repo = getattr(self.retrieval_service, "sqlite_repo", None)
                    row = None
                    if sqlite_repo is not None:
                        row = sqlite_repo.get_file(
                            file_id=file_id,
                            include_deleted=True,
                        )
                    display = (
                        row.get("display_name")
                        if row is not None and row.get("display_name")
                        else hit.get("filename", "")
                    )
                    display_name_cache[file_id] = display
                    return display

                hits_with_display = [{**h, "filename": _display_filename(h)} for h in hits]
                used_hits = [
                    h
                    for h in hits_with_display
                    if (h["file_id"], h["chunk_id"]) in used_ref_set
                ]

                citations = [
                    CitationItem(
                        filename=h["filename"],
                        chunk_id=h["chunk_id"],
                        start=h["start"],
                        end=h["end"],
                        text=h["text"],
                    )
                    for h in used_hits
                ]

                hit_items = [HitItem(**h) for h in hits_with_display]
                cited_chunk_ids = [c.chunk_id for c in citations]

                span.set_attribute("citerag.hits_count", len(hits))
                span.set_attribute("citerag.citations_count", len(citations))
                span.set_attribute("citerag.used_chunk_ids_count", len(used_chunk_ids))
                span.set_attribute("citerag.retrieval_ms", retrieval_ms)
                span.set_attribute("citerag.generation_ms", generation_ms)
                span.set_attribute("citerag.rerank_mode", req.rerank_mode)
                span.set_attribute("citerag.retrieve_top_k", effective_retrieve_top_k)
                span.set_attribute("citerag.rerank_ms", rerank_ms)
                span.set_attribute(
                    "citerag.top_hit_chunk_ids",
                    ",".join(str(cid) for cid in reranked_hit_chunk_ids),
                )
                span.set_attribute(
                    "citerag.top_hit_chunk_ids",
                    ",".join(str(cid) for cid in reranked_hit_chunk_ids),
                )
                span.set_attribute("citerag.prompt_tokens", prompt_tokens)
                span.set_attribute("citerag.completion_tokens", completion_tokens)
                span.set_attribute("citerag.total_tokens", total_tokens)
                span.set_attribute("citerag.estimated_cost_usd", estimated_cost_usd)
    

                duration_ms = elapsed_ms(start)
                ASK_COUNT.labels(status="ok", retrieval_mode=retrieval_mode).inc()
                ASK_LATENCY.labels(retrieval_mode=retrieval_mode).observe(
                    duration_ms / 1000
                )
                RETRIEVAL_HITS.labels(retrieval_mode=retrieval_mode).observe(len(hits))
                CITATION_COUNT.labels(retrieval_mode=retrieval_mode).observe(
                    len(citations)
                )

                log_event(
                    "ask",
                    question=req.question,
                    file_id=req.file_id,
                    retrieval_mode=retrieval_mode,
                    top_k=req.top_k,
                    retrieve_top_k=req.retrieve_top_k,
                    effective_retrieve_top_k=effective_retrieve_top_k,
                    question_type=question_type,
                    rerank_mode=req.rerank_mode,
                    rerank_ms=rerank_ms,
                    hits_count=len(hits),
                    citations_count=len(citations),
                    used_chunk_ids_count=len(used_chunk_ids),
                    top_hit_chunk_ids=reranked_hit_chunk_ids,
                    cited_chunk_ids=cited_chunk_ids,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    duration_ms=duration_ms,
                    generator_mode=self.generator.__class__.__name__,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                )

                return llm_result["answer"], citations, hit_items, question_type, effective_retrieve_top_k

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("citerag.error", True)

            duration_ms = elapsed_ms(start)
            ASK_COUNT.labels(status="failed", retrieval_mode=retrieval_mode).inc()

            log_event(
                "ask_failed",
                question=req.question,
                file_id=req.file_id,
                retrieval_mode=retrieval_mode,
                top_k=req.top_k,
                retrieve_top_k=req.retrieve_top_k,
                rerank_mode=req.rerank_mode,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    def ask(self, req: AskRequest) -> AskResponse:
        answer, citations, _, _, _ = self._run_ask(req)
        return AskResponse(
            question=req.question,
            answer=answer,
            citations=citations,
        )

    def ask_debug(self, req: AskRequest) -> AskDebugResponse:
        answer, citations, hits, question_type, effective_retrieve_top_k = self._run_ask(req)
        return AskDebugResponse(
            question=req.question,
            answer=answer,
            citations=citations,
            hits=hits,
            question_type=question_type,
            effective_retrieve_top_k=effective_retrieve_top_k,
        )
