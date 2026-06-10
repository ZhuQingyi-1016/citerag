from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger("uvicorn.error")


class BaseAnswerGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        hits: list[dict[str, Any]],
        question_type: str = "general",
    ) -> dict[str, Any]:
        """
        Return:
        {
            "answer": str,
            "used_chunk_refs": list[{"file_id": str, "chunk_id": int}]
        }
        """
        raise NotImplementedError


def _looks_chinese_question(question: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in question)


def build_prompt(
    question: str,
    hits: list[dict[str, Any]],
    question_type: str = "general",
) -> str:
    context_blocks = [f"[chunk_id={h['chunk_id']}]\n{h['text']}" for h in hits]
    context = "\n\n".join(context_blocks)

    language_instruction = "Answer in Chinese." if _looks_chinese_question(question) else "Answer in the same language as the question."
    process_instruction = ""
    if question_type == "process":
        process_instruction = (
            "This is a development-process question. "
            "Answer in a stage-by-stage or timeline structure (e.g., early -> later -> current). "
            "If the retrieved context does not explicitly provide a development process, "
            "state that clearly and do not fabricate one."
        )

    return (
        "You are a citation-grounded assistant.\n"
        "Use only the provided context to answer the user's question.\n"
        "If the answer cannot be determined from the context, say so clearly.\n"
        f"{language_instruction}\n"
        f"{process_instruction}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n"
    )


class FakeAnswerGenerator(BaseAnswerGenerator):
    def _clean_text(self, text: str) -> str:
        # 尽量去掉一些不适合直接拿来回答的噪声
        bad_markers = [
            '"question":',
            '"answer":',
            '"citations":',
            '"hits":',
            '"score":',
            "<file_id>",
        ]

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(marker in line for marker in bad_markers):
                continue
            lines.append(line)

        cleaned = " ".join(lines)
        return cleaned[:400]

    def generate(
        self,
        question: str,
        hits: list[dict],
        question_type: str = "general",
    ) -> dict:
        if not hits:
            return {
                "answer": "我在已上传的文档中没有检索到相关内容，因此无法回答。",
                "used_chunk_ids": [],
                "used_chunk_refs": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            }

        selected_hits = hits[: min(3, len(hits))]
        used_chunk_ids = [h["chunk_id"] for h in selected_hits]
        used_chunk_refs = [
            {
                "file_id": h["file_id"],
                "chunk_id": h["chunk_id"],
            }
            for h in selected_hits
        ]
        bullet_lines = []
        for h in selected_hits:
            text = h["text"].strip().replace("\n", " ")
            bullet_lines.append(f"- {text[:120]}")

        answer = "根据当前检索内容，可以总结为：\n" + "\n".join(bullet_lines)

        return {
            "answer": answer,
            "used_chunk_ids": used_chunk_ids,
            "used_chunk_refs": used_chunk_refs,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        }


class OpenAICompatibleAnswerGenerator(BaseAnswerGenerator):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("AIHUBMIX_API_KEY")
        self.base_url = base_url or os.getenv("AI_BASE_URL", "https://aihubmix.com/v1")
        self.model = model or os.getenv("AI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("Missing API key for OpenAI-compatible generator")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        question: str,
        hits: list[dict],
        question_type: str = "general",
    ) -> dict:
        prompt = build_prompt(
            question=question,
            hits=hits,
            question_type=question_type,
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You answer questions using only retrieved context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

        selected_hits = hits[: min(3, len(hits))]
        used_chunk_ids = [h["chunk_id"] for h in selected_hits]
        used_chunk_refs = [
            {
                "file_id": h["file_id"],
                "chunk_id": h["chunk_id"],
            }
            for h in selected_hits
        ]

        estimated_cost_usd = self.estimate_cost_usd(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return {
            "answer": answer,
            "used_chunk_ids": used_chunk_ids,
            "used_chunk_refs": used_chunk_refs,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
            },
        }
    
    def estimate_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        # 这里先写你当前模型的“手动配置价格”
        # 后面你可以再把它抽到 settings 里
        input_price_per_1m = 0.15
        output_price_per_1m = 0.60

        cost = (
            (prompt_tokens / 1_000_000) * input_price_per_1m
            + (completion_tokens / 1_000_000) * output_price_per_1m
        )
        return round(cost, 8)


def get_answer_generator() -> BaseAnswerGenerator:
    mode = os.getenv("GENERATOR_MODE", "fake").lower()
    logger.warning("GENERATOR_MODE = %s", mode)

    if mode == "real":
        logger.warning("Using OpenAICompatibleAnswerGenerator")
        return OpenAICompatibleAnswerGenerator()

    logger.warning("Using FakeAnswerGenerator")
    return FakeAnswerGenerator()
