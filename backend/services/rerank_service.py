import asyncio
from importlib import import_module
from typing import Any, Optional

import httpx


DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
DEFAULT_BGE_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_QWEN_RERANK_MODEL = "gte-rerank-v2"


class RerankService:
    def __init__(self) -> None:
        self._cross_encoders: dict[str, Any] = {}

    async def rerank(
        self,
        query: str,
        ranked_chunks: list,
        provider: str,
        model: Optional[str] = None,
        api_key: str = "",
        base_url: Optional[str] = None,
    ) -> list:
        provider = (provider or "").strip().lower()
        if not ranked_chunks or provider in {"", "none", "off", "false"}:
            return ranked_chunks

        try:
            if provider in {"bge", "cross_encoder", "cross-encoder", "local"}:
                return await self._rerank_cross_encoder(
                    query=query,
                    ranked_chunks=ranked_chunks,
                    model=model or DEFAULT_BGE_RERANK_MODEL,
                )

            if provider in {"qwen", "dashscope", "alibaba"}:
                return await self._rerank_dashscope(
                    query=query,
                    ranked_chunks=ranked_chunks,
                    model=model or DEFAULT_QWEN_RERANK_MODEL,
                    api_key=api_key,
                    base_url=base_url,
                )
        except Exception as exc:
            print(f"[Rerank Warning] {type(exc).__name__}: {exc}")

        return ranked_chunks

    async def _rerank_cross_encoder(
        self,
        query: str,
        ranked_chunks: list,
        model: str,
    ) -> list:
        cross_encoder = await self._get_cross_encoder(model)
        pairs = [(query, ranked.chunk.content) for ranked in ranked_chunks]
        scores = await asyncio.to_thread(cross_encoder.predict, pairs)

        reranked = []
        for ranked, score in zip(ranked_chunks, scores):
            ranked.rerank_score = float(score)
            ranked.score = float(score)
            reranked.append(ranked)

        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)

    async def _get_cross_encoder(self, model: str) -> Any:
        if model in self._cross_encoders:
            return self._cross_encoders[model]

        def load_model() -> Any:
            sentence_transformers = import_module("sentence_transformers")
            cross_encoder_cls = getattr(sentence_transformers, "CrossEncoder")
            return cross_encoder_cls(model)

        cross_encoder = await asyncio.to_thread(load_model)
        self._cross_encoders[model] = cross_encoder
        return cross_encoder

    async def _rerank_dashscope(
        self,
        query: str,
        ranked_chunks: list,
        model: str,
        api_key: str,
        base_url: Optional[str],
    ) -> list:
        if not api_key:
            raise ValueError("API key required for Qwen/DashScope rerank")

        endpoint = base_url.rstrip("/") if base_url else DASHSCOPE_RERANK_URL
        documents = [ranked.chunk.content for ranked in ranked_chunks]
        payload = {
            "model": model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
                "top_n": len(documents),
            },
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        results = data.get("output", {}).get("results", [])
        by_index = {int(item["index"]): item for item in results if "index" in item}

        reranked = []
        for index, ranked in enumerate(ranked_chunks):
            item = by_index.get(index)
            if not item:
                reranked.append(ranked)
                continue

            score = item.get("relevance_score", ranked.score)
            ranked.rerank_score = float(score)
            ranked.score = float(score)
            reranked.append(ranked)

        return sorted(
            reranked,
            key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
            reverse=True,
        )


rerank_service = RerankService()
