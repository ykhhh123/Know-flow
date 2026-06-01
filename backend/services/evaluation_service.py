from time import perf_counter
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import DocumentChunk
from models.schemas import (
    RAGEvaluationCase,
    RAGEvaluationCaseResult,
    RAGEvaluationExpectedSource,
    RAGEvaluationResponse,
    RAGEvaluationRetrievedSource,
)
from services.rag_service import RetrievalFilters, rag_service


class EvaluationService:
    async def evaluate_rag(
        self,
        cases: List[RAGEvaluationCase],
        api_key: str,
        session: AsyncSession,
        top_k: int = 5,
        provider: str = "openai",
        base_url: Optional[str] = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ) -> RAGEvaluationResponse:
        """Evaluate RAG retrieval with Recall@K, MRR and average latency."""
        results: list[RAGEvaluationCaseResult] = []

        for case in cases:
            started_at = perf_counter()
            chunks = await rag_service.search_similar(
                query=case.query,
                api_key=api_key,
                session=session,
                limit=top_k,
                provider=provider,
                base_url=base_url,
                use_local_embedding=use_local_embedding,
                retrieval_mode=retrieval_mode,
                filters=filters,
                use_rerank=use_rerank,
                rerank_provider=rerank_provider,
                rerank_model=rerank_model,
            )
            latency_ms = (perf_counter() - started_at) * 1000

            case_result = self._score_case(case, chunks, latency_ms)
            results.append(case_result)

        total_cases = len(results)
        recall_at_k = (
            sum(result.recall for result in results) / total_cases
            if total_cases
            else 0.0
        )
        mrr = (
            sum(result.reciprocal_rank for result in results) / total_cases
            if total_cases
            else 0.0
        )
        avg_latency_ms = (
            sum(result.latency_ms for result in results) / total_cases
            if total_cases
            else 0.0
        )

        return RAGEvaluationResponse(
            total_cases=total_cases,
            topK=top_k,
            recall_at_k=round(recall_at_k, 4),
            mrr=round(mrr, 4),
            avg_latency_ms=round(avg_latency_ms, 2),
            results=results,
        )

    def _score_case(
        self,
        case: RAGEvaluationCase,
        chunks: List[DocumentChunk],
        latency_ms: float,
    ) -> RAGEvaluationCaseResult:
        matched_expected_indexes: set[int] = set()
        first_relevant_rank = None
        retrieved_sources: list[RAGEvaluationRetrievedSource] = []

        for rank, chunk in enumerate(chunks, start=1):
            matched_indexes = [
                index
                for index, expected in enumerate(case.expected_sources)
                if self._matches_expected_source(chunk, expected)
            ]
            matched = bool(matched_indexes)

            if matched:
                matched_expected_indexes.update(matched_indexes)
                if first_relevant_rank is None:
                    first_relevant_rank = rank

            retrieved_sources.append(
                RAGEvaluationRetrievedSource(
                    rank=rank,
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    chunk_index=chunk.chunk_index,
                    matched=matched,
                    content_preview=self._preview(chunk.content),
                    retrieval_score=getattr(chunk, "retrieval_score", None),
                    vector_rank=getattr(chunk, "vector_rank", None),
                    keyword_rank=getattr(chunk, "keyword_rank", None),
                    rerank_score=getattr(chunk, "rerank_score", None),
                )
            )

        expected_count = len(case.expected_sources)
        recall = (
            len(matched_expected_indexes) / expected_count
            if expected_count
            else 0.0
        )
        reciprocal_rank = (
            1 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )

        return RAGEvaluationCaseResult(
            query=case.query,
            hit=first_relevant_rank is not None,
            rank=first_relevant_rank,
            reciprocal_rank=round(reciprocal_rank, 4),
            recall=round(recall, 4),
            latency_ms=round(latency_ms, 2),
            expected_sources=case.expected_sources,
            retrieved_sources=retrieved_sources,
        )

    def _matches_expected_source(
        self,
        chunk: DocumentChunk,
        expected: RAGEvaluationExpectedSource,
    ) -> bool:
        if expected.chunk_id and str(chunk.id) == expected.chunk_id:
            return True

        if not expected.document_id:
            return False

        if str(chunk.document_id) != expected.document_id:
            return False

        return expected.chunk_index is None or chunk.chunk_index == expected.chunk_index

    def _preview(self, content: str, max_length: int = 240) -> str:
        compact = " ".join(content.split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 3] + "..."


evaluation_service = EvaluationService()
