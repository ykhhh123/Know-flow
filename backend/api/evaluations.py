from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_session
from models.schemas import RAGEvaluationRequest, RAGEvaluationResponse
from services.evaluation_service import evaluation_service
from services.rag_service import RetrievalFilters

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])

MAX_EVALUATION_CASES = 100
MAX_TOP_K = 50


def _to_retrieval_filters(filters) -> RetrievalFilters | None:
    if filters is None:
        return None

    return RetrievalFilters(
        knowledge_bases=filters.knowledge_bases,
        document_types=filters.document_types,
        created_after=filters.created_after,
        created_before=filters.created_before,
        tags=filters.tags,
        document_ids=filters.document_ids,
    )


@router.post("/rag", response_model=RAGEvaluationResponse)
async def evaluate_rag(
    request: RAGEvaluationRequest,
    session: AsyncSession = Depends(get_session),
):
    """Evaluate RAG retrieval with Recall@K, MRR and average latency."""
    if not request.cases:
        raise HTTPException(400, "At least one evaluation case is required")

    if len(request.cases) > MAX_EVALUATION_CASES:
        raise HTTPException(
            400,
            f"At most {MAX_EVALUATION_CASES} evaluation cases are allowed per run",
        )

    if request.topK < 1 or request.topK > MAX_TOP_K:
        raise HTTPException(400, f"topK must be between 1 and {MAX_TOP_K}")

    if (
        request.retrievalMode.lower() != "keyword"
        and not request.useLocalEmbedding
        and not request.apiKey
    ):
        raise HTTPException(400, "API key required")

    for index, case in enumerate(request.cases):
        if not case.expected_sources:
            raise HTTPException(
                400,
                f"cases[{index}].expected_sources must not be empty",
            )

        for source_index, source in enumerate(case.expected_sources):
            if not source.document_id and not source.chunk_id:
                raise HTTPException(
                    400,
                    (
                        f"cases[{index}].expected_sources[{source_index}] must include "
                        "document_id or chunk_id"
                    ),
                )

    try:
        return await evaluation_service.evaluate_rag(
            cases=request.cases,
            api_key=request.apiKey,
            session=session,
            top_k=request.topK,
            provider=request.provider,
            base_url=request.baseUrl,
            use_local_embedding=request.useLocalEmbedding,
            retrieval_mode=request.retrievalMode,
            filters=_to_retrieval_filters(request.filters),
            use_rerank=request.useRerank,
            rerank_provider=request.rerankProvider,
            rerank_model=request.rerankModel,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
