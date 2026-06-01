from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional
from uuid import UUID

from sqlalchemy import String, cast, desc, func, or_, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.database import Document, DocumentChunk
from services.embedding_service import embedding_service
from services.rerank_service import rerank_service


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20


@dataclass
class RetrievalFilters:
    knowledge_bases: Optional[list[str]] = None
    document_types: Optional[list[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None


@dataclass
class RankedChunk:
    chunk: DocumentChunk
    score: float
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RAGService:
    async def search_similar(
        self,
        query: str,
        api_key: str,
        session: AsyncSession,
        limit: int = 5,
        provider: str = "openai",
        base_url: str = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        candidate_limit: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Search document chunks with vector/keyword hybrid recall, RRF and rerank."""
        ranked_chunks = await self.search_ranked(
            query=query,
            api_key=api_key,
            session=session,
            limit=limit,
            provider=provider,
            base_url=base_url,
            use_local_embedding=use_local_embedding,
            retrieval_mode=retrieval_mode,
            filters=filters,
            use_rerank=use_rerank,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            candidate_limit=candidate_limit,
        )

        chunks: list[DocumentChunk] = []
        for ranked in ranked_chunks:
            setattr(ranked.chunk, "retrieval_score", ranked.score)
            setattr(ranked.chunk, "vector_rank", ranked.vector_rank)
            setattr(ranked.chunk, "keyword_rank", ranked.keyword_rank)
            setattr(ranked.chunk, "vector_score", ranked.vector_score)
            setattr(ranked.chunk, "keyword_score", ranked.keyword_score)
            setattr(ranked.chunk, "rerank_score", ranked.rerank_score)
            chunks.append(ranked.chunk)
        return chunks

    async def search_ranked(
        self,
        query: str,
        api_key: str,
        session: AsyncSession,
        limit: int = 5,
        provider: str = "openai",
        base_url: str = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        candidate_limit: Optional[int] = None,
    ) -> list[RankedChunk]:
        limit = max(1, limit)
        retrieval_mode = (retrieval_mode or "hybrid").strip().lower()
        if retrieval_mode not in {"hybrid", "vector", "keyword"}:
            retrieval_mode = "hybrid"
        candidate_limit = candidate_limit or max(
            MIN_CANDIDATES,
            limit * DEFAULT_CANDIDATE_MULTIPLIER,
        )

        vector_results: list[RankedChunk] = []
        keyword_results: list[RankedChunk] = []

        if retrieval_mode in {"hybrid", "vector"}:
            query_embedding = await embedding_service.get_single_embedding(
                query, api_key, provider, base_url, use_local_embedding
            )
            vector_results = await self._vector_search(
                query_embedding=query_embedding,
                session=session,
                limit=candidate_limit,
                filters=filters,
            )

        if retrieval_mode in {"hybrid", "keyword"}:
            keyword_results = await self._keyword_search(
                query=query,
                session=session,
                limit=candidate_limit,
                filters=filters,
            )

        if retrieval_mode == "vector":
            fused = vector_results
        elif retrieval_mode == "keyword":
            fused = keyword_results
        else:
            fused = self._rrf_fuse(
                vector_results=vector_results,
                keyword_results=keyword_results,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

        if not fused:
            return []

        rerank_candidates = fused[:candidate_limit]
        if use_rerank and rerank_provider:
            rerank_candidates = await rerank_service.rerank(
                query=query,
                ranked_chunks=rerank_candidates,
                provider=rerank_provider,
                model=rerank_model,
                api_key=api_key,
                base_url=base_url,
            )

        return rerank_candidates[:limit]

    async def get_context_for_query(
        self,
        query: str,
        api_key: str,
        session: AsyncSession,
        limit: int = 5,
        provider: str = "openai",
        base_url: str = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ) -> str:
        """Get relevant context for a query."""
        chunks = await self.search_similar(
            query=query,
            api_key=api_key,
            session=session,
            limit=limit,
            provider=provider,
            base_url=base_url,
            use_local_embedding=use_local_embedding,
            retrieval_mode=retrieval_mode,
            filters=filters,
            use_rerank=use_rerank,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
        )

        if not chunks:
            return ""

        return "\n\n".join([chunk.content for chunk in chunks])

    async def _vector_search(
        self,
        query_embedding: list[float],
        session: AsyncSession,
        limit: int,
        filters: Optional[RetrievalFilters],
    ) -> list[RankedChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(DocumentChunk, distance.label("distance"))
            .options(selectinload(DocumentChunk.document))
            .join(Document)
            .where(*self._build_filter_clauses(filters))
            .order_by(distance)
            .limit(limit)
        )
        result = await session.execute(stmt)

        ranked: list[RankedChunk] = []
        for rank, (chunk, distance_value) in enumerate(result.all(), start=1):
            distance_float = float(distance_value or 0)
            ranked.append(
                RankedChunk(
                    chunk=chunk,
                    score=1 / (DEFAULT_RRF_K + rank),
                    vector_rank=rank,
                    vector_score=1 - distance_float,
                )
            )
        return ranked

    async def _keyword_search(
        self,
        query: str,
        session: AsyncSession,
        limit: int,
        filters: Optional[RetrievalFilters],
    ) -> list[RankedChunk]:
        tsvector = func.to_tsvector("simple", DocumentChunk.content)
        tsquery = func.websearch_to_tsquery("simple", query)
        keyword_score = func.ts_rank_cd(tsvector, tsquery).label("keyword_score")
        stmt = (
            select(DocumentChunk, keyword_score)
            .options(selectinload(DocumentChunk.document))
            .join(Document)
            .where(*self._build_filter_clauses(filters), tsvector.op("@@")(tsquery))
            .order_by(desc(keyword_score))
            .limit(limit)
        )
        result = await session.execute(stmt)

        ranked: list[RankedChunk] = []
        for rank, (chunk, score) in enumerate(result.all(), start=1):
            ranked.append(
                RankedChunk(
                    chunk=chunk,
                    score=1 / (DEFAULT_RRF_K + rank),
                    keyword_rank=rank,
                    keyword_score=float(score or 0),
                )
            )
        return ranked

    def _rrf_fuse(
        self,
        vector_results: list[RankedChunk],
        keyword_results: list[RankedChunk],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[RankedChunk]:
        fused: dict[str, RankedChunk] = {}

        for ranked in vector_results:
            chunk_id = str(ranked.chunk.id)
            score = vector_weight / (DEFAULT_RRF_K + (ranked.vector_rank or 0))
            fused[chunk_id] = RankedChunk(
                chunk=ranked.chunk,
                score=score,
                vector_rank=ranked.vector_rank,
                vector_score=ranked.vector_score,
            )

        for ranked in keyword_results:
            chunk_id = str(ranked.chunk.id)
            score = keyword_weight / (DEFAULT_RRF_K + (ranked.keyword_rank or 0))
            if chunk_id not in fused:
                fused[chunk_id] = RankedChunk(
                    chunk=ranked.chunk,
                    score=score,
                    keyword_rank=ranked.keyword_rank,
                    keyword_score=ranked.keyword_score,
                )
                continue

            existing = fused[chunk_id]
            existing.score += score
            existing.keyword_rank = ranked.keyword_rank
            existing.keyword_score = ranked.keyword_score

        return sorted(fused.values(), key=lambda item: item.score, reverse=True)

    def _build_filter_clauses(self, filters: Optional[RetrievalFilters]) -> list:
        if filters is None:
            return []

        clauses = []
        knowledge_bases = self._clean_strings(filters.knowledge_bases)
        document_types = self._normalize_document_types(filters.document_types)
        tags = self._clean_strings(filters.tags)
        document_ids = self._parse_document_ids(filters.document_ids)

        if knowledge_bases:
            normalized = [item.lower() for item in knowledge_bases]
            knowledge_base_expr = func.lower(func.coalesce(Document.knowledge_base, "default"))
            clauses.append(knowledge_base_expr.in_(normalized))

        if document_types:
            clauses.append(Document.file_type.in_(document_types))

        if filters.created_after:
            clauses.append(Document.created_at >= filters.created_after)

        if filters.created_before:
            clauses.append(Document.created_at <= filters.created_before)

        if tags:
            tags_expr = func.string_to_array(func.coalesce(Document.tags, ""), ",")
            clauses.append(tags_expr.op("&&")(cast(tags, ARRAY(String))))

        if document_ids:
            clauses.append(Document.id.in_(document_ids))

        return clauses

    def _clean_strings(self, values: Optional[Iterable[str]]) -> list[str]:
        if not values:
            return []
        return [value.strip() for value in values if value and value.strip()]

    def _normalize_document_types(self, values: Optional[Iterable[str]]) -> list[str]:
        normalized = []
        for value in self._clean_strings(values):
            normalized.append(value if value.startswith(".") else f".{value}")
        return normalized

    def _parse_document_ids(self, values: Optional[Iterable[str]]) -> list[UUID]:
        parsed = []
        for value in self._clean_strings(values):
            parsed.append(UUID(value))
        return parsed


rag_service = RAGService()
