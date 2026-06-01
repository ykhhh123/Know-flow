import json
import re
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Citation, DocumentChunk

CITATION_TRAILER_PREFIX = "<<<OK_CITATIONS_JSON:"
CITATION_TRAILER_SUFFIX = ">>>"
REFUSAL_MESSAGE = "知识库中未找到依据，无法基于已上传文档回答该问题。"


class CitationService:
    def build_sources(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        sources = []
        for index, chunk in enumerate(chunks, start=1):
            document = getattr(chunk, "document", None)
            document_id = str(chunk.document_id)
            page_number = self._extract_page_number(chunk.content)

            sources.append(
                {
                    "citationNumber": index,
                    "chunkId": str(chunk.id),
                    "documentId": document_id,
                    "documentTitle": getattr(document, "title", None) or "Untitled document",
                    "chunkIndex": chunk.chunk_index,
                    "pageNumber": page_number,
                    "sectionTitle": self._extract_section_title(chunk.content),
                    "snippet": self._preview(chunk.content),
                    "retrievalScore": getattr(chunk, "retrieval_score", None),
                    "sourceUrl": self._build_source_url(document_id, page_number),
                }
            )
        return sources

    def render_context(self, sources: list[dict[str, Any]], chunks: list[DocumentChunk]) -> str:
        context_blocks = []
        for source, chunk in zip(sources, chunks):
            page = source["pageNumber"] if source["pageNumber"] is not None else "unknown"
            section = source["sectionTitle"] or "unknown"
            context_blocks.append(
                "\n".join(
                    [
                        f"[{source['citationNumber']}]",
                        f"Document: {source['documentTitle']}",
                        f"Page: {page}",
                        f"Section: {section}",
                        f"Chunk ID: {source['chunkId']}",
                        "Content:",
                        chunk.content,
                    ]
                )
            )
        return "\n\n".join(context_blocks)

    def build_trailer(self, answer: str, sources: list[dict[str, Any]]) -> str:
        used_numbers = self.extract_used_numbers(answer, len(sources))
        cited_sources = [
            source
            for source in sources
            if source["citationNumber"] in used_numbers
        ]
        payload = {
            "sources": cited_sources,
            "consistency": {
                "hasEvidence": bool(sources),
                "usedCitationNumbers": used_numbers,
                "invalidCitationNumbers": self.extract_invalid_numbers(answer, len(sources)),
            },
        }
        return (
            f"\n\n{CITATION_TRAILER_PREFIX}"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            f"{CITATION_TRAILER_SUFFIX}"
        )

    def split_trailer(self, text: str) -> tuple[str, dict[str, Any] | None]:
        start = text.find(CITATION_TRAILER_PREFIX)
        if start == -1:
            return text, None

        json_start = start + len(CITATION_TRAILER_PREFIX)
        end = text.find(CITATION_TRAILER_SUFFIX, json_start)
        if end == -1:
            return text[:start].rstrip(), None

        payload_text = text[json_start:end]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = None

        return text[:start].rstrip(), payload

    async def save_message_citations(
        self,
        session: AsyncSession,
        message_id: uuid.UUID,
        payload: dict[str, Any] | None,
    ) -> None:
        if not payload:
            return

        sources = payload.get("sources") or []
        for source in sources:
            session.add(
                Citation(
                    message_id=message_id,
                    chunk_id=uuid.UUID(source["chunkId"]),
                    document_id=uuid.UUID(source["documentId"]),
                    marker_index=int(source["citationNumber"]),
                    document_title=source.get("documentTitle"),
                    section_title=source.get("sectionTitle"),
                    page_number=source.get("pageNumber"),
                    chunk_index=source.get("chunkIndex"),
                    snippet=source.get("snippet"),
                    retrieval_score=source.get("retrievalScore"),
                )
            )

        await session.commit()

    def citation_to_dict(self, citation: Citation) -> dict[str, Any]:
        page_number = citation.page_number
        return {
            "citationNumber": citation.marker_index,
            "chunkId": str(citation.chunk_id),
            "documentId": str(citation.document_id),
            "documentTitle": citation.document_title or "Untitled document",
            "chunkIndex": citation.chunk_index,
            "pageNumber": page_number,
            "sectionTitle": citation.section_title,
            "snippet": citation.snippet,
            "retrievalScore": citation.retrieval_score,
            "sourceUrl": self._build_source_url(str(citation.document_id), page_number),
        }

    def extract_used_numbers(self, answer: str, max_number: int) -> list[int]:
        numbers = {
            int(match)
            for match in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(match) <= max_number
        }
        return sorted(numbers)

    def extract_invalid_numbers(self, answer: str, max_number: int) -> list[int]:
        numbers = {
            int(match)
            for match in re.findall(r"\[(\d+)\]", answer)
            if int(match) < 1 or int(match) > max_number
        }
        return sorted(numbers)

    def _preview(self, content: str, max_length: int = 360) -> str:
        compact = " ".join(content.split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 3] + "..."

    def _extract_page_number(self, content: str) -> int | None:
        patterns = [
            r"---\s*Page\s+(\d+)\s*---",
            r"Page\s+(\d+)",
            r"第\s*(\d+)\s*页",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_section_title(self, content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()[:120]
            if len(stripped) <= 80 and not stripped.endswith((".", "。", "!", "！", "?", "？")):
                return stripped[:120]
        return None

    def _build_source_url(self, document_id: str, page_number: int | None) -> str:
        url = f"/api/documents/{document_id}/file"
        if page_number:
            return f"{url}#page={page_number}"
        return url


citation_service = CitationService()
