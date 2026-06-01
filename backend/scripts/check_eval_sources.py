import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.database import async_session_maker
from models.database import Document, DocumentChunk


async def main() -> None:
    eval_path = Path("evaluation_sets/rag_eval.json")
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    expected_chunk_ids = [
        source["chunk_id"]
        for case in data["cases"]
        for source in case["expected_sources"]
        if source.get("chunk_id")
    ]
    unique_expected_chunk_ids = sorted(set(expected_chunk_ids))
    expected_document_ids = {
        source["document_id"]
        for case in data["cases"]
        for source in case["expected_sources"]
        if source.get("document_id")
    }

    async with async_session_maker() as session:
        chunk_result = await session.execute(
            select(DocumentChunk.id).where(DocumentChunk.id.in_(unique_expected_chunk_ids))
        )
        found_chunk_ids = {str(chunk_id) for chunk_id in chunk_result.scalars().all()}

        document_result = await session.execute(
            select(Document.id, Document.title).order_by(Document.created_at.desc())
        )
        documents = [(str(doc_id), title) for doc_id, title in document_result.all()]

    missing_chunk_ids = sorted(set(unique_expected_chunk_ids) - found_chunk_ids)
    print(f"expected_chunk_refs={len(expected_chunk_ids)}")
    print(f"unique_expected_chunk_ids={len(unique_expected_chunk_ids)}")
    print(f"found_unique_chunk_ids={len(found_chunk_ids)}")
    print(f"missing_unique_chunk_ids={len(missing_chunk_ids)}")
    for chunk_id in missing_chunk_ids:
        print(f"missing_chunk_id={chunk_id}")
    print(f"expected_document_ids={sorted(expected_document_ids)}")
    print("current_documents=")
    for doc_id, title in documents:
        print(f"  {doc_id}  {title}")


if __name__ == "__main__":
    asyncio.run(main())
