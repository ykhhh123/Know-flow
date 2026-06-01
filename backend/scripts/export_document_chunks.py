import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from sqlalchemy import select

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.database import async_session_maker
from models.database import Document, DocumentChunk


async def export_chunks(output_path: Path, output_format: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with async_session_maker() as session:
        result = await session.execute(
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by(Document.title.asc(), DocumentChunk.chunk_index.asc())
        )
        rows = result.all()

    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as file:
            for chunk_id, document_id, document_title, chunk_index, content in rows:
                file.write(
                    json.dumps(
                        {
                            "chunk_id": str(chunk_id),
                            "document_id": str(document_id),
                            "document_title": document_title,
                            "chunk_index": chunk_index,
                            "chunk_text": content,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "chunk_id",
                    "document_id",
                    "document_title",
                    "chunk_index",
                    "chunk_text",
                ],
            )
            writer.writeheader()
            for chunk_id, document_id, document_title, chunk_index, content in rows:
                writer.writerow(
                    {
                        "chunk_id": str(chunk_id),
                        "document_id": str(document_id),
                        "document_title": document_title,
                        "chunk_index": chunk_index,
                        "chunk_text": content,
                    }
                )

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export document chunks.")
    parser.add_argument(
        "--output",
        default="exports/document_chunks.csv",
        help="Output file path. Defaults to exports/document_chunks.csv",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Export format. Defaults to csv",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    count = asyncio.run(export_chunks(output_path, args.format))
    print(f"Exported {count} chunks to {output_path}")


if __name__ == "__main__":
    main()
