import argparse
import csv
import json
import shutil
from pathlib import Path


OLD_DOCUMENT_ID_TO_TITLE = {
    "39f1a577-76cd-4330-8f25-2eb91aa9cd87": "25EVEN.pdf",
    "22a6edc2-9fe1-4ba9-bb49-9dfe81e4816a": "SMORE.pdf",
    "b9ee6de9-12b5-4d80-bab9-da26899d0db7": "PGL.pdf",
}


def load_chunk_index(chunks_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with chunks_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {
            (row["document_title"], int(row["chunk_index"])): row
            for row in reader
        }


def update_eval_sources(eval_path: Path, chunks_path: Path) -> tuple[int, list[str]]:
    chunk_index = load_chunk_index(chunks_path)
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    updated = 0
    missing: list[str] = []
    for case_index, case in enumerate(data.get("cases", [])):
        for source_index, source in enumerate(case.get("expected_sources", [])):
            old_document_id = source.get("document_id")
            chunk_index_value = source.get("chunk_index")
            title = OLD_DOCUMENT_ID_TO_TITLE.get(old_document_id)

            if title is None or chunk_index_value is None:
                missing.append(f"case={case_index} source={source_index}")
                continue

            row = chunk_index.get((title, int(chunk_index_value)))
            if row is None:
                missing.append(
                    f"case={case_index} source={source_index} title={title} "
                    f"chunk_index={chunk_index_value}"
                )
                continue

            source["document_id"] = row["document_id"]
            source["chunk_id"] = row["chunk_id"]
            updated += 1

    backup_path = eval_path.with_suffix(eval_path.suffix + ".bak")
    shutil.copy2(eval_path, backup_path)
    eval_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update eval expected sources with current exported chunks."
    )
    parser.add_argument(
        "--eval",
        default="evaluation_sets/rag_eval.json",
        help="Evaluation JSON path.",
    )
    parser.add_argument(
        "--chunks",
        default="exports/document_chunks.csv",
        help="Current chunk export CSV path.",
    )
    args = parser.parse_args()

    updated, missing = update_eval_sources(Path(args.eval), Path(args.chunks))
    print(f"updated_sources={updated}")
    print(f"missing_sources={len(missing)}")
    for item in missing:
        print(item)


if __name__ == "__main__":
    main()
