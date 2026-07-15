import re
import logging
from pathlib import Path

from src.schemas import RawDocument, CleanedDocument

logger = logging.getLogger(__name__)


def strip_references(text: str) -> str:
    """Remove the References/Bibliography section (heading through its content,
    bounded by the next heading or end of document)."""
    pattern = r'^#{1,6}\s+[*_]{0,2}(?:references|bibliography)[*_]{0,2}\s*$.*?(?=^#{1,6}\s+\S|\Z)'
    return re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL).strip()


def clean_document(doc: RawDocument) -> CleanedDocument:
    """Apply cleaning steps to a RawDocument's raw_text and build a CleanedDocument."""
    cleaned_text = strip_references(doc.raw_text)
    return CleanedDocument(
        id=doc.id,
        source_type=doc.source_type,
        title=doc.title,
        cleaned_text=cleaned_text,
        source_url=doc.source_url,
        content_hash=doc.content_hash,
        ingested_at=doc.ingested_at,
        source_metadata=doc.source_metadata,
    )


def append_document(output_path: Path, doc: CleanedDocument) -> None:
    """Cheap path: add one new cleaned document without touching the rest of the file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(doc.model_dump_json() + "\n")


def write_all_documents(output_path: Path, docs: dict[str, CleanedDocument]) -> None:
    """Expensive path: rewrite the entire file, used only when replacing a changed entry."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs.values():
            f.write(doc.model_dump_json() + "\n")


def clean_all_documents(
    input_path: Path = Path("data/raw_documents.jsonl"),
    output_path: Path = Path("data/cleaned_documents.jsonl"),
) -> list[CleanedDocument]:
    """Orchestrator: read raw_documents.jsonl, skip documents whose content_hash
    hasn't changed since last cleaning, (re)clean the rest, log-and-continue on
    per-document failure. Writes incrementally so a crash mid-run doesn't lose
    progress already made (same resilience pattern as ingest_arxiv_papers)."""
    cleaned_documents: list[CleanedDocument] = []

    existing_docs: dict[str, CleanedDocument] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = CleanedDocument.model_validate_json(line)
                existing_docs[doc.id] = doc

    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return cleaned_documents

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [line for line in f if line.strip()]

    for line in raw_lines:
        try:
            raw_doc = RawDocument.model_validate_json(line)
        except Exception as e:
            logger.error(f"Failed to parse a raw document line: {e}")
            continue

        existing_cleaned_doc = existing_docs.get(raw_doc.id)

        if existing_cleaned_doc is not None and existing_cleaned_doc.content_hash == raw_doc.content_hash:
            logger.info(f"Skipping unchanged document {raw_doc.id}: {raw_doc.title}")
            cleaned_documents.append(existing_cleaned_doc)
            continue

        try:
            cleaned_doc = clean_document(raw_doc)
        except Exception as e:
            logger.error(f"Failed to clean document {raw_doc.id}: {raw_doc.title}. Error: {e}")
            continue

        cleaned_documents.append(cleaned_doc)
        existing_docs[cleaned_doc.id] = cleaned_doc

        if existing_cleaned_doc is not None:
            # this id existed before with a different hash: a genuine update,
            # so the stale entry must be replaced -> full rewrite
            write_all_documents(output_path, existing_docs)
        else:
            # brand new document: cheap append, no rewrite needed
            append_document(output_path, cleaned_doc)

        logger.info(f"Cleaned document {raw_doc.id}: {raw_doc.title}")

    return cleaned_documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = clean_all_documents()
    print(f"Total cleaned documents: {len(docs)}")