import time
import hashlib
import logging
from pathlib import Path
import re
from datetime import date
import requests
import feedparser
import pymupdf4llm

from src.schemas import RawDocument, ArxivMetadata, SourceType

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
PDF_STORAGE_DIR = Path("data/raw_pdfs")


def search_arxiv(query: str, max_results: int = 50) -> list[feedparser.FeedParserDict]:
    """Query arXiv API, return parsed feed entries (one per paper)."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=10)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    return feed.entries



def parse_entry_to_metadata(entry: feedparser.FeedParserDict) -> tuple[str, ArxivMetadata]:
    """Convert a single feedparser entry into an ArxivMetadata object."""
    published_date = date(*entry.published_parsed[:3])
    updated_date_candidate = date(*entry.updated_parsed[:3])
    arxiv = ArxivMetadata(
        arxiv_id = re.findall(r'[0-9]{4}\.[0-9]{5}', entry.id)[0],
        authors=[author.name for author in entry.authors],
        published_date=published_date,
        updated_date = updated_date_candidate if updated_date_candidate != published_date else None,
        abstract=entry.summary,
        categories=[tag['term'] for tag in entry.tags],
        doi=entry.get('arxiv_doi', None),
        journal_reference=entry.get('arxiv_journal_ref', None),
        pdf_url = next((link.href for link in entry.links if link.get('title') == 'pdf'), None),
        version=int(entry.id.split('v')[-1]),
    )
    return (entry.title,arxiv)


def download_pdf(pdf_url: str, save_path: Path) -> Path:
    """Download the PDF from pdf_url, save it to save_path, return the path."""
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path


def extract_text(pdf_path: Path) -> str:
    """Run pymupdf4llm extraction on the saved PDF, return markdown text."""
    markdown = pymupdf4llm.to_markdown(pdf_path)
    return markdown


def compute_content_hash(text: str) -> str:
    """Return a sha256 hash of the given text, for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_raw_document(metadata: ArxivMetadata, title: str, raw_text: str, source_url: str) -> RawDocument:
    """Assemble a RawDocument from metadata + extracted text."""
    return RawDocument(
        source_type=SourceType.ARXIV,
        title=title,
        raw_text=raw_text,
        source_url=source_url,
        content_hash=compute_content_hash(raw_text),
        source_metadata=metadata,
    )


def append_document(output_path: Path, doc: RawDocument) -> None:
    """Cheap path: add one new document without touching the rest of the file."""
    with open(output_path, "a") as f:
        f.write(doc.model_dump_json() + "\n")


def write_all_documents(output_path: Path, docs: dict[str, RawDocument]) -> None:
    """Expensive path: rewrite the entire file from scratch, used only when an
    existing entry needs to be replaced by a newer version."""
    with open(output_path, "w") as f:
        for doc in docs.values():
            f.write(doc.model_dump_json() + "\n")


def ingest_arxiv_papers(
    query: str, max_results: int = 50, output_path: Path = Path("data/raw_documents.jsonl")
) -> list[RawDocument]:
    """Orchestrator: search -> download -> extract -> build -> cleanup, with
    rate limiting (time.sleep) and log-and-continue error handling per paper.
    Detects revised papers (newer version) and replaces the stale entry."""
    documents: list[RawDocument] = []
    entries = search_arxiv(query, max_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_docs: dict[str, RawDocument] = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = RawDocument.model_validate_json(line)
                existing_docs[doc.source_metadata.arxiv_id] = doc

    for entry in entries:
        save_path: Path | None = None
        try:
            title, metadata = parse_entry_to_metadata(entry)

            existing = existing_docs.get(metadata.arxiv_id)
            if existing is not None and existing.source_metadata.version >= metadata.version:
                logger.info(f"Skipping up-to-date paper {metadata.arxiv_id}: {title}")
                continue

            if metadata.pdf_url is None:
                logger.warning(f"PDF not found for {title}")
                continue

            save_path = PDF_STORAGE_DIR / f"{metadata.arxiv_id}.pdf"
            download_pdf(metadata.pdf_url, save_path)
            raw_text = extract_text(save_path)
            save_path.unlink()

            doc = build_raw_document(metadata, title, raw_text, entry.link)
            existing_docs[metadata.arxiv_id] = doc
            documents.append(doc)

            if existing is not None:
                write_all_documents(output_path, existing_docs)
            else:
                append_document(output_path, doc)

        except Exception as e:
            if save_path is not None and save_path.exists():
                save_path.unlink()
            logger.warning(f"Failed processing entry {entry.get('id', 'unknown')}: {e}")
            continue
        finally:
            time.sleep(3)

    return documents

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # entries = search_arxiv("all:LLM", max_results=1)
    # entry = entries[0]
    # print(entry.keys())
    # print(entry)

    docs = ingest_arxiv_papers("all:LLM", max_results=3)