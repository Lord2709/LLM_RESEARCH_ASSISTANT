import logging
from pathlib import Path

from src.embedding.embed import load_chunks
from src.chunking.chunk import count_tokens
from src.schemas import Chunk

logger = logging.getLogger(__name__)

# Keeps a single request comfortably under Groq's free-tier 12K TPM cap
# (leaving headroom for the system prompt, question, and output tokens) while
# also bounding cost/context-dilution regardless of which provider is used.
MAX_CONTEXT_TOKENS = 6000


def get_chunks_by_id(chunk_ids: list[str], chunks_path: Path = Path("data/chunks.jsonl")) -> dict[str, Chunk]:
    """Load chunks.jsonl and return {chunk_id: Chunk} for the given ids."""
    all_chunks = load_chunks(chunks_path)
    chunks_by_id = {chunk.id: chunk for chunk in all_chunks}
    return {chunk_id: chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id}


def truncate_to_token_budget(
    chunk_ids: list[str], chunks_by_id: dict[str, Chunk], max_tokens: int = MAX_CONTEXT_TOKENS
) -> list[str]:
    """Keep chunks in order (best-first, since chunk_ids arrives already
    reranked) until the token budget is exhausted. Always keeps at least the
    first chunk even if it alone exceeds the budget, so a single highly
    relevant match is never dropped entirely just because it's large."""
    kept_ids: list[str] = []
    running_total = 0

    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        token_count = count_tokens(chunk.chunk_text)
        if kept_ids and running_total + token_count > max_tokens:
            break

        kept_ids.append(chunk_id)
        running_total += token_count

    if len(kept_ids) < len(chunk_ids):
        logger.info(
            f"Context truncated to fit {max_tokens}-token budget: "
            f"kept {len(kept_ids)} of {len(chunk_ids)} chunks (~{running_total} tokens)"
        )

    return kept_ids


def build_citation_map(chunk_ids: list[str], chunks_by_id: dict[str, Chunk]) -> dict[int, dict]:
    """Assign one citation number per source document (grouped by document_id),
    in order of first appearance in chunk_ids (i.e. best-first reranked order).
    Every contributing chunk_id is tracked under its document's citation number,
    so display stays clean ([1], [2]...) while chunk-level traceability is kept
    for later citation verification."""
    citation_map: dict[int, dict] = {}
    document_id_to_citation_number: dict[str, int] = {}
    next_citation_number = 1

    for chunk_id in chunk_ids:
        chunk = chunks_by_id[chunk_id]
        document_id = chunk.document_id

        if document_id not in document_id_to_citation_number:
            document_id_to_citation_number[document_id] = next_citation_number
            citation_map[next_citation_number] = {
                "title": chunk.title,
                "source_url": chunk.source_url,
                "chunk_ids": [],
            }
            next_citation_number += 1

        citation_number = document_id_to_citation_number[document_id]
        citation_map[citation_number]["chunk_ids"].append(chunk_id)

    return citation_map


def build_context_string(
    chunk_ids: list[str], chunks_by_id: dict[str, Chunk], citation_map: dict[int, dict]
) -> str:
    """Assemble chunk texts (best-first order preserved) into one context block,
    each chunk tagged with its document-level citation number."""
    chunk_id_to_citation_number = {
        chunk_id: citation_number
        for citation_number, info in citation_map.items()
        for chunk_id in info["chunk_ids"]
    }

    sections = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id[chunk_id]
        citation_number = chunk_id_to_citation_number[chunk_id]
        sections.append(f"[{citation_number}] {chunk.chunk_text}")

    return "\n\n".join(sections)


def build_context(chunk_ids: list[str], max_tokens: int = MAX_CONTEXT_TOKENS) -> tuple[str, dict[int, dict]]:
    """Orchestrator: given reranked chunk_ids (best first), truncate to a
    token budget, then return the final context string to send to the LLM
    plus the citation map ([N] -> title, source_url, contributing chunk_ids).
    Truncating before building the citation map keeps both in sync with
    what's actually sent to the LLM -- no citation numbers referencing
    chunks that got dropped for budget reasons."""
    chunks_by_id = get_chunks_by_id(chunk_ids)
    truncated_ids = truncate_to_token_budget(chunk_ids, chunks_by_id, max_tokens)

    citation_map = build_citation_map(truncated_ids, chunks_by_id)
    context_string = build_context_string(truncated_ids, chunks_by_id, citation_map)
    return context_string, citation_map


if __name__ == "__main__":
    from src.retrieval.rerank import rerank_search

    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    reranked = rerank_search(test_query)
    ids_in_order = [chunk_id for chunk_id, _ in reranked]

    context_string, citation_map = build_context(ids_in_order)
    print(context_string)
    print("\n--- citation map ---")
    for number, info in citation_map.items():
        print(f"[{number}] {info['title']} ({info['source_url']}) - {len(info['chunk_ids'])} chunk(s)")
