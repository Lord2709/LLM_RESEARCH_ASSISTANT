import re
import logging

from src.generation.generate import answer_query

logger = logging.getLogger(__name__)


def extract_cited_numbers(answer_text: str) -> set[int]:
    """Find every [N] marker actually present in the LLM's answer text."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}


def build_references(cited_numbers: set[int], citation_map: dict[int, dict]) -> list[dict]:
    """Build the final references list, but only for citation numbers the LLM
    actually used in its answer (citation_map may contain more chunks/sources
    than the model chose to cite). Sorted by citation number for a clean,
    predictable reference list."""
    references = []
    for number in sorted(cited_numbers):
        info = citation_map.get(number)
        if info is None:
            logger.warning(f"Answer cites [{number}], which does not exist in the citation map.")
            continue
        references.append({
            "number": number,
            "title": info["title"],
            "source_url": info["source_url"],
        })
    return references


def format_response(answer_text: str, citation_map: dict[int, dict]) -> dict:
    """Package the raw answer + citation map into the final structured
    response: the answer text as-is (inline [N] markers preserved) plus a
    clean references list for the caller (API/frontend) to render."""
    cited_numbers = extract_cited_numbers(answer_text)
    references = build_references(cited_numbers, citation_map)
    return {
        "answer": answer_text,
        "references": references,
    }


def answer_with_citations(query: str) -> dict:
    """Orchestrator: generate the answer, then format it with a references list."""
    answer_text, citation_map = answer_query(query)
    return format_response(answer_text, citation_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    response = answer_with_citations(test_query)

    print(response["answer"])
    print("\n--- references ---")
    for ref in response["references"]:
        print(f"[{ref['number']}] {ref['title']} ({ref['source_url']})")
