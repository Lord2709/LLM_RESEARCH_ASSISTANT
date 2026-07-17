import logging

from src.schemas import Chunk
from src.context.context import get_chunks_by_id
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank_search

logger = logging.getLogger(__name__)

# Hand-labeled (query, expected_document_id) pairs. Document-level relevance:
# a query "hits" if the correct paper appears anywhere in the ranked results,
# regardless of which specific chunk. Expand this set as the corpus grows.
EVAL_SET = [
    {"query": "GQA to MLA transition for financial LLM deployment", "expected_document_id": "48aedaa1-e17c-4c3c-a44a-4eee0409bbcd"},
    {"query": "layer-adaptive FreqFold size KV cache compression", "expected_document_id": "48aedaa1-e17c-4c3c-a44a-4eee0409bbcd"},
    {"query": "retrieval augmented generation toolkit for LLM applications", "expected_document_id": "304eb132-1315-40e3-85dc-1d7d9f1487ba"},
    {"query": "modular RAG system with pluggable retrieval components", "expected_document_id": "304eb132-1315-40e3-85dc-1d7d9f1487ba"},
    {"query": "fully binarized large language models trained from scratch", "expected_document_id": "ca1f7c95-7af9-4523-b136-72cf03cf17ba"},
    {"query": "autoregressive distillation for binarized neural networks", "expected_document_id": "ca1f7c95-7af9-4523-b136-72cf03cf17ba"},
]


def reciprocal_rank(ranked_chunk_ids: list[str], expected_document_id: str, chunks_by_id: dict[str, Chunk]) -> float:
    """1/rank of the first chunk whose document_id matches expected_document_id,
    or 0.0 if the correct document never appears in the ranked results."""
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk and chunk.document_id == expected_document_id:
            return 1 / rank
    return 0.0


def mean_reciprocal_rank(eval_set: list[dict], search_fn, top_k: int = 10) -> float:
    """Average reciprocal rank across every query in eval_set, using search_fn
    (hybrid_search or rerank_search) to produce the ranked chunk_ids."""
    reciprocal_ranks = []

    for item in eval_set:
        results = search_fn(item["query"], top_k=top_k)
        ranked_chunk_ids = [chunk_id for chunk_id, _ in results]

        chunks_by_id = get_chunks_by_id(ranked_chunk_ids)
        rr = reciprocal_rank(ranked_chunk_ids, item["expected_document_id"], chunks_by_id)
        reciprocal_ranks.append(rr)

        logger.info(f"query={item['query']!r} reciprocal_rank={rr:.4f}")

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def run_evaluation(top_k: int = 10) -> None:
    """Compare MRR before reranking (hybrid_search alone) vs after
    (rerank_search), on the same query set, to measure reranking's actual
    contribution rather than relying on eyeballing single examples."""
    print("--- hybrid_search (pre-rerank) ---")
    hybrid_mrr = mean_reciprocal_rank(EVAL_SET, hybrid_search, top_k=top_k)

    print("\n--- rerank_search (post-rerank) ---")
    rerank_mrr = mean_reciprocal_rank(EVAL_SET, rerank_search, top_k=top_k)

    print(f"\nHybrid search MRR:  {hybrid_mrr:.4f}")
    print(f"Reranked MRR:       {rerank_mrr:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
