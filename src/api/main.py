import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embedding.embed import get_embedding_model, get_chroma_collection
from src.retrieval.bm25 import load_bm25_index
from src.retrieval.hybrid import dense_search, sparse_search, reciprocal_rank_fusion, FETCH_K
from src.retrieval.rerank import get_reranker_model, get_chunk_texts, rerank, TOP_K as RERANK_TOP_K
from src.context.context import build_context
from src.generation.generate import get_groq_client, generate_answer
from src.generation.cite import format_response
from src.verification.verify import get_anthropic_client, verify_answer
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model/client exactly once at server startup, stored on
    app.state and reused across every request. This is the whole point of
    the API layer: the standalone scripts in each stage reload models per
    call, which is fine for one-off testing but would be disastrous for a
    running server (BGE-M3 and the reranker are slow to load, as seen in
    the terminal output from earlier stages)."""
    logger.info("Loading models and clients...")
    app.state.embedding_model = get_embedding_model()
    app.state.chroma_collection = get_chroma_collection()
    app.state.bm25_index, app.state.bm25_chunk_ids = load_bm25_index()
    app.state.reranker_model = get_reranker_model()
    app.state.groq_client = get_groq_client()
    app.state.anthropic_client = get_anthropic_client()
    logger.info("Startup complete.")

    yield

    logger.info("Shutting down.")


app = FastAPI(title="LLM Research Assistant", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    references: list[dict]
    verification: list[dict]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full pipeline end to end: hybrid retrieval -> rerank ->
    context construction -> generation -> citation formatting -> verification.
    Composes the lower-level functions directly (rather than calling
    hybrid_search/rerank_search/answer_query/verify_query, which each
    reload their own models internally) so every step reuses the
    models/clients loaded once at startup."""
    logger.info(f"Received query: {request.query!r}")

    dense_ids = dense_search(
        request.query, app.state.embedding_model, app.state.chroma_collection, top_k=FETCH_K
    )
    sparse_ids = sparse_search(
        request.query, app.state.bm25_index, app.state.bm25_chunk_ids, top_k=FETCH_K
    )
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    candidate_ids = [chunk_id for chunk_id, _ in fused[:FETCH_K]]
    logger.info(f"Retrieved {len(candidate_ids)} candidates (dense={len(dense_ids)}, sparse={len(sparse_ids)})")

    chunk_texts_by_id = get_chunk_texts(candidate_ids)
    reranked = rerank(
        request.query, candidate_ids, chunk_texts_by_id, app.state.reranker_model, top_k=RERANK_TOP_K
    )
    final_chunk_ids = [chunk_id for chunk_id, _ in reranked]
    logger.info(f"Reranked down to top {len(final_chunk_ids)} chunks")

    context_string, citation_map = build_context(final_chunk_ids)

    try:
        answer_text = generate_answer(request.query, context_string, app.state.groq_client)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="The answer generation service is temporarily unavailable (rate limited or down). Please try again shortly.",
        )

    response = format_response(answer_text, citation_map)

    try:
        response["verification"] = verify_answer(answer_text, citation_map, app.state.anthropic_client)
    except Exception as e:
        logger.error(f"Citation verification failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="The citation verification service is temporarily unavailable (rate limited or down). Please try again shortly.",
        )

    verdict_counts: dict[str, int] = {}
    for item in response["verification"]:
        verdict_counts[item["verdict"]] = verdict_counts.get(item["verdict"], 0) + 1
    logger.info(f"Verification summary: {verdict_counts}")

    return QueryResponse(**response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
