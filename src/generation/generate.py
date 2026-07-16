import os
import logging

from dotenv import load_dotenv
from groq import Groq

from src.retrieval.rerank import rerank_search
from src.context.context import build_context

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.15

SYSTEM_PROMPT = """You are a research assistant that answers questions about \
machine learning and LLM topics using only the context excerpts provided below, \
which come from research papers and technical documentation.

Rules:
1. Answer only using information contained in the provided context. Do not use \
outside knowledge, even if you are confident it is correct.
2. Every factual claim in your answer must be followed by the citation \
marker(s) from the context that support it, e.g. [1] or [1][2].
3. Never invent a citation number that does not appear in the provided context.
4. If the context does not contain enough information to answer the question, \
say so explicitly instead of guessing or filling gaps with outside knowledge.
5. Keep the answer focused and directly responsive to the question; do not pad \
it with unrelated content from the context."""


def get_groq_client() -> Groq:
    """Load the Groq API key from .env and return a client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def generate_answer(
    query: str,
    context_string: str,
    client: Groq,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> str:
    """Send the system prompt + context + question to Groq and return the
    model's answer text."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_string}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content


def answer_query(query: str) -> tuple[str, dict]:
    """Orchestrator: rerank -> build context -> generate answer. Returns the
    answer text plus the citation map ([N] -> title, source_url, chunk_ids)
    needed for the next stage (citation generation)."""
    reranked = rerank_search(query)
    chunk_ids = [chunk_id for chunk_id, _ in reranked]

    context_string, citation_map = build_context(chunk_ids)

    client = get_groq_client()
    answer = generate_answer(query, context_string, client)

    return answer, citation_map


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    answer, citation_map = answer_query(test_query)

    print(answer)
    print("\n--- citations ---")
    for number, info in citation_map.items():
        print(f"[{number}] {info['title']} ({info['source_url']})")
