import os
import re
import json
import logging

from dotenv import load_dotenv
from anthropic import Anthropic

from src.chunking.chunk import split_into_sentences
from src.context.context import get_chunks_by_id
from src.generation.generate import answer_query
from src.generation.cite import format_response

logger = logging.getLogger(__name__)

load_dotenv()

JUDGE_MODEL_NAME = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM_PROMPT = """You are a fact-checking judge. You will be given a \
CLAIM (one sentence from an AI-generated answer) and EVIDENCE (source text the \
claim is supposed to be based on).

Decide whether the evidence supports the claim:
- "supported": the evidence directly backs up the claim.
- "partially_supported": the evidence is related but doesn't fully back up \
every detail in the claim.
- "unsupported": the evidence does not back up the claim, or the claim \
contains information not present in the evidence at all.

Respond with ONLY a JSON object, no other text, in this exact form:
{"verdict": "supported" | "partially_supported" | "unsupported", "reason": "<one sentence explanation>"}"""


def get_anthropic_client() -> Anthropic:
    """Load the Anthropic API key from .env and return a client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found. Add it to your .env file.")
    return Anthropic(api_key=api_key)


def extract_citation_numbers(sentence: str) -> set[int]:
    """Find every [N] marker present in a single sentence."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", sentence)}


def get_evidence_text(citation_numbers: set[int], citation_map: dict[int, dict]) -> str:
    """Concatenate the chunk_text of every chunk backing the given citation
    numbers, so the judge sees all evidence that citation could be drawing from."""
    chunk_ids: list[str] = []
    for number in citation_numbers:
        info = citation_map.get(number)
        if info:
            chunk_ids.extend(info["chunk_ids"])

    chunks_by_id = get_chunks_by_id(chunk_ids)
    return "\n\n".join(
        chunks_by_id[chunk_id].chunk_text for chunk_id in chunk_ids if chunk_id in chunks_by_id
    )


def judge_claim(claim: str, evidence_text: str, client: Anthropic) -> dict:
    """Ask Claude Haiku whether evidence_text supports claim. Returns a dict
    with 'verdict' and 'reason', defaulting to 'unknown' if parsing fails."""
    response = client.messages.create(
        model=JUDGE_MODEL_NAME,
        max_tokens=200,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_text}"}],
    )
    raw_text = response.content[0].text

    # Claude sometimes wraps the JSON in a markdown code fence despite being
    # told not to. Pull out the {...} substring rather than trusting the
    # model followed the "no other text" instruction literally.
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        logger.error(f"No JSON object found in judge response: {raw_text}")
        return {"verdict": "unknown", "reason": "Judge response could not be parsed."}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.error(f"Failed to parse judge response as JSON: {raw_text}")
        return {"verdict": "unknown", "reason": "Judge response could not be parsed."}


def verify_answer(answer_text: str, citation_map: dict[int, dict]) -> list[dict]:
    """Split the answer into sentences; for each sentence carrying a citation,
    verify it against the evidence text for that citation. Sentences with no
    citation are skipped, since there's nothing to check them against."""
    client = get_anthropic_client()
    sentences = split_into_sentences(answer_text)

    results = []
    for sentence in sentences:
        citation_numbers = extract_citation_numbers(sentence)
        if not citation_numbers:
            continue

        evidence_text = get_evidence_text(citation_numbers, citation_map)
        if not evidence_text:
            logger.warning(f"No evidence text found for citations {citation_numbers} in sentence: {sentence}")
            continue

        try:
            verdict = judge_claim(sentence, evidence_text, client)
        except Exception as e:
            logger.error(f"Failed to verify sentence: {sentence}. Error: {e}")
            verdict = {"verdict": "error", "reason": str(e)}

        results.append({
            "sentence": sentence,
            "citations": sorted(citation_numbers),
            "verdict": verdict.get("verdict", "unknown"),
            "reason": verdict.get("reason", ""),
        })

    return results


def verify_query(query: str) -> dict:
    """Orchestrator: generate the answer, format it with clean references,
    then verify every cited sentence against its full evidence text (which
    needs the un-filtered citation_map, not the display-only references list)."""
    answer_text, citation_map = answer_query(query)
    response = format_response(answer_text, citation_map)
    response["verification"] = verify_answer(answer_text, citation_map)
    return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    response = verify_query(test_query)

    print(response["answer"])
    print("\n--- verification ---")
    for result in response["verification"]:
        print(f"[{result['citations']}] {result['verdict']}: {result['reason']}")
        print(f"  sentence: {result['sentence']}\n")
