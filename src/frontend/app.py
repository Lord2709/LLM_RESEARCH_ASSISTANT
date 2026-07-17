import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

# icon, text color, background color per verdict
VERDICT_STYLES = {
    "supported": ("✅", "#166534", "#dcfce7"),
    "partially_supported": ("⚠️", "#854d0e", "#fef9c3"),
    "unsupported": ("❌", "#991b1b", "#fee2e2"),
    "unknown": ("❓", "#374151", "#f3f4f6"),
    "error": ("❌", "#991b1b", "#fee2e2"),
}


def query_backend(query: str) -> dict:
    """Call the FastAPI /query endpoint and return the parsed JSON response.
    Raises requests.exceptions.RequestException with the backend's actual
    `detail` message (e.g. "rate limited, try again shortly") rather than a
    generic HTTP status line, if the backend returned one."""
    response = requests.post(API_URL, json={"query": query}, timeout=120)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise requests.exceptions.RequestException(detail or str(e)) from e
    return response.json()


def verdict_badge(verdict: str) -> str:
    icon, color, bg = VERDICT_STYLES.get(verdict, VERDICT_STYLES["unknown"])
    label = verdict.replace("_", " ").upper()
    return (
        f'<span style="background-color:{bg}; color:{color}; padding:2px 10px; '
        f'border-radius:12px; font-size:0.8em; font-weight:600; white-space:nowrap;">'
        f"{icon} {label}</span>"
    )


def render_response(result: dict) -> None:
    st.markdown(result["answer"])

    if result["references"]:
        st.markdown("**References**")
        for ref in result["references"]:
            st.markdown(f"[{ref['number']}] [{ref['title']}]({ref['source_url']})")

    verification = result.get("verification", [])
    if verification:
        verdict_counts: dict[str, int] = {}
        for item in verification:
            verdict_counts[item["verdict"]] = verdict_counts.get(item["verdict"], 0) + 1

        st.markdown("**Verification summary**")
        cols = st.columns(len(verdict_counts))
        for col, (verdict, count) in zip(cols, verdict_counts.items()):
            icon, color, bg = VERDICT_STYLES.get(verdict, VERDICT_STYLES["unknown"])
            label = verdict.replace("_", " ").title()
            col.markdown(
                f'<div style="text-align:center; background-color:{bg}; border-radius:10px; padding:12px 4px;">'
                f'<span style="font-size:1.4em;">{icon}</span><br>'
                f'<span style="font-size:1.3em; font-weight:700; color:{color};">{count}</span><br>'
                f'<span style="font-size:0.75em; color:{color};">{label}</span></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Sentence-by-sentence verification"):
            for item in verification:
                st.markdown(f'{verdict_badge(item["verdict"])} &nbsp; {item["sentence"]}', unsafe_allow_html=True)
                st.caption(item["reason"])
                st.divider()


st.set_page_config(page_title="LLM Research Assistant", page_icon="🔬", layout="wide")

with st.sidebar:
    st.title("🔬 LLM Research Assistant")
    st.caption("Cited, verified answers over a corpus of LLM/RAG/agentic-AI research papers.")
    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. Hybrid search (BM25 + dense embeddings, fused via RRF)\n"
        "2. Cross-encoder reranking\n"
        "3. Answer generation (Groq Llama-3.3-70B)\n"
        "4. Independent citation verification (Claude Haiku 4.5)"
    )

st.title("Ask a research question")

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["query"])
    with st.chat_message("assistant"):
        render_response(entry["result"])

query = st.chat_input("Ask a question about ML/LLM research...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, generating, and verifying..."):
            try:
                result = query_backend(query)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend API: {e}")
                result = None

        if result:
            render_response(result)
            st.session_state.history.append({"query": query, "result": result})
