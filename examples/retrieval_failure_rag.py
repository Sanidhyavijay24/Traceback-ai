"""
Traceback AI - Example: RAG Pipeline with Retrieval Failure.

Demonstrates how Traceback AI diagnoses a failure where the retrieval step returns
off-topic chunks, causing downstream LLM degradation or refusal.
"""

import os
from pathlib import Path

FORCE_DEMO = os.environ.get("TRACEBACK_FORCE_DEMO") == "1"

if not FORCE_DEMO:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v

from tracebackai import trace

GEMINI_API_KEY = None if FORCE_DEMO else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
DEMO_MODE = not bool(GEMINI_API_KEY and GEMINI_API_KEY.strip())

if not DEMO_MODE:
    from tracebackai.integrations.gemini import TracedGemini
    client = TracedGemini(api_key=GEMINI_API_KEY)
else:
    client = None


@trace(step_type="retrieval")
def search_knowledge_base(query: str) -> list[str]:
    """
    Simulate a corrupted/misconfigured index returning off-topic documents.
    User asks about database connection pooling, but retriever returns tomato gardening tips.
    """
    return [
        "Gardening tomatoes requires well-drained loamy soil with a neutral pH balance between 6.0 and 6.8.",
        "Water tomato seedlings deeply twice a week at the base of the plant to prevent early blight.",
        "Pruning tomato suckers directs sunlight and essential nutrients to ripening fruit.",
    ]


@trace(step_type="prompt")
def construct_rag_prompt(query: str, context_docs: list[str]) -> str:
    """Format prompt with the retrieved documents."""
    context_str = "\n".join(context_docs)
    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"Answer using the context above."
    )


@trace(step_type="llm")
def generate_response(prompt: str) -> str:
    """Call LLM or return canned response in demo mode."""
    if DEMO_MODE:
        return (
            "Database connection pooling utilizes well-drained loamy soil with a balanced pH "
            "and deep watering twice a week to maximize query throughput and reduce connection latency."
        )
    resp = client.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return str(resp)


@trace(pipeline=True)
def run_rag_pipeline(user_query: str) -> str:
    """Execute the full RAG pipeline."""
    docs = search_knowledge_base(user_query)
    prompt = construct_rag_prompt(user_query, docs)
    return generate_response(prompt)


if __name__ == "__main__":
    query = "How does database connection pooling improve PostgreSQL throughput?"
    print(f"Running RAG pipeline for query: '{query}'")
    if DEMO_MODE:
        print("[demo mode: using simulated LLM response]\n")
    else:
        print("[live mode: calling Gemini API]\n")

    result = run_rag_pipeline(query)
    print(f"Output:\n{result}\n")
    print("-> Trace recorded! Run 'tb list' or 'tb blame <run_id>' to see root-cause failure attribution.")
