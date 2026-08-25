"""
Traceback AI - Example Standalone RAG Pipeline.

Demonstrates tracing a multi-step RAG workflow using Google Gemini.
Runs in zero-secret demo mode if GEMINI_API_KEY is not configured in .env.
"""

import os
from pathlib import Path

FORCE_DEMO = os.environ.get("TRACEBACK_FORCE_DEMO") == "1"

# Load .env if present and not forced into demo mode
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


def mock_retrieve(query: str) -> list[str]:
    """Simulate vector store retrieval returning relevant document chunks."""
    return [
        "Retrieval-augmented generation (RAG) combines search retrieval with generative LLMs.",
        "RAG pipelines perform context retrieval to augment generation with domain knowledge.",
        "Retrieval-augmented generation minimizes hallucinations by grounding model answers in retrieved text.",
    ]


@trace(step_type="retrieval")
def retrieve(query: str) -> list[str]:
    """Retrieve relevant context chunks for the query."""
    return mock_retrieve(query)


@trace(step_type="prompt")
def build_prompt(query: str, chunks: list[str]) -> str:
    """Construct prompt with retrieved context."""
    formatted_context = "\n".join(chunks)
    return (
        f"Context information is below:\n"
        f"---------------------\n"
        f"{formatted_context}\n"
        f"---------------------\n"
        f"Given the context above, answer the question: {query}\n"
        f"Answer concisely in 1-2 sentences."
    )


@trace(step_type="llm")
def call_llm(prompt: str) -> str:
    """Generate answer from Gemini or return canned response in demo mode."""
    if DEMO_MODE:
        return (
            "Retrieval-augmented generation (RAG) combines a retriever that fetches "
            "relevant context with a generator that produces an answer conditioned on "
            "that context. [demo mode - no GEMINI_API_KEY configured]"
        )
    resp = client.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return str(resp)


@trace(pipeline=True)
def answer(query: str) -> str:
    """Root RAG pipeline."""
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)
    return call_llm(prompt)


if __name__ == "__main__":
    if DEMO_MODE:
        print("[demo mode: set GEMINI_API_KEY in .env to use live Gemini model]\n")
    else:
        print("[live mode: using Gemini API]\n")
    result = answer("What is retrieval-augmented generation?")
    print(f"Result: {result}")
