"""
Tests for example pipelines in zero-secret demo mode.
"""

import os
import subprocess
import sys
import pytest

from tracebackai.store import Store


def test_simple_rag_example_runs_in_demo_mode(tmp_path, monkeypatch):
    """
    Verify examples/simple_rag.py runs in demo mode without GEMINI_API_KEY,
    exits with return code 0, and saves a valid execution trace.
    """
    db_file = tmp_path / "test_example_demo.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    monkeypatch.setenv("TRACEBACK_FORCE_DEMO", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "simple_rag.py")
    res = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert res.returncode == 0
    assert "[demo mode: set GEMINI_API_KEY in .env to use live Gemini model]" in res.stdout
    assert "Result: Retrieval-augmented generation (RAG)" in res.stdout

    store = Store(db_path=str(db_file))
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert loaded.pipeline_name == "answer"
    assert len(loaded.steps) == 4  # retrieve, build_prompt, call_llm, answer
