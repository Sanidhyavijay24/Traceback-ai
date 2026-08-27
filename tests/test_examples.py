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


def test_retrieval_failure_rag_example(tmp_path, monkeypatch):
    """Verify examples/retrieval_failure_rag.py runs and persists trace with retrieval blamed."""
    from tracebackai.blame import blame_run

    db_file = tmp_path / "test_rag_fail.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    monkeypatch.setenv("TRACEBACK_FORCE_DEMO", "1")

    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "retrieval_failure_rag.py")
    res = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert res.returncode == 0
    assert "Trace recorded!" in res.stdout

    store = Store(db_path=str(db_file))
    runs = store.list_runs()
    assert len(runs) == 1

    blame_res = blame_run(runs[0]["run_id"], store=store)
    assert blame_res.primary_step is not None
    assert blame_res.primary_step.name == "search_knowledge_base"
    assert blame_res.blame_score > 0.70


def test_tool_agent_failure_example(tmp_path, monkeypatch):
    """Verify examples/tool_agent_failure.py runs and persists trace with tool blamed."""
    from tracebackai.blame import blame_run

    db_file = tmp_path / "test_tool_fail.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    monkeypatch.setenv("TRACEBACK_FORCE_DEMO", "1")

    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tool_agent_failure.py")
    res = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert res.returncode == 0
    assert "Trace recorded!" in res.stdout

    store = Store(db_path=str(db_file))
    runs = store.list_runs()
    assert len(runs) == 1

    blame_res = blame_run(runs[0]["run_id"], store=store)
    assert blame_res.primary_step is not None
    assert blame_res.primary_step.name == "query_stock_market_api"
    assert blame_res.blame_score >= 0.70


def test_healthy_customer_support_agent_example(tmp_path, monkeypatch):
    """Verify examples/healthy_customer_support_agent.py runs with low blame score."""
    from tracebackai.blame import blame_run

    db_file = tmp_path / "test_healthy_support.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    monkeypatch.setenv("TRACEBACK_FORCE_DEMO", "1")

    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "healthy_customer_support_agent.py")
    res = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert res.returncode == 0
    assert "Trace recorded!" in res.stdout

    store = Store(db_path=str(db_file))
    runs = store.list_runs()
    assert len(runs) == 1

    blame_res = blame_run(runs[0]["run_id"], store=store)
    assert blame_res.blame_score < 0.45


def test_cascading_failure_example(tmp_path, monkeypatch):
    """Verify examples/cascading_failure.py records 2 runs and detects diff regression."""
    from tracebackai.blame import diff_runs

    db_file = tmp_path / "test_cascading.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    monkeypatch.setenv("TRACEBACK_FORCE_DEMO", "1")

    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "cascading_failure.py")
    res = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert res.returncode == 0
    assert "Both traces recorded!" in res.stdout

    store = Store(db_path=str(db_file))
    runs = store.list_runs()
    assert len(runs) == 2

    # Oldest run is index 1, newest is index 0
    refusal_run_id = runs[0]["run_id"]
    healthy_run_id = runs[1]["run_id"]

    diff_res = diff_runs(healthy_run_id, refusal_run_id, store=store)
    assert diff_res.verdict == "REGRESSION"
    assert len(diff_res.regressed_steps) > 0
    assert "execute_instruction" in [step_b.name for (_, step_b, _) in diff_res.regressed_steps]
