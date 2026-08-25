"""
Tests for Traceback AI @trace decorator and TraceContext.
"""

import os
import pytest

from tracebackai.models import Trace
from tracebackai.store import Store
from tracebackai.tracer import TraceContext, get_active_trace, trace


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Ensure every test runs against an isolated SQLite database."""
    db_file = tmp_path / "test_traces.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    return db_file


def test_basic_pipeline_tracing():
    """Verify standard pipeline execution with child steps."""

    @trace(step_type="retrieval")
    def retrieve(query: str) -> list[str]:
        return [f"chunk for {query}"]

    @trace(step_type="llm")
    def call_llm(prompt: str) -> str:
        return f"answer to {prompt}"

    @trace(pipeline=True)
    def my_pipeline(query: str) -> str:
        chunks = retrieve(query)
        return call_llm(chunks[0])

    res = my_pipeline("what is RAG?")
    assert res == "answer to chunk for what is RAG?"

    # Active trace must be reset after root execution completes
    assert get_active_trace() is None

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["pipeline_name"] == "my_pipeline"
    assert runs[0]["step_count"] == 3

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 3
    assert loaded.steps[0].name == "retrieve"
    assert loaded.steps[0].step_type == "retrieval"
    assert loaded.steps[0].output == ["chunk for what is RAG?"]
    assert loaded.steps[0].token_count is not None
    assert loaded.steps[1].name == "call_llm"
    assert loaded.steps[2].name == "my_pipeline"
    assert loaded.final_output == "answer to chunk for what is RAG?"


def test_standalone_single_function_trace():
    """Verify standalone @trace without explicit pipeline=True creates a single-step trace."""

    @trace(step_type="tool", name="calculate_total")
    def compute(a: int, b: int) -> int:
        return a + b

    result = compute(10, 20)
    assert result == 30
    assert get_active_trace() is None

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["pipeline_name"] == "calculate_total"
    assert runs[0]["step_count"] == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 1
    assert loaded.steps[0].name == "calculate_total"
    assert loaded.steps[0].step_type == "tool"
    assert loaded.steps[0].output == 30


def test_nested_pipeline_warning_and_flattening():
    """Verify nested pipeline calls emit RuntimeWarning, flatten into root, and do not corrupt state."""

    @trace(pipeline=True)
    def inner_pipeline(x: str) -> str:
        return f"inner_{x}"

    @trace(pipeline=True)
    def outer_pipeline(x: str) -> str:
        return inner_pipeline(x)

    with pytest.warns(RuntimeWarning, match="Nested pipeline call detected"):
        result1 = outer_pipeline("test1")
    assert result1 == "inner_test1"
    assert get_active_trace() is None

    # Call a second time to ensure no leaked global state
    with pytest.warns(RuntimeWarning, match="Nested pipeline call detected"):
        result2 = outer_pipeline("test2")
    assert result2 == "inner_test2"
    assert get_active_trace() is None

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 2
    assert runs[0]["step_count"] == 2
    assert runs[1]["step_count"] == 2


def test_exception_captured_and_reraised():
    """Verify that exceptions are recorded in step.error and not swallowed."""

    @trace(step_type="tool")
    def faulty_tool(val: int):
        raise ValueError("Invalid input value")

    @trace(pipeline=True)
    def failing_pipeline():
        faulty_tool(42)

    with pytest.raises(ValueError, match="Invalid input value"):
        failing_pipeline()

    assert get_active_trace() is None

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 2
    assert loaded.steps[0].name == "faulty_tool"
    assert loaded.steps[0].error is not None
    assert "ValueError: Invalid input value" in loaded.steps[0].error
    assert loaded.steps[1].name == "failing_pipeline"
    assert loaded.steps[1].error is not None


def test_trace_context_manager():
    """Verify programmatic tracing using TraceContext and StepContext."""
    with TraceContext("manual_agent") as ctx:
        with ctx.step("step_fetch", step_type="retrieval", input="query text") as s:
            s.record(output=["doc1", "doc2"], metadata={"k": 2})

        with ctx.step("step_generate", step_type="llm", input="prompt text") as s:
            s.record(output="final response", cost_usd=0.002)

    assert get_active_trace() is None

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["pipeline_name"] == "manual_agent"

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 2
    assert loaded.steps[0].name == "step_fetch"
    assert loaded.steps[0].input == "query text"
    assert loaded.steps[0].output == ["doc1", "doc2"]
    assert loaded.steps[0].metadata["k"] == 2

    assert loaded.steps[1].name == "step_generate"
    assert loaded.steps[1].cost_usd == 0.002
    assert loaded.final_output == "final response"


def test_trace_context_exception_handling():
    """Verify TraceContext correctly records exceptions during context block."""
    with pytest.raises(RuntimeError, match="Step failure"):
        with TraceContext("error_agent") as ctx:
            with ctx.step("failing_step", step_type="tool") as s:
                raise RuntimeError("Step failure")

    assert get_active_trace() is None
    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1
    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 1
    assert loaded.steps[0].error is not None
    assert "RuntimeError: Step failure" in loaded.steps[0].error
