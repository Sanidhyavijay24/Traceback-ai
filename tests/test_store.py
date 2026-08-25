"""
Tests for Traceback AI SQLite store and serialization.
"""

from datetime import datetime
import pytest

from tracebackai.models import Step, Trace
from tracebackai.store import MAX_STRING_LENGTH, TRUNCATION_MARKER, Store


@pytest.fixture
def temp_store(tmp_path):
    """Fixture providing an isolated SQLite store for testing."""
    db_file = tmp_path / "test_traces.db"
    return Store(db_path=str(db_file))


def test_store_save_and_load_roundtrip(temp_store):
    """Verify that a complete trace saves and reloads with all fields intact."""
    step1 = Step(
        name="step_one",
        step_type="retrieval",
        index=0,
        input={"query": "hello world"},
        output=["chunk1", "chunk2"],
        latency_ms=45.2,
        token_count=120,
        cost_usd=0.0002,
        metadata={"model": "test-embed"},
        score=0.88,
    )
    step2 = Step(
        name="step_two",
        step_type="llm",
        index=1,
        input="prompt text",
        output="response text",
        latency_ms=300.5,
        token_count=500,
        cost_usd=0.0015,
        metadata={"temperature": 0.7},
        score=0.95,
        error=None,
    )

    trace = Trace(
        pipeline_name="sample_pipeline",
        steps=[step1, step2],
        final_output="response text",
        metadata={"user_id": "u123"},
    )
    trace.end_ts = trace.start_ts + 0.35

    temp_store.save_trace(trace)

    loaded = temp_store.load_trace(trace.run_id)
    assert loaded.run_id == trace.run_id
    assert loaded.pipeline_name == "sample_pipeline"
    assert loaded.final_output == "response text"
    assert loaded.metadata == {"user_id": "u123"}
    assert len(loaded.steps) == 2

    assert loaded.steps[0].name == "step_one"
    assert loaded.steps[0].step_type == "retrieval"
    assert loaded.steps[0].input == {"query": "hello world"}
    assert loaded.steps[0].output == ["chunk1", "chunk2"]
    assert loaded.steps[0].token_count == 120
    assert loaded.steps[0].score == 0.88

    assert loaded.steps[1].name == "step_two"
    assert loaded.steps[1].output == "response text"
    assert loaded.steps[1].cost_usd == 0.0015


def test_store_complex_serialization(temp_store):
    """Verify serialization of numpy arrays, datetimes, and large string truncation."""
    np = pytest.importorskip("numpy")
    now = datetime(2026, 8, 25, 4, 0, 0)
    arr = np.array([1.0, 2.5, 3.8])
    huge_str = "a" * (MAX_STRING_LENGTH + 500)

    step = Step(
        name="complex_step",
        input={"array": arr, "time": now},
        output={"large": huge_str},
    )
    trace = Trace(pipeline_name="complex_pipe", steps=[step])

    temp_store.save_trace(trace)
    loaded = temp_store.load_trace(trace.run_id)

    assert loaded.steps[0].input["array"] == [1.0, 2.5, 3.8]
    assert loaded.steps[0].input["time"] == now.isoformat()
    assert loaded.steps[0].output["large"].endswith(TRUNCATION_MARKER)
    assert len(loaded.steps[0].output["large"]) == MAX_STRING_LENGTH + len(TRUNCATION_MARKER)


def test_store_list_and_delete_runs(temp_store):
    """Verify listing and deleting runs from the SQLite database."""
    trace_a = Trace(pipeline_name="pipe_a")
    trace_b = Trace(pipeline_name="pipe_b")
    temp_store.save_trace(trace_a)
    temp_store.save_trace(trace_b)

    runs = temp_store.list_runs()
    assert len(runs) == 2

    pipe_a_runs = temp_store.list_runs(pipeline_name="pipe_a")
    assert len(pipe_a_runs) == 1
    assert pipe_a_runs[0]["run_id"] == trace_a.run_id

    temp_store.delete_run(trace_a.run_id)
    runs_after_delete = temp_store.list_runs()
    assert len(runs_after_delete) == 1
    assert runs_after_delete[0]["run_id"] == trace_b.run_id


def test_store_load_nonexistent_raises(temp_store):
    """Verify loading an invalid run_id raises ValueError."""
    with pytest.raises(ValueError, match="Trace run not found"):
        temp_store.load_trace("nonexistent_id_123")
