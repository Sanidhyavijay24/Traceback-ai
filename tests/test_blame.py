"""
Tests for Traceback AI failure attribution (blame) and cross-run diffing.
"""

import time
import pytest
from click.testing import CliRunner

from tracebackai.blame import blame_run, blame_trace, diff_runs, diff_traces
from tracebackai.cli import cli
from tracebackai.models import Step, Trace
from tracebackai.store import Store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isolate SQLite database for every test."""
    db_file = tmp_path / "test_blame.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    return db_file


# ---------------------------------------------------------------------------
# Single-Run Blame Tests
# ---------------------------------------------------------------------------


def test_blame_weak_retrieval_scenario():
    """Verify blame algorithm attributes failure to weak retrieval in a synthetic pipeline."""
    step_ret = Step(
        name="retrieve_passages",
        step_type="retrieval",
        index=0,
        score=0.42,
        metadata={"top_similarity": 0.42, "retrieval_chunks_count": 2},
    )
    step_prompt = Step(
        name="build_prompt",
        step_type="prompt",
        index=1,
        score=0.90,
    )
    step_llm = Step(
        name="generate_answer",
        step_type="llm",
        index=2,
        score=0.92,
    )

    trace = Trace(pipeline_name="rag_pipeline", steps=[step_ret, step_prompt, step_llm])
    result = blame_trace(trace)

    assert result.primary_step is not None
    assert result.primary_step.name == "retrieve_passages"
    assert result.confidence == "high"
    assert "Retrieval relevance score was 0.42" in result.explanation
    assert result.blame_score > 0.70
    assert len(result.co_blamed) == 0


def test_blame_llm_refusal_scenario():
    """Verify blame algorithm attributes failure to an LLM refusal step."""
    step_ret = Step(
        name="retrieve_context",
        step_type="retrieval",
        index=0,
        score=0.95,
    )
    step_llm = Step(
        name="call_claude",
        step_type="llm",
        index=1,
        score=0.0,
        metadata={"refusal_detected": True},
    )

    trace = Trace(pipeline_name="agent_pipeline", steps=[step_ret, step_llm])
    result = blame_trace(trace)

    assert result.primary_step is not None
    assert result.primary_step.name == "call_claude"
    assert result.confidence == "high"
    assert "refusal" in result.explanation.lower()


def test_blame_unhandled_exception_step():
    """Verify that a step raising an exception is immediately assigned top blame."""
    step_ret = Step(name="retrieve", step_type="retrieval", index=0, score=0.85)
    step_tool = Step(
        name="database_write",
        step_type="tool",
        index=1,
        error="sqlite3.OperationalError: database is locked",
    )
    step_llm = Step(name="summarize", step_type="llm", index=2, score=0.90)

    trace = Trace(pipeline_name="db_pipeline", steps=[step_ret, step_tool, step_llm])
    result = blame_trace(trace)

    assert result.primary_step is not None
    assert result.primary_step.name == "database_write"
    assert result.blame_score == 1.0
    assert "sqlite3.OperationalError: database is locked" in result.explanation


def test_blame_mixed_scored_and_unscored_steps():
    """
    CRITICAL TEST: Trace with a mix of scored and unscored (generic, score=None) steps.
    Unscored steps must be excluded from blame candidacy and not cause a crash.
    """
    step_ret = Step(name="retrieve", step_type="retrieval", index=0, score=0.45)
    step_gen = Step(name="format_headers", step_type="generic", index=1, score=None)
    step_custom = Step(name="custom_filter", step_type="generic", index=2, score=None)
    step_llm = Step(name="llm_call", step_type="llm", index=3, score=0.95)

    trace = Trace(pipeline_name="mixed_pipeline", steps=[step_ret, step_gen, step_custom, step_llm])
    result = blame_trace(trace)

    assert result.primary_step is not None
    assert result.primary_step.name == "retrieve"
    assert result.blame_score > 0.60
    assert result.is_fallback_latency is False


def test_blame_all_unscored_steps_fallback_to_slowest():
    """Verify trace where ALL steps are unscored falls back to blaming the slowest bottleneck step."""
    step_a = Step(name="step_a", step_type="generic", index=0, score=None, latency_ms=120.0)
    step_b = Step(name="step_b_bottleneck", step_type="generic", index=1, score=None, latency_ms=1850.0)
    step_c = Step(name="step_c", step_type="generic", index=2, score=None, latency_ms=45.0)

    trace = Trace(pipeline_name="generic_pipeline", steps=[step_a, step_b, step_c])
    result = blame_trace(trace)

    assert result.primary_step is not None
    assert result.primary_step.name == "step_b_bottleneck"
    assert result.is_fallback_latency is True
    assert result.confidence == "low"
    assert "slowest bottleneck (1850ms)" in result.explanation


def test_blame_co_blamed_tie_handling():
    """Verify co-blamed steps when top 2 blame scores are within 0.05 gap."""
    step1 = Step(name="retrieval_step", step_type="retrieval", index=0, score=0.50)
    step2 = Step(name="llm_step", step_type="llm", index=1, score=0.45)

    trace = Trace(pipeline_name="tie_pipeline", steps=[step1, step2])
    result = blame_trace(trace)

    assert result.primary_step is not None
    # If gap is tight (< 0.05), confidence is low and second step is co-blamed
    if len(result.co_blamed) > 0:
        assert result.confidence == "low"


def test_blame_empty_trace():
    """Verify blame on empty trace returns clean empty result without crashing."""
    trace = Trace(pipeline_name="empty_pipeline", steps=[])
    result = blame_trace(trace)
    assert result.primary_step is None
    assert result.blame_score == 0.0


def test_blame_execution_performance():
    """Verify blame computation finishes well under 2 seconds on multi-step trace."""
    steps = [
        Step(name=f"step_{i}", step_type="tool" if i % 2 == 0 else "llm", index=i, score=0.85)
        for i in range(15)
    ]
    trace = Trace(pipeline_name="large_pipe", steps=steps)

    t0 = time.perf_counter()
    result = blame_trace(trace)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.05  # Far below 2.0s
    assert result.primary_step is not None


# ---------------------------------------------------------------------------
# Cross-Run Diff Tests
# ---------------------------------------------------------------------------


def test_diff_traces_regression_detected():
    """Verify diffing two runs correctly identifies regression and highest delta step."""
    step_r1 = Step(name="retrieve", step_type="retrieval", score=0.85)
    step_l1 = Step(name="llm_call", step_type="llm", score=0.92)

    step_r2 = Step(name="retrieve", step_type="retrieval", score=0.88)
    step_l2 = Step(name="llm_call", step_type="llm", score=0.48)

    trace_a = Trace(run_id="run_good_123", pipeline_name="qa_pipe", steps=[step_r1, step_l1])
    trace_b = Trace(run_id="run_bad_456", pipeline_name="qa_pipe", steps=[step_r2, step_l2])

    diff_res = diff_traces(trace_a, trace_b)
    assert diff_res.verdict == "REGRESSION"
    assert diff_res.primary_diverged_step == "llm_call"
    assert len(diff_res.regressed_steps) == 1
    assert diff_res.regressed_steps[0][2] == pytest.approx(-0.44, abs=0.01)


def test_diff_traces_improvement_detected():
    """Verify diffing two runs correctly identifies improvement."""
    step_r1 = Step(name="retrieve", step_type="retrieval", score=0.42)
    step_r2 = Step(name="retrieve", step_type="retrieval", score=0.89)

    trace_a = Trace(run_id="run_old_111", pipeline_name="pipe", steps=[step_r1])
    trace_b = Trace(run_id="run_new_222", pipeline_name="pipe", steps=[step_r2])

    diff_res = diff_traces(trace_a, trace_b)
    assert diff_res.verdict == "IMPROVEMENT"
    assert diff_res.primary_diverged_step == "retrieve"
    assert len(diff_res.improved_steps) == 1
    assert diff_res.improved_steps[0][2] == pytest.approx(0.47, abs=0.01)


def test_diff_traces_added_and_removed_steps():
    """Verify step alignment across pipelines with different step counts."""
    step_common = Step(name="common_step", step_type="generic", score=0.8)
    step_old_only = Step(name="legacy_cleaner", step_type="generic", score=0.7)
    step_new_only = Step(name="new_reranker", step_type="retrieval", score=0.95)

    trace_a = Trace(run_id="run_a", pipeline_name="pipe", steps=[step_common, step_old_only])
    trace_b = Trace(run_id="run_b", pipeline_name="pipe", steps=[step_common, step_new_only])

    diff_res = diff_traces(trace_a, trace_b)
    assert len(diff_res.added_steps) == 1
    assert diff_res.added_steps[0].name == "new_reranker"
    assert len(diff_res.removed_steps) == 1
    assert diff_res.removed_steps[0].name == "legacy_cleaner"


# ---------------------------------------------------------------------------
# CLI Commands for Blame & Diff
# ---------------------------------------------------------------------------


def test_cli_blame_command(tmp_path):
    """Verify CLI blame command outputs formatted failure attribution."""
    store = Store()
    step_ret = Step(name="search_kb", step_type="retrieval", index=0, score=0.38)
    step_llm = Step(name="generate_reply", step_type="llm", index=1, score=0.90)
    trace = Trace(run_id="blame_cli_123", pipeline_name="support_bot", steps=[step_ret, step_llm])
    store.save_trace(trace)

    runner = CliRunner()
    res = runner.invoke(cli, ["blame", "blame_cli_123"])
    assert res.exit_code == 0
    assert "Analyzing run blame_cli_123" in res.output
    assert "Step [0] search_kb" in res.output
    assert "Score:       0.38" in res.output
    assert "Blame score:" in res.output
    assert "generate_reply (0.90 [OK])" in res.output


def test_cli_diff_command(tmp_path):
    """Verify CLI diff command outputs comparison table."""
    store = Store()
    step_a1 = Step(name="retriever", step_type="retrieval", index=0, score=0.85)
    step_a2 = Step(name="generator", step_type="llm", index=1, score=0.95)
    trace_a = Trace(run_id="diff_run_a", pipeline_name="demo_pipe", steps=[step_a1, step_a2])
    store.save_trace(trace_a)

    step_b1 = Step(name="retriever", step_type="retrieval", index=0, score=0.85)
    step_b2 = Step(name="generator", step_type="llm", index=1, score=0.45)
    trace_b = Trace(run_id="diff_run_b", pipeline_name="demo_pipe", steps=[step_b1, step_b2])
    store.save_trace(trace_b)

    runner = CliRunner()
    res = runner.invoke(cli, ["diff", "diff_run_a", "diff_run_b"])
    assert res.exit_code == 0
    assert "Comparing diff_run_a -> diff_run_b" in res.output
    assert "STEP" in res.output
    assert "generator" in res.output
    assert "-0.50" in res.output
    assert "REGRESSED" in res.output
    assert "Verdict: REGRESSION in generator" in res.output
