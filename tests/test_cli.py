"""
Tests for Traceback AI Click CLI commands.
"""

from click.testing import CliRunner
import pytest

from tracebackai.cli import cli
from tracebackai.models import Step, Trace
from tracebackai.store import Store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Ensure tests run against an isolated SQLite database."""
    db_file = tmp_path / "test_cli.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    return db_file


def test_cli_list_empty():
    """Verify cli output when no runs are stored."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "No trace runs found." in result.output


def test_cli_list_and_show():
    """Verify list and show commands display stored trace data correctly."""
    store = Store()
    step = Step(
        name="retrieve_docs",
        step_type="retrieval",
        input={"query": "test query"},
        output=["doc a", "doc b"],
        latency_ms=150.0,
        token_count=85,
        cost_usd=0.0005,
        score=0.85,
    )
    trace = Trace(
        run_id="abc123456789",
        pipeline_name="search_pipeline",
        steps=[step],
        final_output="done",
    )
    trace.end_ts = trace.start_ts + 0.15
    store.save_trace(trace)

    runner = CliRunner()
    list_res = runner.invoke(cli, ["list"])
    assert list_res.exit_code == 0
    assert "abc123456789" in list_res.output
    assert "search_pipeline" in list_res.output

    # List with pipeline filter
    list_filtered = runner.invoke(cli, ["list", "--pipeline", "search_pipeline"])
    assert list_filtered.exit_code == 0
    assert "abc123456789" in list_filtered.output

    # List with non-matching filter
    list_empty = runner.invoke(cli, ["list", "--pipeline", "other_pipe"])
    assert list_empty.exit_code == 0
    assert "No trace runs found." in list_empty.output

    # Show trace
    show_res = runner.invoke(cli, ["show", "abc123456789"])
    assert show_res.exit_code == 0
    assert "Run: abc123456789" in show_res.output
    assert "search_pipeline" in show_res.output
    assert "retrieve_docs" in show_res.output
    assert "retrieval" in show_res.output
    assert "tokens=85" in show_res.output
    assert "score=0.85 [OK]" in show_res.output
    assert "cost=$0.0005" in show_res.output


def test_cli_show_verbose_and_errors():
    """Verify show --verbose flag and error reporting."""
    store = Store()
    step = Step(
        name="failing_step",
        step_type="tool",
        input={"key": "val"},
        output=None,
        error="Traceback (most recent call last):\n  RuntimeError: DB connection lost",
    )
    trace = Trace(
        run_id="err123456789",
        pipeline_name="error_pipe",
        steps=[step],
    )
    store.save_trace(trace)

    runner = CliRunner()
    show_res = runner.invoke(cli, ["show", "err123456789", "--verbose"])
    assert show_res.exit_code == 0
    assert "ERROR: RuntimeError: DB connection lost" in show_res.output
    assert '{"key": "val"}' in show_res.output


def test_cli_show_not_found():
    """Verify show command displays error on missing run_id."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "missing_id"])
    assert result.exit_code == 0
    assert "Error: Run ID 'missing_id' not found." in result.output


def test_cli_serve_help():
    """Verify traceback serve --help displays options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" in result.output
    assert "--no-browser" in result.output
