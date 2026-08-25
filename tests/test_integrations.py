"""
Tests for Traceback AI SDK integrations and CLI run eval gate.
"""

from unittest.mock import MagicMock
import pytest
from click.testing import CliRunner

from tracebackai.cli import cli
from tracebackai.integrations.anthropic import _wrap_messages_create, patch_anthropic
from tracebackai.integrations.gemini import TracedGemini
from tracebackai.integrations.langchain import TracebackCallbackHandler
from tracebackai.integrations.openai import _wrap_completions_create, patch_openai
from tracebackai.models import Trace
from tracebackai.store import Store
from tracebackai.tracer import TraceContext, trace


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isolate SQLite database for tests."""
    db_file = tmp_path / "test_integrations.db"
    monkeypatch.setenv("TRACEBACK_DB_PATH", str(db_file))
    return db_file


def test_gemini_wrapper_with_active_trace():
    """Verify Gemini wrapper records step into active trace."""
    gemini_client = TracedGemini.__new__(TracedGemini)
    gemini_client.api_key = "fake_key"
    gemini_client._sdk_type = "new"
    mock_models = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Hello from Gemini 2.5 Flash"
    mock_resp.usage_metadata.prompt_token_count = 12
    mock_resp.usage_metadata.candidates_token_count = 18
    mock_models.generate_content.return_value = mock_resp
    gemini_client._client = MagicMock()
    gemini_client._client.models = mock_models

    with TraceContext("gemini_pipeline") as ctx:
        resp = gemini_client.generate_content(
            model="gemini-2.5-flash",
            contents="Explain quantum computing in one sentence.",
        )

    assert resp == mock_resp
    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 1
    assert loaded.steps[0].name == "gemini_gemini-2.5-flash"
    assert loaded.steps[0].step_type == "llm"
    assert loaded.steps[0].output == "Hello from Gemini 2.5 Flash"
    assert loaded.steps[0].token_count == 30


def test_anthropic_wrapper_with_active_trace():
    """Verify Anthropic wrapper records step into active trace."""
    mock_orig = MagicMock()
    mock_msg = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Hello from Claude"
    mock_msg.content = [mock_block]
    mock_msg.usage.input_tokens = 15
    mock_msg.usage.output_tokens = 8
    mock_msg.stop_reason = "end_turn"
    mock_orig.return_value = mock_msg

    wrapped = _wrap_messages_create(mock_orig)

    with TraceContext("anthropic_pipeline") as ctx:
        resp = wrapped(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert resp == mock_msg
    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 1
    assert loaded.steps[0].name == "anthropic_claude-3-5-sonnet"
    assert loaded.steps[0].step_type == "llm"
    assert loaded.steps[0].output == "Hello from Claude"
    assert loaded.steps[0].token_count == 23
    assert loaded.steps[0].metadata["stop_reason"] == "end_turn"


def test_openai_wrapper_with_active_trace():
    """Verify OpenAI wrapper records step into active trace."""
    mock_orig = MagicMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello from GPT"
    mock_choice.finish_reason = "stop"
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 20
    mock_resp.usage.completion_tokens = 10
    mock_orig.return_value = mock_resp

    wrapped = _wrap_completions_create(mock_orig)

    with TraceContext("openai_pipeline") as ctx:
        resp = wrapped(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert resp == mock_resp
    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 1
    assert loaded.steps[0].name == "openai_gpt-4o"
    assert loaded.steps[0].step_type == "llm"
    assert loaded.steps[0].output == "Hello from GPT"
    assert loaded.steps[0].token_count == 30
    assert loaded.steps[0].metadata["finish_reason"] == "stop"


def test_langchain_callback_handler():
    """Verify LangChain TracebackCallbackHandler captures events."""
    from uuid import uuid4

    handler = TracebackCallbackHandler()
    llm_run_id = uuid4()
    ret_run_id = uuid4()

    with TraceContext("langchain_agent") as ctx:
        # Simulate retriever
        handler.on_retriever_start(
            {"name": "test_retriever"},
            "query text",
            run_id=ret_run_id,
        )
        mock_doc = MagicMock()
        mock_doc.page_content = "Document text snippet"
        handler.on_retriever_end([mock_doc], run_id=ret_run_id)

        # Simulate LLM
        handler.on_llm_start(
            {"name": "test_llm"},
            ["prompt text"],
            run_id=llm_run_id,
        )
        mock_resp = MagicMock()
        mock_gen = MagicMock()
        mock_gen.text = "Answer from model"
        mock_resp.generations = [[mock_gen]]
        handler.on_llm_end(mock_resp, run_id=llm_run_id)

    store = Store()
    runs = store.list_runs()
    assert len(runs) == 1

    loaded = store.load_trace(runs[0]["run_id"])
    assert len(loaded.steps) == 2
    assert loaded.steps[0].name == "test_retriever"
    assert loaded.steps[0].step_type == "retrieval"
    assert loaded.steps[0].output == ["Document text snippet"]

    assert loaded.steps[1].name == "test_llm"
    assert loaded.steps[1].step_type == "llm"
    assert loaded.steps[1].output == "Answer from model"


def test_cli_run_eval_gate_passes(tmp_path):
    """Verify traceback run exits 0 when blame score is below threshold."""
    import textwrap

    script_file = tmp_path / "healthy_pipeline.py"
    script_file.write_text(
        textwrap.dedent(
            """
            import os
            from tracebackai import trace

            @trace(step_type="retrieval")
            def fetch(q: str):
                return [
                    "Retrieval-augmented generation (RAG) connects search retrieval to generative models.",
                    "Retrieval-augmented generation grounds LLM responses with retrieved factual context."
                ]

            @trace(step_type="llm")
            def gen(prompt: str):
                return (
                    "Retrieval-augmented generation (RAG) combines search retrieval with "
                    "generative AI systems to produce grounded and factually accurate answers."
                )

            @trace(pipeline=True)
            def run_app():
                docs = fetch("What is retrieval-augmented generation?")
                return gen(docs[0])

            if __name__ == "__main__":
                run_app()
            """
        ).strip()
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["run", str(script_file), "--fail-on-blame", "0.7"])
    assert res.exit_code == 0
    assert "Eval Gate Passed" in res.output


def test_cli_run_eval_gate_fails_on_blame(tmp_path):
    """Verify traceback run exits 1 when blame score exceeds threshold."""
    import textwrap

    script_file = tmp_path / "failing_pipeline.py"
    script_file.write_text(
        textwrap.dedent(
            """
            from tracebackai import trace

            @trace(step_type="tool")
            def bad_tool():
                raise RuntimeError("Critical API failure")

            @trace(pipeline=True)
            def run_app():
                try:
                    bad_tool()
                except Exception:
                    pass

            if __name__ == "__main__":
                run_app()
            """
        ).strip()
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["run", str(script_file), "--fail-on-blame", "0.5"])
    assert res.exit_code == 1
    assert "Eval Gate Failed" in res.output
