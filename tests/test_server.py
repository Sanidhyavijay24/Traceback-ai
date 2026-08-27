"""
Tests for Traceback AI local Mission Control Dashboard server and REST APIs.
"""

import json
import threading
import time
import urllib.request
import pytest

from tracebackai.dashboard.server import create_server
from tracebackai.models import Step, Trace
from tracebackai.store import Store


@pytest.fixture
def test_store_with_data(tmp_path):
    """Create an isolated test store with sample healthy, failing, and comparison traces."""
    db_file = tmp_path / "test_dashboard.db"
    store = Store(db_path=str(db_file))

    # 1. Healthy trace
    trace_healthy = Trace(
        run_id="run_healthy_001",
        pipeline_name="customer_search",
        steps=[
            Step(
                name="embed_query",
                step_type="generic",
                input={"query": "test query"},
                output={"embedding": [0.1, 0.2]},
                latency_ms=45.0,
                score=0.95,
            ),
            Step(
                name="retrieve_chunks",
                step_type="retrieval",
                input={"query": "test query"},
                output=["chunk 1", "chunk 2"],
                latency_ms=120.0,
                score=0.88,
            ),
            Step(
                name="generate_answer",
                step_type="llm",
                input={"context": "chunk 1"},
                output="Here is the answer.",
                latency_ms=450.0,
                score=0.92,
            ),
        ],
        final_output="Here is the answer.",
    )
    trace_healthy.end_ts = trace_healthy.start_ts + 0.615
    store.save_trace(trace_healthy)

    # 2. Failing / blamed trace (weak retrieval)
    trace_fault = Trace(
        run_id="run_fault_002",
        pipeline_name="customer_search",
        steps=[
            Step(
                name="embed_query",
                step_type="generic",
                input={"query": "complex question"},
                output={"embedding": [0.0, 0.1]},
                latency_ms=50.0,
                score=0.90,
            ),
            Step(
                name="retrieve_chunks",
                step_type="retrieval",
                input={"query": "complex question"},
                output=[],
                metadata={"retrieval_chunks_count": 0},
                latency_ms=110.0,
                score=0.20,
            ),
            Step(
                name="generate_answer",
                step_type="llm",
                input={"context": ""},
                output="I do not know.",
                latency_ms=300.0,
                score=0.45,
            ),
        ],
        final_output="I do not know.",
    )
    trace_fault.end_ts = trace_fault.start_ts + 0.46
    store.save_trace(trace_fault)

    return store


@pytest.fixture
def running_server(test_store_with_data):
    """Start dashboard HTTP server in background thread for testing."""
    server, port = create_server(host="127.0.0.1", port=0, store=test_store_with_data)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)  # allow socket binding

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, bytes, str]:
    """Helper to perform HTTP GET."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def test_dashboard_static_and_spa(running_server):
    """Verify index.html, static assets, and SPA fallback routes load correctly."""
    # 1. Root index.html
    status, body, content_type = _get(f"{running_server}/")
    assert status == 200
    assert "text/html" in content_type
    assert "MISSION CONTROL" in body.decode("utf-8")

    # 2. CSS stylesheet
    status, body, content_type = _get(f"{running_server}/static/styles.css")
    assert status == 200
    assert "text/css" in content_type
    assert "--accent-fault" in body.decode("utf-8")

    # 3. JS bundle
    status, body, content_type = _get(f"{running_server}/static/app.js")
    assert status == 200
    assert "javascript" in content_type
    assert "app" in body.decode("utf-8")

    # 4. SPA route fallback
    status, body, _ = _get(f"{running_server}/runs/run_healthy_001")
    assert status == 200
    assert "MISSION CONTROL" in body.decode("utf-8")


def test_dashboard_api_stats(running_server):
    """Verify /api/stats returns trace metrics."""
    status, body, _ = _get(f"{running_server}/api/stats")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["total_runs"] == 2
    assert data["total_steps"] == 6
    assert "customer_search" in data["pipelines"]


def test_dashboard_api_runs(running_server):
    """Verify /api/runs returns enriched run summaries."""
    status, body, _ = _get(f"{running_server}/api/runs")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    runs = data["runs"]
    assert len(runs) == 2

    # Check that healthy and blamed runs are tagged properly
    run_map = {r["run_id"]: r for r in runs}
    assert run_map["run_healthy_001"]["status"] == "healthy"
    assert run_map["run_fault_002"]["status"] == "blame"
    assert run_map["run_fault_002"]["blame"]["primary_step_name"] == "retrieve_chunks"


def test_dashboard_api_run_detail(running_server):
    """Verify /api/runs/<id> returns complete trace data and blame payload."""
    status, body, _ = _get(f"{running_server}/api/runs/run_fault_002")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["run_id"] == "run_fault_002"
    assert data["pipeline_name"] == "customer_search"
    assert len(data["steps"]) == 3
    assert data["blame"]["primary_step_name"] == "retrieve_chunks"
    assert data["blame"]["blame_score"] > 0.5


def test_dashboard_api_diff(running_server):
    """Verify /api/diff returns step delta comparisons and verdict."""
    url = f"{running_server}/api/diff?a=run_healthy_001&b=run_fault_002"
    status, body, _ = _get(url)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["run_a"] == "run_healthy_001"
    assert data["run_b"] == "run_fault_002"
    assert data["verdict"] == "REGRESSION"
    assert data["primary_diverged_step"] == "retrieve_chunks"
    assert len(data["regressed_steps"]) >= 1


def test_dashboard_api_delete(running_server, test_store_with_data):
    """Verify DELETE /api/runs/<id> removes a run from store."""
    req = urllib.request.Request(f"{running_server}/api/runs/run_healthy_001", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True

    # Confirm it's gone
    remaining = test_store_with_data.list_runs()
    assert len(remaining) == 1
    assert remaining[0]["run_id"] == "run_fault_002"
