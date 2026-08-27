"""
Traceback AI - Local Mission Control Web Dashboard Server.

Provides a lightweight HTTP server reading directly from SQLite Store
and serving the mission control web interface and JSON telemetry APIs.
"""

from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import socket
import threading
from typing import Any, Optional
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser

from tracebackai.blame import blame_trace, diff_traces
from tracebackai.models import Step, Trace
from tracebackai.store import Store

STATIC_DIR = Path(__file__).parent / "static"


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Mission Control Dashboard & Telemetry APIs."""

    def __init__(self, *args: Any, store: Optional[Store] = None, **kwargs: Any) -> None:
        self.store = store or Store()
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Custom request logging to keep console clean and quiet."""
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response with appropriate headers."""
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, file_path: Path, status: int = 200) -> None:
        """Send static file with inferred MIME type."""
        if not file_path.is_file():
            self.send_error(404, "File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(status)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type or "json" in mime_type else mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_DELETE(self) -> None:
        """Handle DELETE requests (e.g. deleting a trace)."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            try:
                self.store.delete_run(run_id)
                self._send_json({"success": True, "deleted": run_id})
            except Exception as e:
                self._send_json({"error": "delete_failed", "message": str(e)}, status=500)
            return

        self._send_json({"error": "not_found", "message": f"Route not found: {self.path}"}, status=404)

    def do_GET(self) -> None:
        """Route GET requests to API handlers or static files."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API routes
        if path.startswith("/api/"):
            self._handle_api(path, query)
            return

        # 2. Static file serving
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):].lstrip("/")
            file_path = (STATIC_DIR / rel_path).resolve()
            # Ensure path traversal protection
            if STATIC_DIR.resolve() in file_path.parents or file_path == STATIC_DIR.resolve():
                self._send_file(file_path)
                return
            self.send_error(403, "Access Denied")
            return

        # 3. Favicon or root static assets
        if path == "/favicon.ico":
            icon_path = STATIC_DIR / "favicon.ico"
            if icon_path.exists():
                self._send_file(icon_path)
                return
            self.send_response(204)
            self.end_headers()
            return

        # 4. SPA Fallback: Serve index.html for all page routes
        index_file = STATIC_DIR / "index.html"
        self._send_file(index_file)

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        """Dispatch API endpoints."""
        clean_path = path.rstrip("/")

        # GET /api/stats
        if clean_path == "/api/stats":
            self._api_get_stats()
            return

        # GET /api/pipelines
        if clean_path == "/api/pipelines":
            self._api_get_pipelines()
            return

        # GET /api/runs
        if clean_path == "/api/runs":
            self._api_get_runs(query)
            return

        # GET /api/runs/<run_id>
        if clean_path.startswith("/api/runs/"):
            run_id = clean_path[len("/api/runs/"):]
            self._api_get_run_detail(run_id)
            return

        # GET /api/blame/<run_id>
        if clean_path.startswith("/api/blame/"):
            run_id = clean_path[len("/api/blame/"):]
            self._api_get_blame(run_id)
            return

        # GET /api/diff?a=<id1>&b=<id2>
        if clean_path == "/api/diff":
            self._api_get_diff(query)
            return

        self._send_json({"error": "not_found", "message": f"Unknown API endpoint: {path}"}, status=404)

    def _api_get_stats(self) -> None:
        """Return global telemetry stats."""
        try:
            runs = self.store.list_runs(limit=1000)
            total_runs = len(runs)
            total_steps = sum(r.get("step_count", 0) for r in runs)
            pipelines = sorted(list({r.get("pipeline_name") or "unnamed" for r in runs}))

            # Quick fault analysis across recent runs
            fault_runs_count = 0
            for r in runs[:50]:
                try:
                    trace = self.store.load_trace(r["run_id"])
                    blame_res = blame_trace(trace)
                    if blame_res.blame_score > 0.35 or any(s.error for s in trace.steps):
                        fault_runs_count += 1
                except Exception:
                    pass

            self._send_json({
                "db_path": str(self.store.db_path),
                "total_runs": total_runs,
                "total_steps": total_steps,
                "pipelines_count": len(pipelines),
                "pipelines": pipelines,
                "recent_fault_count": fault_runs_count,
            })
        except Exception as e:
            self._send_json({"error": "stats_error", "message": str(e)}, status=500)

    def _api_get_pipelines(self) -> None:
        """Return list of distinct pipeline names."""
        try:
            runs = self.store.list_runs(limit=1000)
            pipelines = sorted(list({r.get("pipeline_name") or "unnamed" for r in runs}))
            self._send_json({"pipelines": pipelines})
        except Exception as e:
            self._send_json({"error": "pipelines_error", "message": str(e)}, status=500)

    def _api_get_runs(self, query: dict[str, list[str]]) -> None:
        """Return list of runs with blame status, error indicators, and metrics."""
        pipeline = query.get("pipeline", [None])[0]
        if pipeline == "all" or pipeline == "":
            pipeline = None

        limit = 50
        if "limit" in query:
            try:
                limit = int(query["limit"][0])
            except ValueError:
                pass

        try:
            runs = self.store.list_runs(pipeline_name=pipeline, limit=limit)
            enriched_runs: list[dict[str, Any]] = []

            for r in runs:
                run_id = r["run_id"]
                # Load trace to get status & blame indicator
                status = "healthy"
                blame_info: Optional[dict[str, Any]] = None

                try:
                    trace = self.store.load_trace(run_id)
                    # Check for explicit errors first
                    has_error = any(s.error for s in trace.steps)
                    blame_res = blame_trace(trace)

                    if has_error:
                        status = "error"
                    elif blame_res.blame_score > 0.35 and not blame_res.is_fallback_latency:
                        status = "blame"
                    else:
                        status = "healthy"

                    if blame_res.primary_step:
                        blame_info = {
                            "primary_step_name": blame_res.primary_step.name,
                            "primary_step_index": blame_res.primary_step.index,
                            "primary_step_type": blame_res.primary_step.step_type,
                            "blame_score": blame_res.blame_score,
                            "confidence": blame_res.confidence,
                            "explanation": blame_res.explanation,
                            "is_fallback_latency": blame_res.is_fallback_latency,
                        }
                except Exception:
                    pass

                enriched_runs.append({
                    "run_id": r["run_id"],
                    "pipeline_name": r["pipeline_name"] or "unnamed",
                    "start_ts": r["start_ts"],
                    "end_ts": r["end_ts"],
                    "step_count": r["step_count"],
                    "duration_ms": r["duration_ms"],
                    "status": status,
                    "blame": blame_info,
                })

            self._send_json({"runs": enriched_runs, "count": len(enriched_runs)})
        except Exception as e:
            self._send_json({"error": "list_runs_error", "message": str(e)}, status=500)

    def _api_get_run_detail(self, run_id: str) -> None:
        """Return full trace telemetry and computed blame analysis."""
        try:
            trace = self.store.load_trace(run_id)
        except ValueError:
            self._send_json({"error": "not_found", "message": f"Run not found: {run_id}"}, status=404)
            return
        except Exception as e:
            self._send_json({"error": "load_error", "message": str(e)}, status=500)
            return

        blame_res = blame_trace(trace)

        # Convert trace & steps to JSON-friendly dict
        steps_data: list[dict[str, Any]] = []
        for s in trace.steps:
            steps_data.append({
                "step_id": s.step_id,
                "name": s.name,
                "step_type": s.step_type,
                "index": s.index,
                "input": s.input,
                "output": s.output,
                "start_ts": s.start_ts,
                "end_ts": s.end_ts,
                "latency_ms": s.latency_ms,
                "token_count": s.token_count,
                "cost_usd": s.cost_usd,
                "metadata": s.metadata or {},
                "score": s.score,
                "error": s.error,
                "parent_step_id": getattr(s, "parent_step_id", None),
            })

        blame_data = {
            "primary_step_id": blame_res.primary_step.step_id if blame_res.primary_step else None,
            "primary_step_index": blame_res.primary_step.index if blame_res.primary_step else None,
            "primary_step_name": blame_res.primary_step.name if blame_res.primary_step else None,
            "blame_score": blame_res.blame_score,
            "confidence": blame_res.confidence,
            "explanation": blame_res.explanation,
            "is_fallback_latency": blame_res.is_fallback_latency,
            "co_blamed_step_ids": [s.step_id for s in blame_res.co_blamed],
        }

        total_ms = (trace.end_ts - trace.start_ts) * 1000 if trace.end_ts and trace.start_ts else 0.0
        total_tokens = sum(s.token_count or 0 for s in trace.steps)
        total_cost = sum(s.cost_usd or 0.0 for s in trace.steps)
        has_error = any(s.error for s in trace.steps)

        status = "healthy"
        if has_error:
            status = "error"
        elif blame_res.blame_score > 0.35 and not blame_res.is_fallback_latency:
            status = "blame"

        self._send_json({
            "run_id": trace.run_id,
            "pipeline_name": trace.pipeline_name or "unnamed",
            "start_ts": trace.start_ts,
            "end_ts": trace.end_ts,
            "duration_ms": total_ms,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "final_output": trace.final_output,
            "metadata": trace.metadata or {},
            "status": status,
            "steps": steps_data,
            "blame": blame_data,
        })

    def _api_get_blame(self, run_id: str) -> None:
        """Return dedicated blame attribution result for a run."""
        try:
            trace = self.store.load_trace(run_id)
        except ValueError:
            self._send_json({"error": "not_found", "message": f"Run not found: {run_id}"}, status=404)
            return

        blame_res = blame_trace(trace)
        self._send_json({
            "run_id": run_id,
            "pipeline_name": trace.pipeline_name,
            "primary_step": {
                "step_id": blame_res.primary_step.step_id,
                "name": blame_res.primary_step.name,
                "step_type": blame_res.primary_step.step_type,
                "index": blame_res.primary_step.index,
                "score": blame_res.primary_step.score,
                "latency_ms": blame_res.primary_step.latency_ms,
                "error": blame_res.primary_step.error,
            } if blame_res.primary_step else None,
            "blame_score": blame_res.blame_score,
            "confidence": blame_res.confidence,
            "explanation": blame_res.explanation,
            "is_fallback_latency": blame_res.is_fallback_latency,
            "co_blamed": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "step_type": s.step_type,
                    "index": s.index,
                    "score": s.score,
                }
                for s in blame_res.co_blamed
            ],
        })

    def _api_get_diff(self, query: dict[str, list[str]]) -> None:
        """Return step-by-step diff comparison between run A and run B."""
        run_a = query.get("a", [None])[0]
        run_b = query.get("b", [None])[0]

        if not run_a or not run_b:
            self._send_json({"error": "missing_params", "message": "Parameters 'a' and 'b' are required."}, status=400)
            return

        try:
            trace_a = self.store.load_trace(run_a)
        except ValueError:
            self._send_json({"error": "not_found", "message": f"Run A '{run_a}' not found."}, status=404)
            return

        try:
            trace_b = self.store.load_trace(run_b)
        except ValueError:
            self._send_json({"error": "not_found", "message": f"Run B '{run_b}' not found."}, status=404)
            return

        diff_res = diff_traces(trace_a, trace_b)

        def _step_dict(s: Step) -> dict[str, Any]:
            return {
                "step_id": s.step_id,
                "name": s.name,
                "step_type": s.step_type,
                "index": s.index,
                "score": s.score,
                "latency_ms": s.latency_ms,
                "token_count": s.token_count,
                "error": s.error,
            }

        self._send_json({
            "run_a": diff_res.run_a,
            "run_b": diff_res.run_b,
            "pipeline_name": diff_res.pipeline_name,
            "verdict": diff_res.verdict,
            "primary_diverged_step": diff_res.primary_diverged_step,
            "explanation": diff_res.explanation,
            "regressed_steps": [
                {"step_a": _step_dict(sa), "step_b": _step_dict(sb), "delta": delta}
                for sa, sb, delta in diff_res.regressed_steps
            ],
            "improved_steps": [
                {"step_a": _step_dict(sa), "step_b": _step_dict(sb), "delta": delta}
                for sa, sb, delta in diff_res.improved_steps
            ],
            "stable_steps": [
                {"step_a": _step_dict(sa), "step_b": _step_dict(sb), "delta": delta}
                for sa, sb, delta in diff_res.stable_steps
            ],
            "added_steps": [_step_dict(s) for s in diff_res.added_steps],
            "removed_steps": [_step_dict(s) for s in diff_res.removed_steps],
            "trace_a_summary": {
                "run_id": trace_a.run_id,
                "start_ts": trace_a.start_ts,
                "step_count": len(trace_a.steps),
                "duration_ms": (trace_a.end_ts - trace_a.start_ts) * 1000 if trace_a.end_ts and trace_a.start_ts else None,
            },
            "trace_b_summary": {
                "run_id": trace_b.run_id,
                "start_ts": trace_b.start_ts,
                "step_count": len(trace_b.steps),
                "duration_ms": (trace_b.end_ts - trace_b.start_ts) * 1000 if trace_b.end_ts and trace_b.start_ts else None,
            },
        })


def make_handler_class(store: Store) -> type[DashboardHandler]:
    """Factory creating handler with injected Store instance."""
    class CustomDashboardHandler(DashboardHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, store=store, **kwargs)
    return CustomDashboardHandler


def find_free_port(host: str, start_port: int = 7788, max_attempts: int = 20) -> int:
    """Find an open network port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port


def create_server(
    host: str = "127.0.0.1",
    port: int = 7788,
    db_path: Optional[str] = None,
    store: Optional[Store] = None,
) -> tuple[ThreadingHTTPServer, int]:
    """Instantiate ThreadingHTTPServer configured with Store."""
    store_inst = store or Store(db_path=db_path)
    handler_class = make_handler_class(store_inst)

    if port == 0:
        server = ThreadingHTTPServer((host, 0), handler_class)
        actual_port = server.server_address[1]
        return server, actual_port

    bound_port = port
    try:
        server = ThreadingHTTPServer((host, port), handler_class)
    except OSError:
        bound_port = find_free_port(host, port)
        server = ThreadingHTTPServer((host, bound_port), handler_class)

    actual_port = server.server_address[1]
    return server, actual_port


def start_server(
    host: str = "127.0.0.1",
    port: int = 7788,
    open_browser: bool = True,
    db_path: Optional[str] = None,
    store: Optional[Store] = None,
    blocking: bool = True,
) -> tuple[ThreadingHTTPServer, int]:
    """
    Start the local Mission Control HTTP dashboard server.
    """
    server, actual_port = create_server(host=host, port=port, db_path=db_path, store=store)
    url = f"http://{host}:{actual_port}"

    print(f"\n========================================================")
    print(f"  TRACEBACK AI // MISSION CONTROL TELEMETRY DASHBOARD")
    print(f"========================================================")
    print(f"  • Local URL:     {url}")
    print(f"  • SQLite Store:  {getattr(server.RequestHandlerClass, 'store', Store(db_path=db_path)).db_path}")
    print(f"  • Status:        ACTIVE (Press Ctrl+C to stop)")
    print(f"========================================================\n")

    if open_browser:
        def _open() -> None:
            webbrowser.open(url)
        timer = threading.Timer(0.3, _open)
        timer.daemon = True
        timer.start()

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Traceback Mission Control server...")
            server.shutdown()
            server.server_close()

    return server, actual_port
