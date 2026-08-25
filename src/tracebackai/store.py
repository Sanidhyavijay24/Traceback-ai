"""
Traceback AI - SQLite Persistence Store.

Provides robust storage and retrieval of execution traces with safe serialization.
"""

from datetime import date, datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Optional

from tracebackai.models import Step, Trace

MAX_STRING_LENGTH = 10_000
TRUNCATION_MARKER = "... [truncated]"


def sanitize_data(obj: Any) -> Any:
    """Recursively convert and sanitize complex objects for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, str):
        if len(obj) > MAX_STRING_LENGTH:
            return obj[:MAX_STRING_LENGTH] + TRUNCATION_MARKER
        return obj
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return sanitize_data(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): sanitize_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_data(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: sanitize_data(v) for k, v in vars(obj).items() if not k.startswith("_")}
    text = str(obj)
    if len(text) > MAX_STRING_LENGTH:
        return text[:MAX_STRING_LENGTH] + TRUNCATION_MARKER
    return text


def safe_json_dumps(data: Any) -> str:
    """Serialize any data structure to a JSON string with truncation safety."""
    if data is None:
        return ""
    try:
        sanitized = sanitize_data(data)
        return json.dumps(sanitized, ensure_ascii=False)
    except Exception:
        text = str(data)
        if len(text) > MAX_STRING_LENGTH:
            text = text[:MAX_STRING_LENGTH] + TRUNCATION_MARKER
        return json.dumps(text)


def safe_json_loads(text: Optional[str]) -> Any:
    """Deserialize a JSON string back into Python types."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


class Store:
    """SQLite trace storage manager."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize SQLite database at given path or TRACEBACK_DB_PATH / default path."""
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            env_path = os.environ.get("TRACEBACK_DB_PATH")
            if env_path:
                self.db_path = Path(env_path)
            else:
                self.db_path = Path.home() / ".traceback" / "traces.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema tables."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_name TEXT,
                    start_ts REAL,
                    end_ts REAL,
                    final_output TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                    name TEXT,
                    step_type TEXT,
                    step_index INTEGER,
                    input TEXT,
                    output TEXT,
                    start_ts REAL,
                    end_ts REAL,
                    latency_ms REAL,
                    token_count INTEGER,
                    cost_usd REAL,
                    metadata TEXT,
                    score REAL,
                    error TEXT
                );
                """
            )
            conn.commit()

    def save_trace(self, trace: Trace) -> None:
        """Persist a complete trace and its steps into the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, pipeline_name, start_ts, end_ts, final_output, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.run_id,
                    trace.pipeline_name,
                    trace.start_ts,
                    trace.end_ts,
                    safe_json_dumps(trace.final_output),
                    safe_json_dumps(trace.metadata),
                ),
            )

            # Delete prior steps if updating an existing run
            conn.execute("DELETE FROM steps WHERE run_id = ?", (trace.run_id,))

            for idx, step in enumerate(trace.steps):
                conn.execute(
                    """
                    INSERT INTO steps (
                        step_id, run_id, name, step_type, step_index,
                        input, output, start_ts, end_ts, latency_ms,
                        token_count, cost_usd, metadata, score, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        step.step_id,
                        trace.run_id,
                        step.name,
                        step.step_type,
                        step.index if step.index is not None else idx,
                        safe_json_dumps(step.input),
                        safe_json_dumps(step.output),
                        step.start_ts,
                        step.end_ts,
                        step.latency_ms,
                        step.token_count,
                        step.cost_usd,
                        safe_json_dumps(step.metadata),
                        step.score,
                        step.error,
                    ),
                )
            conn.commit()

    def load_trace(self, run_id: str) -> Trace:
        """Load a trace and its steps by run_id."""
        with self._get_connection() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not run_row:
                raise ValueError(f"Trace run not found: {run_id}")

            step_rows = conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index ASC, start_ts ASC",
                (run_id,),
            ).fetchall()

            steps = [
                Step(
                    step_id=row["step_id"],
                    run_id=row["run_id"],
                    name=row["name"] or "",
                    step_type=row["step_type"] or "generic",
                    index=row["step_index"],
                    input=safe_json_loads(row["input"]),
                    output=safe_json_loads(row["output"]),
                    start_ts=row["start_ts"],
                    end_ts=row["end_ts"],
                    latency_ms=row["latency_ms"],
                    token_count=row["token_count"],
                    cost_usd=row["cost_usd"],
                    metadata=safe_json_loads(row["metadata"]) or {},
                    score=row["score"],
                    error=row["error"],
                )
                for row in step_rows
            ]

            return Trace(
                run_id=run_row["run_id"],
                pipeline_name=run_row["pipeline_name"] or "",
                steps=steps,
                start_ts=run_row["start_ts"],
                end_ts=run_row["end_ts"],
                final_output=safe_json_loads(run_row["final_output"]),
                metadata=safe_json_loads(run_row["metadata"]) or {},
            )

    def list_runs(
        self, pipeline_name: Optional[str] = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List summary info for recently recorded runs."""
        with self._get_connection() as conn:
            if pipeline_name:
                rows = conn.execute(
                    """
                    SELECT r.run_id, r.pipeline_name, r.start_ts, r.end_ts,
                           COUNT(s.step_id) AS step_count
                    FROM runs r
                    LEFT JOIN steps s ON r.run_id = s.run_id
                    WHERE r.pipeline_name = ?
                    GROUP BY r.run_id
                    ORDER BY r.start_ts DESC
                    LIMIT ?
                    """,
                    (pipeline_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT r.run_id, r.pipeline_name, r.start_ts, r.end_ts,
                           COUNT(s.step_id) AS step_count
                    FROM runs r
                    LEFT JOIN steps s ON r.run_id = s.run_id
                    GROUP BY r.run_id
                    ORDER BY r.start_ts DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [
                {
                    "run_id": row["run_id"],
                    "pipeline_name": row["pipeline_name"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"],
                    "step_count": row["step_count"],
                    "duration_ms": (
                        (row["end_ts"] - row["start_ts"]) * 1000
                        if row["end_ts"] and row["start_ts"]
                        else None
                    ),
                }
                for row in rows
            ]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its corresponding steps from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()

    def get_tool_error_rate(self, tool_name: str, limit: int = 10) -> float:
        """Compute the error rate of a named tool across recent executions."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT error FROM steps
                WHERE name = ? AND step_type = 'tool'
                ORDER BY start_ts DESC
                LIMIT ?
                """,
                (tool_name, limit),
            ).fetchall()

            if not rows:
                return 0.0

            error_count = sum(1 for r in rows if r["error"] is not None and str(r["error"]).strip() != "")
            return error_count / len(rows)

