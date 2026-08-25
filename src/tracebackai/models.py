"""
Traceback AI - Data Models.

Defines Step and Trace dataclasses used to represent pipeline executions.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Optional
import uuid


@dataclass
class Step:
    """Represents a single executed step within an agent/LLM trace."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: str = ""
    name: str = ""
    step_type: str = "generic"  # "retrieval" | "llm" | "tool" | "prompt" | "generic"
    index: int = 0
    input: Any = None
    output: Any = None
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    latency_ms: Optional[float] = None
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # filled by scorer in Phase 2
    error: Optional[str] = None


@dataclass
class Trace:
    """Represents an entire execution trace containing ordered steps."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    pipeline_name: str = ""
    steps: list[Step] = field(default_factory=list)
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    final_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
