"""
Traceback AI - LLM Agent Execution Tracer with Failure Attribution.
"""

from tracebackai.blame import (
    BlameResult,
    DiffResult,
    blame_run,
    blame_trace,
    diff_runs,
    diff_traces,
)
from tracebackai.models import Step, Trace
from tracebackai.scorers import (
    BaseScorer,
    LLMScorer,
    RetrievalScorer,
    ScorerRegistry,
    ToolScorer,
    WEAK_RETRIEVAL_THRESHOLD,
)
from tracebackai.scoring import score_trace
from tracebackai.store import Store
from tracebackai.token_utils import count_tokens
from tracebackai.tracer import TraceContext, get_active_trace, trace

__version__ = "0.1.0"
__all__ = [
    "Step",
    "Trace",
    "Store",
    "trace",
    "TraceContext",
    "get_active_trace",
    "count_tokens",
    "score_trace",
    "BaseScorer",
    "RetrievalScorer",
    "LLMScorer",
    "ToolScorer",
    "ScorerRegistry",
    "WEAK_RETRIEVAL_THRESHOLD",
    "blame_trace",
    "blame_run",
    "diff_traces",
    "diff_runs",
    "BlameResult",
    "DiffResult",
]
