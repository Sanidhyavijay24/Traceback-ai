"""
Traceback AI - Execution Tracer.

Provides the @trace decorator and TraceContext context manager for capturing
agent and pipeline execution spans into structured traces.
"""

import functools
import time
import traceback as _tb
import warnings
from typing import Any, Callable, Optional

from tracebackai.models import Step, Trace
from tracebackai.scoring import score_trace
from tracebackai.store import Store
from tracebackai.token_utils import count_tokens

_active_trace: Optional[Trace] = None


def _serialize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Format function call arguments into a structured input representation."""
    if not args and not kwargs:
        return None
    if len(args) == 1 and not kwargs:
        return args[0]
    result: dict[str, Any] = {}
    if args:
        result["args"] = list(args)
    if kwargs:
        result["kwargs"] = kwargs
    return result


def get_active_trace() -> Optional[Trace]:
    """Return the currently active execution trace if one exists."""
    return _active_trace


def trace(
    func: Optional[Callable[..., Any]] = None,
    *,
    step_type: str = "generic",
    name: Optional[str] = None,
    pipeline: bool = False,
) -> Any:
    """
    Decorator to instrument a function as a trace step or root pipeline.

    Usage:
        @trace
        def generic_step(x): ...

        @trace(step_type="retrieval")
        def retrieve_chunks(q): ...

        @trace(pipeline=True)
        def my_agent(q): ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _active_trace

            # A pipeline=True call while a trace is already active is a nested pipeline
            if pipeline and _active_trace is not None:
                warnings.warn(
                    f"Nested pipeline call detected ({name or fn.__name__!r} called "
                    f"while trace {_active_trace.run_id!r} is active). Flattening "
                    f"into the parent trace for v1.0 — see Known Hard Parts.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            is_root = _active_trace is None

            if is_root:
                _active_trace = Trace(pipeline_name=name or fn.__name__)

            step = Step(
                run_id=_active_trace.run_id,
                name=name or fn.__name__,
                step_type=step_type,
                input=_serialize_args(args, kwargs),
                start_ts=time.time(),
            )

            try:
                result = fn(*args, **kwargs)
                step.output = result
                return result
            except Exception:
                step.error = _tb.format_exc()
                raise
            finally:
                step.end_ts = time.time()
                step.latency_ms = (step.end_ts - step.start_ts) * 1000
                step.token_count = count_tokens(step.input, step.output)

                if _active_trace is not None:
                    step.index = len(_active_trace.steps)
                    _active_trace.steps.append(step)

                if is_root and _active_trace is not None:
                    _active_trace.end_ts = time.time()
                    _active_trace.final_output = step.output
                    score_trace(_active_trace)
                    Store().save_trace(_active_trace)
                    _active_trace = None

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class StepContext:
    """Context manager for a single step inside a TraceContext."""

    def __init__(
        self,
        trace_obj: Trace,
        name: str,
        step_type: str = "generic",
        input_data: Any = None,
    ) -> None:
        self.trace_obj = trace_obj
        self.step = Step(
            run_id=trace_obj.run_id,
            name=name,
            step_type=step_type,
            input=input_data,
            start_ts=time.time(),
        )

    def __enter__(self) -> "StepContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        if exc_val is not None:
            self.step.error = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
        self.step.end_ts = time.time()
        self.step.latency_ms = (self.step.end_ts - self.step.start_ts) * 1000
        self.step.token_count = count_tokens(self.step.input, self.step.output)
        self.step.index = len(self.trace_obj.steps)
        self.trace_obj.steps.append(self.step)
        return False

    def record(
        self,
        output: Any = None,
        input: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Explicitly record or update step attributes."""
        if output is not None:
            self.step.output = output
        if input is not None:
            self.step.input = input
        if metadata:
            self.step.metadata.update(metadata)
        if cost_usd is not None:
            self.step.cost_usd = cost_usd


class TraceContext:
    """Context manager for creating a Trace scope programmatically."""

    def __init__(
        self,
        pipeline_name: str = "pipeline",
        metadata: Optional[dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.metadata = metadata or {}
        self.db_path = db_path
        self.trace: Optional[Trace] = None
        self._is_root = False

    def __enter__(self) -> "TraceContext":
        global _active_trace
        self._is_root = _active_trace is None
        if self._is_root:
            self.trace = Trace(
                pipeline_name=self.pipeline_name,
                metadata=self.metadata,
                start_ts=time.time(),
            )
            _active_trace = self.trace
        else:
            self.trace = _active_trace
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        global _active_trace
        if self._is_root and self.trace is not None:
            self.trace.end_ts = time.time()
            if self.trace.steps and self.trace.final_output is None:
                self.trace.final_output = self.trace.steps[-1].output
            score_trace(self.trace)
            Store(db_path=self.db_path).save_trace(self.trace)
            _active_trace = None
        return False

    def step(
        self,
        name: str,
        step_type: str = "generic",
        input: Any = None,
    ) -> StepContext:
        """Create a child step context."""
        if self.trace is None:
            raise RuntimeError("TraceContext is not active")
        return StepContext(self.trace, name=name, step_type=step_type, input_data=input)
