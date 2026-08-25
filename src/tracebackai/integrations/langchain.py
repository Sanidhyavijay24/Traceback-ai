"""
Traceback AI - LangChain Integration.

Provides TracebackCallbackHandler for capturing LangChain execution events.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence
from uuid import UUID

from tracebackai.models import Step
from tracebackai.token_utils import count_tokens
from tracebackai.tracer import get_active_trace

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore


class TracebackCallbackHandler(BaseCallbackHandler if _LANGCHAIN_AVAILABLE else object):  # type: ignore
    """LangChain callback handler recording pipeline executions into active trace."""

    def __init__(self) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("langchain package is not installed. Install with: pip install traceback-ai[langchain]")
        super().__init__()
        self._active_steps: dict[str, Step] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        active_trace = get_active_trace()
        if active_trace is None:
            return

        name = (serialized or {}).get("name") or (metadata or {}).get("ls_model_name") or "langchain_llm"
        step = Step(
            run_id=active_trace.run_id,
            name=name,
            step_type="llm",
            input=prompts[0] if len(prompts) == 1 else prompts,
            start_ts=time.time(),
            metadata=metadata or {},
        )
        self._active_steps[str(run_id)] = step

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        active_trace = get_active_trace()
        if step is None or active_trace is None:
            return

        output_text = ""
        if hasattr(response, "generations") and response.generations:
            gen_list = response.generations[0]
            if gen_list and hasattr(gen_list[0], "text"):
                output_text = gen_list[0].text

        step.output = output_text
        step.end_ts = time.time()
        step.latency_ms = (step.end_ts - step.start_ts) * 1000
        step.token_count = count_tokens(step.input, step.output)
        step.index = len(active_trace.steps)
        active_trace.steps.append(step)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        active_trace = get_active_trace()
        if active_trace is None:
            return

        name = (serialized or {}).get("name") or "langchain_retriever"
        step = Step(
            run_id=active_trace.run_id,
            name=name,
            step_type="retrieval",
            input=query,
            start_ts=time.time(),
            metadata=metadata or {},
        )
        self._active_steps[str(run_id)] = step

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        active_trace = get_active_trace()
        if step is None or active_trace is None:
            return

        chunks: list[str] = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                chunks.append(doc.page_content)
            else:
                chunks.append(str(doc))

        step.output = chunks
        step.end_ts = time.time()
        step.latency_ms = (step.end_ts - step.start_ts) * 1000
        step.token_count = count_tokens(step.input, step.output)
        step.index = len(active_trace.steps)
        active_trace.steps.append(step)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        active_trace = get_active_trace()
        if active_trace is None:
            return

        name = (serialized or {}).get("name") or "langchain_tool"
        step = Step(
            run_id=active_trace.run_id,
            name=name,
            step_type="tool",
            input=inputs if inputs is not None else input_str,
            start_ts=time.time(),
            metadata=metadata or {},
        )
        self._active_steps[str(run_id)] = step

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        step = self._active_steps.pop(str(run_id), None)
        active_trace = get_active_trace()
        if step is None or active_trace is None:
            return

        step.output = output
        step.end_ts = time.time()
        step.latency_ms = (step.end_ts - step.start_ts) * 1000
        step.token_count = count_tokens(step.input, step.output)
        step.index = len(active_trace.steps)
        active_trace.steps.append(step)
