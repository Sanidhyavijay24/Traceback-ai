"""
Traceback AI - OpenAI SDK Integration.

Provides TracedOpenAI wrapper and monkeypatching utility for tracing OpenAI chat completions.
"""

import functools
import time
from typing import Any, Optional

from tracebackai.models import Step
from tracebackai.token_utils import count_tokens
from tracebackai.tracer import get_active_trace

try:
    import openai
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = object  # type: ignore


def _extract_openai_content(response: Any) -> str:
    """Extract output text from OpenAI chat completion response."""
    if hasattr(response, "choices") and response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
            return first_choice.message.content or ""
    return str(response)


def _wrap_completions_create(original_create: Any) -> Any:
    """Wrap openai.chat.completions.create to record execution as an LLM step."""
    @functools.wraps(original_create)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        active_trace = get_active_trace()
        if active_trace is None:
            return original_create(*args, **kwargs)

        model = kwargs.get("model", "gpt")
        messages = kwargs.get("messages", [])

        step = Step(
            run_id=active_trace.run_id,
            name=f"openai_{model}",
            step_type="llm",
            input=messages if len(messages) > 1 else (messages[0] if messages else None),
            start_ts=time.time(),
            metadata={"model": model, "provider": "openai"},
        )

        try:
            response = original_create(*args, **kwargs)
            out_text = _extract_openai_content(response)
            step.output = out_text

            if hasattr(response, "usage") and response.usage:
                in_tok = getattr(response.usage, "prompt_tokens", None)
                out_tok = getattr(response.usage, "completion_tokens", None)
                if in_tok is not None and out_tok is not None:
                    step.token_count = in_tok + out_tok
                    step.metadata["input_tokens"] = in_tok
                    step.metadata["output_tokens"] = out_tok
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    step.metadata["finish_reason"] = first_choice.finish_reason

            return response
        except Exception as e:
            import traceback as _tb
            step.error = _tb.format_exc()
            raise
        finally:
            step.end_ts = time.time()
            step.latency_ms = (step.end_ts - step.start_ts) * 1000
            if step.token_count is None:
                step.token_count = count_tokens(step.input, step.output)
            step.index = len(active_trace.steps)
            active_trace.steps.append(step)

    return wrapper


def patch_openai() -> None:
    """Monkeypatch the OpenAI SDK to automatically trace all chat.completions.create calls."""
    if not _OPENAI_AVAILABLE:
        raise ImportError("openai package is not installed. Install with: pip install traceback-ai[openai]")

    import openai.resources.chat.completions
    if not getattr(openai.resources.chat.completions.Completions.create, "_is_traced", False):
        orig = openai.resources.chat.completions.Completions.create
        wrapped = _wrap_completions_create(orig)
        wrapped._is_traced = True  # type: ignore
        openai.resources.chat.completions.Completions.create = wrapped


class TracedOpenAI(OpenAI if _OPENAI_AVAILABLE else object):  # type: ignore
    """OpenAI client that automatically records trace steps for all requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Install with: pip install traceback-ai[openai]")
        super().__init__(*args, **kwargs)
        self.chat.completions.create = _wrap_completions_create(self.chat.completions.create)
