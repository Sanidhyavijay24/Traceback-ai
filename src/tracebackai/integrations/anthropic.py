"""
Traceback AI - Anthropic SDK Integration.

Provides TracedAnthropic wrapper and monkeypatching utility for tracing Anthropic API calls.
"""

import functools
import time
from typing import Any, Optional

from tracebackai.models import Step
from tracebackai.token_utils import count_tokens
from tracebackai.tracer import get_active_trace

try:
    import anthropic
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    Anthropic = object  # type: ignore


def _extract_anthropic_content(response: Any) -> str:
    """Extract text content from Anthropic Message response."""
    if hasattr(response, "content") and isinstance(response.content, list):
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_parts)
    return str(response)


def _wrap_messages_create(original_create: Any) -> Any:
    """Wrap anthropic.messages.create to record execution as an LLM step."""
    @functools.wraps(original_create)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        active_trace = get_active_trace()
        if active_trace is None:
            return original_create(*args, **kwargs)

        model = kwargs.get("model", "claude")
        messages = kwargs.get("messages", [])
        prompt_str = str(messages)

        step = Step(
            run_id=active_trace.run_id,
            name=f"anthropic_{model}",
            step_type="llm",
            input=messages if len(messages) > 1 else (messages[0] if messages else None),
            start_ts=time.time(),
            metadata={"model": model, "provider": "anthropic"},
        )

        try:
            response = original_create(*args, **kwargs)
            out_text = _extract_anthropic_content(response)
            step.output = out_text

            if hasattr(response, "usage") and response.usage:
                in_tok = getattr(response.usage, "input_tokens", None)
                out_tok = getattr(response.usage, "output_tokens", None)
                if in_tok is not None and out_tok is not None:
                    step.token_count = in_tok + out_tok
                    step.metadata["input_tokens"] = in_tok
                    step.metadata["output_tokens"] = out_tok
            if hasattr(response, "stop_reason"):
                step.metadata["stop_reason"] = response.stop_reason

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


def patch_anthropic() -> None:
    """Monkeypatch the Anthropic SDK to automatically trace all messages.create calls."""
    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is not installed. Install with: pip install traceback-ai[anthropic]")

    import anthropic.resources.messages
    if not getattr(anthropic.resources.messages.Messages.create, "_is_traced", False):
        orig = anthropic.resources.messages.Messages.create
        wrapped = _wrap_messages_create(orig)
        wrapped._is_traced = True  # type: ignore
        anthropic.resources.messages.Messages.create = wrapped


class TracedAnthropic(Anthropic if _ANTHROPIC_AVAILABLE else object):  # type: ignore
    """Anthropic client that automatically records trace steps for all requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Install with: pip install traceback-ai[anthropic]")
        super().__init__(*args, **kwargs)
        self.messages.create = _wrap_messages_create(self.messages.create)
