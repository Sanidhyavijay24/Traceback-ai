"""
Traceback AI - Google Gemini SDK Integration.

Provides TracedGemini wrapper and automatic tracing utilities for Gemini API calls.
"""

import functools
import os
import time
from typing import Any, Optional

from tracebackai.models import Step
from tracebackai.token_utils import count_tokens
from tracebackai.tracer import get_active_trace

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore
        _GENAI_AVAILABLE = True
    except ImportError:
        _GENAI_AVAILABLE = False


def _extract_gemini_content(response: Any) -> str:
    """Extract output text from Gemini response object."""
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            parts_text = [p.text for p in candidate.content.parts if hasattr(p, "text")]
            return "\n".join(parts_text)
    return str(response)


class TracedGemini:
    """Wrapper around Google Gemini client that automatically records trace steps."""

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "Google GenAI SDK is not installed. Install with: pip install google-genai"
            )
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        # Try google.genai (new SDK) first, fallback to google.generativeai
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key, **kwargs)
            self._sdk_type = "new"
        except Exception:
            import google.generativeai as legacy_genai
            if self.api_key:
                legacy_genai.configure(api_key=self.api_key)
            self._client = legacy_genai
            self._sdk_type = "legacy"

    def generate_content(
        self,
        model: str = "gemini-3.6-flash",
        contents: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate content while recording an LLM trace step."""
        active_trace = get_active_trace()
        if active_trace is None:
            if self._sdk_type == "new":
                return self._client.models.generate_content(model=model, contents=contents, **kwargs)
            else:
                m = self._client.GenerativeModel(model)
                return m.generate_content(contents, **kwargs)

        step = Step(
            run_id=active_trace.run_id,
            name=f"gemini_{model.replace('/', '_')}",
            step_type="llm",
            input=contents,
            start_ts=time.time(),
            metadata={"model": model, "provider": "gemini"},
        )

        try:
            if self._sdk_type == "new":
                response = self._client.models.generate_content(model=model, contents=contents, **kwargs)
            else:
                m = self._client.GenerativeModel(model)
                response = m.generate_content(contents, **kwargs)

            out_text = _extract_gemini_content(response)
            step.output = out_text

            # Extract token metrics if present
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                in_tok = getattr(response.usage_metadata, "prompt_token_count", None)
                out_tok = getattr(response.usage_metadata, "candidates_token_count", None)
                if in_tok is not None and out_tok is not None:
                    step.token_count = in_tok + out_tok
                    step.metadata["input_tokens"] = in_tok
                    step.metadata["output_tokens"] = out_tok

            return response
        except Exception:
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
