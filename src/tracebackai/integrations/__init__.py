"""
Traceback AI - SDK Integrations.

Exposes optional wrappers and helpers for Gemini, Anthropic, OpenAI, and LangChain.
"""

__all__ = [
    "TracedGemini",
    "TracedAnthropic",
    "patch_anthropic",
    "TracedOpenAI",
    "patch_openai",
    "TracebackCallbackHandler",
]


def __getattr__(name: str):
    if name == "TracedGemini":
        from tracebackai.integrations.gemini import TracedGemini
        return TracedGemini
    if name in ("TracedAnthropic", "patch_anthropic"):
        from tracebackai.integrations.anthropic import TracedAnthropic, patch_anthropic
        return locals()[name]
    if name in ("TracedOpenAI", "patch_openai"):
        from tracebackai.integrations.openai import TracedOpenAI, patch_openai
        return locals()[name]
    if name == "TracebackCallbackHandler":
        from tracebackai.integrations.langchain import TracebackCallbackHandler
        return TracebackCallbackHandler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
