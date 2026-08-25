"""
Traceback AI - Scorers Registry and Discoveries.
"""

from typing import Optional

from tracebackai.scorers.base import BaseScorer
from tracebackai.scorers.llm import LLMScorer
from tracebackai.scorers.retrieval import RetrievalScorer, WEAK_RETRIEVAL_THRESHOLD
from tracebackai.scorers.tool import ToolScorer


class ScorerRegistry:
    """Registry mapping step types to BaseScorer instances."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._scorers: dict[str, BaseScorer] = {}
        if register_defaults:
            self.register(RetrievalScorer())
            self.register(LLMScorer())
            self.register(ToolScorer())

    def register(self, scorer: BaseScorer) -> None:
        """Register a custom or standard scorer for its step_type."""
        self._scorers[scorer.step_type] = scorer

    def get(self, step_type: str) -> Optional[BaseScorer]:
        """Retrieve the scorer instance registered for a step_type."""
        return self._scorers.get(step_type)


__all__ = [
    "BaseScorer",
    "RetrievalScorer",
    "LLMScorer",
    "ToolScorer",
    "ScorerRegistry",
    "WEAK_RETRIEVAL_THRESHOLD",
]
