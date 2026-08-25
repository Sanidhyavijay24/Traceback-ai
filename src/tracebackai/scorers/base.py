"""
Traceback AI - Base Scorer ABC and Registry.

Defines the abstract interface for step evaluation and the registry for dispatching.
"""

from abc import ABC, abstractmethod
from typing import Optional

from tracebackai.models import Step


class BaseScorer(ABC):
    """Abstract base class for all step-level scorers."""

    step_type: str = "generic"

    @abstractmethod
    def score(self, step: Step) -> float:
        """
        Compute health score for a step.

        Must return a float clamped in [0.0, 1.0]. Higher is healthier.
        """
        pass

    def can_score(self, step: Step) -> bool:
        """Check if this scorer is applicable to the given step."""
        return step.step_type == self.step_type and step.error is None
