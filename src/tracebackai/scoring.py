"""
Traceback AI - Trace Scoring.

Applies registered step scorers to an execution trace in-place before persistence.
"""

from typing import Optional

from tracebackai.models import Trace
from tracebackai.scorers import ScorerRegistry


def score_trace(trace: Trace, registry: Optional[ScorerRegistry] = None) -> None:
    """
    Mutate trace.steps in place, computing and assigning step.score where applicable.

    Pure storage layers do not depend on scorers; scoring is triggered explicitly
    prior to persistence.
    """
    if registry is None:
        registry = ScorerRegistry()

    for step in trace.steps:
        scorer = registry.get(step.step_type)
        if scorer is not None and scorer.can_score(step):
            try:
                step.score = scorer.score(step)
            except Exception:
                # If scoring raises unexpectedly, preserve step without failing trace
                pass
