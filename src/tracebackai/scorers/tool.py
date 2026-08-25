"""
Traceback AI - Tool Call Step Scorer.

Evaluates tool call outputs against expected types, emptiness, and historical error rates.
"""

from typing import Any, Optional

from tracebackai.models import Step
from tracebackai.scorers.base import BaseScorer
from tracebackai.store import Store


def _is_empty_output(val: Any) -> bool:
    """Check if output payload is None or empty collection/string."""
    if val is None:
        return True
    if isinstance(val, (str, list, tuple, dict, set)) and len(val) == 0:
        return True
    return False


def _check_expected_type(val: Any, expected: Any) -> bool:
    """Check if value matches the expected type description."""
    if expected is None:
        return True
    if isinstance(expected, type):
        return isinstance(val, expected)
    if isinstance(expected, str):
        type_str = expected.lower()
        if type_str in ("dict", "dictionary", "json", "object"):
            return isinstance(val, dict)
        if type_str in ("list", "array", "sequence"):
            return isinstance(val, (list, tuple))
        if type_str in ("str", "string", "text"):
            return isinstance(val, str)
        if type_str in ("int", "integer", "number", "float"):
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        if type_str in ("bool", "boolean"):
            return isinstance(val, bool)
    return True


class ToolScorer(BaseScorer):
    """Scorer for tool invocations checking output validity and reliability history."""

    step_type: str = "tool"

    def __init__(self, store: Optional[Store] = None) -> None:
        self.store = store

    def score(self, step: Step) -> float:
        """Compute tool health score."""
        # 1. Exception check
        if step.error is not None:
            return 0.0

        # 2. Empty output check
        if _is_empty_output(step.output):
            base_score = 0.2
        else:
            # 3. Expected type check
            expected_type = step.metadata.get("expected_type")
            if expected_type is not None:
                if _check_expected_type(step.output, expected_type):
                    base_score = 1.0
                else:
                    base_score = 0.5
            else:
                base_score = 1.0

        # 4. Historical error rate across recent runs
        try:
            store_instance = self.store or Store()
            error_rate = store_instance.get_tool_error_rate(step.name, limit=10)
        except Exception:
            error_rate = 0.0

        step.metadata["historical_error_rate"] = round(error_rate, 4)
        penalized_score = base_score * (1.0 - error_rate)
        final_score = max(0.0, min(1.0, penalized_score))
        return round(final_score, 4)
