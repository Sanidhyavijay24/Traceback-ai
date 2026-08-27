"""
Traceback AI - Failure Attribution and Blame Analysis.

Identifies the single most likely failure-causative step in a trace run
and provides cross-run diff comparisons to pinpoint regressions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from tracebackai.models import Step, Trace
from tracebackai.scorers.retrieval import WEAK_RETRIEVAL_THRESHOLD, get_retrieval_threshold
from tracebackai.store import Store

TYPE_WEIGHTS: dict[str, float] = {
    "retrieval": 1.4,  # retrieval failures compound downstream
    "llm": 1.2,
    "tool": 1.1,
    "prompt": 0.9,
    "generic": 0.8,
}


@dataclass
class BlameResult:
    """Represents the outcome of a failure attribution analysis."""

    primary_step: Optional[Step]
    blame_score: float
    co_blamed: list[Step] = field(default_factory=list)
    explanation: str = ""
    confidence: str = "medium"  # "high" | "medium" | "low"
    is_fallback_latency: bool = False


@dataclass
class DiffResult:
    """Represents comparative step-by-step diff between two trace runs."""

    run_a: str
    run_b: str
    pipeline_name: str
    regressed_steps: list[tuple[Step, Step, float]] = field(default_factory=list)  # (step_a, step_b, delta)
    improved_steps: list[tuple[Step, Step, float]] = field(default_factory=list)
    stable_steps: list[tuple[Step, Step, float]] = field(default_factory=list)
    added_steps: list[Step] = field(default_factory=list)  # in B but not A
    removed_steps: list[Step] = field(default_factory=list)  # in A but not B
    verdict: str = "NEUTRAL"  # "REGRESSION" | "IMPROVEMENT" | "NEUTRAL"
    primary_diverged_step: Optional[str] = None
    explanation: str = ""


def _compute_recency_weight(index: int, total_steps: int) -> float:
    """Calculate recency multiplier giving higher weight to earlier pipeline steps."""
    if total_steps <= 1:
        return 1.0
    return 1.0 + (0.3 * (1.0 - (index / float(total_steps))))


def _generate_explanation(step: Step, score: Optional[float], is_latency_fallback: bool = False) -> str:
    """Generate human-readable explanation based on step metrics and sub-scores."""
    if is_latency_fallback:
        latency_str = f"{step.latency_ms:.0f}ms" if step.latency_ms is not None else "high latency"
        return (
            f"All steps in this run were unscored. Blame fell back to step '{step.name}' "
            f"as the slowest bottleneck ({latency_str})."
        )

    if step.error is not None:
        first_err_line = step.error.strip().splitlines()[-1].strip()
        return f"Step raised an unhandled exception: {first_err_line}."

    stype = step.step_type
    meta = step.metadata or {}
    score_val = score if score is not None else 0.0

    if stype == "retrieval":
        chunks_count = meta.get("retrieval_chunks_count", 0)
        if chunks_count == 0:
            return "Retrieval returned no document chunks. Downstream steps received empty context."
        threshold = get_retrieval_threshold(step)
        top_sim = meta.get("top_similarity")
        if score_val < threshold:
            if top_sim is not None:
                return (
                    f"Retrieval relevance score was {score_val:.2f} (below threshold {threshold:.2f}). "
                    f"Top chunk similarity was {top_sim:.2f}. Downstream steps received low-quality context."
                )
            return (
                f"Retrieved passages had low query similarity ({score_val:.2f} < {threshold:.2f}). "
                f"Downstream steps were starved of relevant context."
            )
        return f"Retrieval relevance was healthy (score: {score_val:.2f} >= threshold {threshold:.2f})."

    if stype == "llm":
        if meta.get("refusal_detected"):
            return "Model triggered a safety or policy refusal response instead of answering the query."
        resp_len = meta.get("response_length_tokens", 0)
        if resp_len < 20:
            return (
                f"Model response was unusually short ({resp_len} tokens). "
                f"May indicate truncation, unexpected stop token, or underspecified prompt."
            )
        return f"LLM response quality sub-scores degraded (overall health score: {score_val:.2f})."

    if stype == "tool":
        if meta.get("historical_error_rate", 0) > 0.3:
            hrate = meta.get("historical_error_rate", 0)
            return (
                f"Tool output health was low ({score_val:.2f}). "
                f"This tool has a high historical failure rate ({hrate:.0%})."
            )
        if step.output is None or (isinstance(step.output, (dict, list, str)) and len(step.output) == 0):
            return "Tool returned null or empty output payload."
        return f"Tool output did not match expected structure or health criteria (score: {score_val:.2f})."

    return f"Step '{step.name}' experienced health degradation with a score of {score_val:.2f}."


def blame_trace(trace: Trace) -> BlameResult:
    """
    Attribute root-cause failure in an execution trace to the most likely step.

    Excludes unscored steps (score is None) from candidate blame calculation.
    Falls back to blaming the slowest step only if all steps are unscored.
    """
    if not trace.steps:
        return BlameResult(
            primary_step=None,
            blame_score=0.0,
            explanation="Trace contains no execution steps.",
            confidence="high",
        )

    total_steps = len(trace.steps)

    # 1. First priority: any step that raised an explicit unhandled exception
    error_steps = [s for s in trace.steps if s.error is not None]
    if error_steps:
        primary = error_steps[0]
        co = error_steps[1:] if len(error_steps) > 1 else []
        return BlameResult(
            primary_step=primary,
            blame_score=1.0,
            co_blamed=co,
            explanation=_generate_explanation(primary, primary.score),
            confidence="high",
        )

    # 2. Candidate scored steps (strictly where score is not None)
    scored_steps = [s for s in trace.steps if s.score is not None]

    if not scored_steps:
        # Fallback: Blame slowest step
        slowest_step = max(trace.steps, key=lambda s: s.latency_ms or 0.0)
        return BlameResult(
            primary_step=slowest_step,
            blame_score=0.5,
            co_blamed=[],
            explanation=_generate_explanation(slowest_step, None, is_latency_fallback=True),
            confidence="low",
            is_fallback_latency=True,
        )

    # 3. Calculate blame score for each scored candidate
    candidates: list[tuple[Step, float]] = []
    for step in scored_steps:
        step_score = max(0.0, min(1.0, step.score if step.score is not None else 1.0))
        t_weight = TYPE_WEIGHTS.get(step.step_type, 1.0)
        r_weight = _compute_recency_weight(step.index, total_steps)
        b_score = (1.0 - step_score) * t_weight * r_weight
        candidates.append((step, b_score))

    # Sort descending by blame score
    candidates.sort(key=lambda item: item[1], reverse=True)

    primary_step, top_blame_score = candidates[0]

    # Co-blame and confidence calculation
    co_blamed: list[Step] = []
    confidence = "high"

    if len(candidates) > 1:
        second_step, second_score = candidates[1]
        gap = top_blame_score - second_score
        if gap < 0.05:
            co_blamed.append(second_step)
            confidence = "low"
        elif gap <= 0.20:
            confidence = "medium"
        else:
            confidence = "high"
    else:
        confidence = "high"

    return BlameResult(
        primary_step=primary_step,
        blame_score=round(top_blame_score, 4),
        co_blamed=co_blamed,
        explanation=_generate_explanation(primary_step, primary_step.score),
        confidence=confidence,
    )


def blame_run(run_id: str, store: Optional[Store] = None) -> BlameResult:
    """Load trace from store and run blame attribution analysis."""
    store_inst = store or Store()
    trace = store_inst.load_trace(run_id)
    return blame_trace(trace)


def diff_traces(trace_a: Trace, trace_b: Trace) -> DiffResult:
    """
    Compare two execution traces step-by-step, aligned by step name.

    Identifies regressed, improved, stable, added, and removed steps.
    """
    steps_a_by_name = {s.name: s for s in trace_a.steps}
    steps_b_by_name = {s.name: s for s in trace_b.steps}

    regressed: list[tuple[Step, Step, float]] = []
    improved: list[tuple[Step, Step, float]] = []
    stable: list[tuple[Step, Step, float]] = []

    # Matched steps
    for name, step_a in steps_a_by_name.items():
        if name in steps_b_by_name:
            step_b = steps_b_by_name[name]
            score_a = step_a.score if step_a.score is not None else 1.0
            score_b = step_b.score if step_b.score is not None else 1.0
            delta = round(score_b - score_a, 4)

            if delta < -0.05:
                regressed.append((step_a, step_b, delta))
            elif delta > 0.05:
                improved.append((step_a, step_b, delta))
            else:
                stable.append((step_a, step_b, delta))

    # Added / Removed
    added = [s for name, s in steps_b_by_name.items() if name not in steps_a_by_name]
    removed = [s for name, s in steps_a_by_name.items() if name not in steps_b_by_name]

    # Sort regressions by most negative delta first
    regressed.sort(key=lambda x: x[2])
    # Sort improvements by most positive delta first
    improved.sort(key=lambda x: x[2], reverse=True)

    # Determine verdict and primary diverged step
    if regressed:
        worst_step_a, worst_step_b, worst_delta = regressed[0]
        verdict = "REGRESSION"
        primary_diverged = worst_step_b.name
        explanation = (
            f"Significant quality regression in '{primary_diverged}' (delta: {worst_delta:+.2f}). "
            f"{_generate_explanation(worst_step_b, worst_step_b.score)}"
        )
    elif improved:
        best_step_a, best_step_b, best_delta = improved[0]
        verdict = "IMPROVEMENT"
        primary_diverged = best_step_b.name
        explanation = f"Quality improvement observed in '{primary_diverged}' (delta: {best_delta:+.2f})."
    else:
        verdict = "NEUTRAL"
        primary_diverged = None
        explanation = "Both runs exhibited comparable step health scores."

    return DiffResult(
        run_a=trace_a.run_id,
        run_b=trace_b.run_id,
        pipeline_name=trace_b.pipeline_name or trace_a.pipeline_name,
        regressed_steps=regressed,
        improved_steps=improved,
        stable_steps=stable,
        added_steps=added,
        removed_steps=removed,
        verdict=verdict,
        primary_diverged_step=primary_diverged,
        explanation=explanation,
    )


def diff_runs(run_id_a: str, run_id_b: str, store: Optional[Store] = None) -> DiffResult:
    """Load two traces from store and compute cross-run diff."""
    store_inst = store or Store()
    trace_a = store_inst.load_trace(run_id_a)
    trace_b = store_inst.load_trace(run_id_b)
    return diff_traces(trace_a, trace_b)
