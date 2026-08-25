"""
Tests for Traceback AI data models.
"""

from tracebackai.models import Step, Trace


def test_step_defaults():
    """Verify default field values for Step dataclass."""
    step = Step(name="test_step")
    assert len(step.step_id) == 8
    assert step.name == "test_step"
    assert step.step_type == "generic"
    assert step.index == 0
    assert step.input is None
    assert step.output is None
    assert step.error is None
    assert step.score is None
    assert isinstance(step.metadata, dict)
    assert step.start_ts > 0


def test_trace_defaults():
    """Verify default field values for Trace dataclass."""
    trace = Trace(pipeline_name="test_pipeline")
    assert len(trace.run_id) == 12
    assert trace.pipeline_name == "test_pipeline"
    assert isinstance(trace.steps, list)
    assert len(trace.steps) == 0
    assert trace.start_ts > 0
    assert trace.end_ts is None
    assert trace.final_output is None
    assert isinstance(trace.metadata, dict)
