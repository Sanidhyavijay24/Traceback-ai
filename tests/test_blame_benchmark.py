from pathlib import Path
import sys
import pytest

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.blame_accuracy import HEALTHY_BLAME_THRESHOLD, run_benchmark


def test_blame_attribution_accuracy_benchmark():
    """
    Run the end-to-end blame accuracy benchmark and assert baseline quality standards:
    - Top-1 attribution accuracy across failure scenarios must be >= 80%.
    - False positive rate on healthy traces must be <= 1 out of 3.
    """
    summary = run_benchmark()

    # Top-1 accuracy assertion
    top1_acc = summary["top1_accuracy"]
    evaluated = summary["failure_scenarios_evaluated"]
    passed = summary["failure_scenarios_passed"]

    assert evaluated >= 14, f"Expected at least 14 evaluated failure scenarios, got {evaluated}"
    assert top1_acc >= 0.80, (
        f"Blame attribution accuracy regressed below 80%: {top1_acc * 100:.1f}% "
        f"({passed}/{evaluated} passed)"
    )

    # False positive rate assertion on healthy traces
    healthy_fps = summary["healthy_false_positives"]
    healthy_total = summary["healthy_scenarios_evaluated"]

    assert healthy_total == 3, f"Expected 3 healthy scenarios, got {healthy_total}"
    assert healthy_fps <= 1, (
        f"Too many false positives on healthy traces: {healthy_fps}/{healthy_total} "
        f"exceeded threshold {HEALTHY_BLAME_THRESHOLD}"
    )
