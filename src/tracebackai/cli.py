"""
Traceback AI - Command Line Interface.

Provides CLI commands for inspecting and debugging agent execution traces.
"""

from datetime import datetime
import json
from typing import Any, Optional
import click

from tracebackai.store import Store


def _format_ts(ts: Optional[float]) -> str:
    """Format Unix timestamp into human-readable date/time string."""
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(ms: Optional[float]) -> str:
    """Format duration in milliseconds to human-friendly string."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def _format_value(val: Any, max_len: int = 80) -> str:
    """Format input/output value for CLI display."""
    if val is None:
        return "None"
    if isinstance(val, (dict, list)):
        try:
            s = json.dumps(val, ensure_ascii=False)
        except Exception:
            s = str(val)
    else:
        s = str(val)
    # Collapse newlines for compact CLI view
    s = " ".join(s.split())
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


@click.group()
@click.version_option(version="0.1.0", prog_name="traceback")
def cli() -> None:
    """Traceback AI - Execution Tracer with Failure Attribution."""
    pass


@cli.command(name="list")
@click.option("--pipeline", "-p", default=None, help="Filter runs by pipeline name.")
@click.option("--limit", "-n", default=20, type=int, help="Maximum number of runs to show.")
def list_cmd(pipeline: Optional[str], limit: int) -> None:
    """List recent execution traces."""
    store = Store()
    runs = store.list_runs(pipeline_name=pipeline, limit=limit)

    if not runs:
        click.echo("No trace runs found.")
        return

    header = f"{'ID':<14} {'PIPELINE':<22} {'STEPS':<8} {'START':<20} {'DURATION':<10}"
    click.echo(header)
    click.echo("-" * len(header))

    for run in runs:
        run_id = run["run_id"]
        pname = run["pipeline_name"] or "unnamed"
        steps_cnt = str(run["step_count"])
        start_str = _format_ts(run["start_ts"])
        dur_str = _format_duration(run["duration_ms"])
        click.echo(f"{run_id:<14} {pname:<22} {steps_cnt:<8} {start_str:<20} {dur_str:<10}")


@cli.command(name="show")
@click.argument("run_id")
@click.option("--verbose", "-v", is_flag=True, help="Show full input and output payloads.")
def show_cmd(run_id: str, verbose: bool) -> None:
    """Show detailed breakdown of a specific execution trace."""
    store = Store()
    try:
        trace = store.load_trace(run_id)
    except ValueError:
        click.secho(f"Error: Run ID '{run_id}' not found.", fg="red", err=True)
        return

    start_str = _format_ts(trace.start_ts)
    total_ms = (trace.end_ts - trace.start_ts) * 1000 if trace.end_ts and trace.start_ts else 0.0

    click.echo(f"Run: {trace.run_id}  |  Pipeline: {trace.pipeline_name or 'unnamed'}  |  {start_str}")
    click.echo("-" * 70)

    total_cost = 0.0
    for idx, step in enumerate(trace.steps):
        step_num = step.index if step.index is not None else idx
        dur_str = f"{step.latency_ms:.0f}ms" if step.latency_ms is not None else "0ms"
        tokens_str = f"tokens={step.token_count}" if step.token_count is not None else ""
        cost_str = f"cost=${step.cost_usd:.4f}" if step.cost_usd is not None else ""
        if step.cost_usd:
            total_cost += step.cost_usd

        # Score formatting
        score_str = ""
        if step.score is not None:
            if step.score >= 0.75:
                score_str = click.style(f"score={step.score:.2f} [OK]", fg="green")
            elif step.score >= 0.50:
                if step.step_type == "retrieval" and step.score < 0.55:
                    score_str = click.style(f"score={step.score:.2f} [!] WEAK RETRIEVAL", fg="yellow")
                else:
                    score_str = click.style(f"score={step.score:.2f} [!]", fg="yellow")
            else:
                if step.step_type == "retrieval":
                    score_str = click.style(f"score={step.score:.2f} [FAIL] WEAK RETRIEVAL", fg="red")
                else:
                    score_str = click.style(f"score={step.score:.2f} [FAIL]", fg="red")

        extra_info = "  ".join(filter(None, [dur_str, tokens_str, cost_str, score_str]))
        click.echo(f"[{step_num}] {step.name:<18} {step.step_type:<12} {extra_info}")

        if step.error:
            last_err_line = step.error.strip().splitlines()[-1].strip()
            click.secho(f"    ERROR: {last_err_line}", fg="red")

        if step.input is not None:
            inp_val = _format_value(step.input, max_len=200 if verbose else 60)
            click.echo(f"    input:  {inp_val}")
        if step.output is not None:
            out_val = _format_value(step.output, max_len=200 if verbose else 60)
            click.echo(f"    output: {out_val}")

    click.echo("-" * 70)
    summary_parts = [f"Total: {total_ms:.0f}ms"]
    if total_cost > 0:
        summary_parts.append(f"Cost: ${total_cost:.4f}")
    if trace.final_output is not None:
        summary_parts.append(f"Final Output: {_format_value(trace.final_output, max_len=60)}")
    click.echo("  |  ".join(summary_parts))


@cli.command(name="blame")
@click.argument("run_id")
def blame_cmd(run_id: str) -> None:
    """Identify the single most likely failure-causative step in a run."""
    from tracebackai.blame import blame_run
    from tracebackai.scorers.retrieval import WEAK_RETRIEVAL_THRESHOLD

    store = Store()
    try:
        trace = store.load_trace(run_id)
    except ValueError:
        click.secho(f"Error: Run ID '{run_id}' not found.", fg="red", err=True)
        return

    result = blame_run(run_id, store=store)

    pname = trace.pipeline_name or "unnamed"
    click.echo(f"Analyzing run {trace.run_id} ({pname}, {len(trace.steps)} steps)...\n")

    if result.primary_step is None:
        click.echo("No steps found to analyze.")
        return

    pstep = result.primary_step
    p_num = pstep.index
    score_display = f"{pstep.score:.2f}" if pstep.score is not None else "N/A (unscored)"
    threshold_info = f"  (threshold: {WEAK_RETRIEVAL_THRESHOLD:.2f})" if pstep.step_type == "retrieval" else ""

    if result.is_fallback_latency:
        click.secho(f"[!] BLAME (Latency Fallback): Step [{p_num}] {pstep.name} ({pstep.step_type})", fg="yellow", bold=True)
    else:
        click.secho(f"[BLAME] Step [{p_num}] {pstep.name}  ({pstep.step_type})", fg="red", bold=True)

    click.echo(f"   Score:       {score_display}{threshold_info}")
    click.echo(f"   Blame score: {result.blame_score:.2f}  ({result.confidence} confidence)")
    click.echo(f"   Reason:      {result.explanation}")
    click.echo()

    if result.co_blamed:
        co_names = ", ".join(f"[{s.index}] {s.name} ({s.step_type})" for s in result.co_blamed)
        click.echo(f"Co-blame: {co_names}")
    else:
        click.echo("Co-blame: none")

    other_steps = [s for s in trace.steps if s.step_id != pstep.step_id and s not in result.co_blamed]
    if other_steps:
        other_formatted: list[str] = []
        for s in other_steps:
            if s.score is not None:
                if s.score >= 0.75:
                    other_formatted.append(f"{s.name} ({s.score:.2f} [OK])")
                elif s.score >= 0.50:
                    other_formatted.append(f"{s.name} ({s.score:.2f} [!])")
                else:
                    other_formatted.append(f"{s.name} ({s.score:.2f} [FAIL])")
            else:
                other_formatted.append(f"{s.name} (unscored)")
        click.echo(f"Other steps: {', '.join(other_formatted)}")


@cli.command(name="diff")
@click.argument("run_id_a")
@click.argument("run_id_b")
def diff_cmd(run_id_a: str, run_id_b: str) -> None:
    """Compare two execution runs to identify regressions or improvements."""
    from tracebackai.blame import diff_runs

    store = Store()
    try:
        trace_a = store.load_trace(run_id_a)
    except ValueError:
        click.secho(f"Error: Run ID '{run_id_a}' not found.", fg="red", err=True)
        return

    try:
        trace_b = store.load_trace(run_id_b)
    except ValueError:
        click.secho(f"Error: Run ID '{run_id_b}' not found.", fg="red", err=True)
        return

    diff_res = diff_runs(run_id_a, run_id_b, store=store)

    click.echo(f"Comparing {run_id_a} -> {run_id_b}")
    click.echo(f"Pipeline: {diff_res.pipeline_name}\n")

    header = f"{'STEP':<20} {'SCORE_A':<10} {'SCORE_B':<10} {'DELTA':<10} {'STATUS'}"
    click.echo(header)
    click.echo("-" * 65)

    all_matched = diff_res.regressed_steps + diff_res.improved_steps + diff_res.stable_steps
    for step_a, step_b, delta in all_matched:
        sa_str = f"{step_a.score:.2f}" if step_a.score is not None else "N/A"
        sb_str = f"{step_b.score:.2f}" if step_b.score is not None else "N/A"
        d_str = f"{delta:+.2f}"

        is_primary = step_b.name == diff_res.primary_diverged_step
        diverged_tag = " <-- highest delta" if is_primary and diff_res.verdict != "NEUTRAL" else ""

        if delta < -0.05:
            status_text = click.style(f"[-] REGRESSED{diverged_tag}", fg="red")
        elif delta > 0.05:
            status_text = click.style(f"[+] improved{diverged_tag}", fg="green")
        else:
            status_text = f"-> stable{diverged_tag}"

        click.echo(f"{step_b.name:<20} {sa_str:<10} {sb_str:<10} {d_str:<10} {status_text}")

    for step in diff_res.added_steps:
        sb_str = f"{step.score:.2f}" if step.score is not None else "N/A"
        status_text = click.style("+ ADDED", fg="cyan")
        click.echo(f"{step.name:<20} {'---':<10} {sb_str:<10} {'---':<10} {status_text}")

    for step in diff_res.removed_steps:
        sa_str = f"{step.score:.2f}" if step.score is not None else "N/A"
        status_text = click.style("- REMOVED", fg="magenta")
        click.echo(f"{step.name:<20} {sa_str:<10} {'---':<10} {'---':<10} {status_text}")

    click.echo()
    if diff_res.verdict == "REGRESSION":
        click.secho(f"Verdict: REGRESSION in {diff_res.primary_diverged_step}", fg="red", bold=True)
    elif diff_res.verdict == "IMPROVEMENT":
        click.secho(f"Verdict: IMPROVEMENT in {diff_res.primary_diverged_step}", fg="green", bold=True)
    else:
        click.echo("Verdict: NEUTRAL")

    if diff_res.explanation:
        click.echo(f"Details: {diff_res.explanation}")


@cli.command(name="run")
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to input JSON file.")
@click.option("--fail-on-blame", type=float, default=None, help="Exit with code 1 if blame score exceeds threshold.")
def run_cmd(script: str, input_path: Optional[str], fail_on_blame: Optional[float]) -> None:
    """Execute a Python script/pipeline and evaluate blame thresholds for CI eval gating."""
    import os
    import subprocess
    import sys
    from tracebackai.blame import blame_run

    env = os.environ.copy()
    if input_path:
        env["TRACEBACK_INPUT_PATH"] = os.path.abspath(input_path)

    cmd = [sys.executable, script]
    if input_path:
        cmd.extend(["--input", input_path])

    store = Store()
    runs_before = {r["run_id"] for r in store.list_runs(limit=100)}

    click.echo(f"Running pipeline script: {script}...")
    sub_res = subprocess.run(cmd, env=env)

    if sub_res.returncode != 0:
        click.secho(f"Script exited with non-zero code {sub_res.returncode}.", fg="red", err=True)
        sys.exit(sub_res.returncode)

    if fail_on_blame is not None:
        runs_after = store.list_runs(limit=10)
        new_runs = [r for r in runs_after if r["run_id"] not in runs_before]
        target_run = new_runs[0] if new_runs else (runs_after[0] if runs_after else None)

        if not target_run:
            click.secho("Eval Gate Error: No execution trace was recorded.", fg="yellow", err=True)
            sys.exit(1)

        result = blame_run(target_run["run_id"], store=store)
        pname = target_run.get("pipeline_name", "pipeline")
        click.echo(f"\nEval Gate Check: Run {target_run['run_id']} ({pname})")
        click.echo(f"Blame score: {result.blame_score:.2f} (Threshold: {fail_on_blame:.2f})")

        if result.blame_score > fail_on_blame:
            click.secho(
                f"[FAIL] Eval Gate Failed: Blame score {result.blame_score:.2f} exceeds threshold {fail_on_blame:.2f}.\n"
                f"Primary culprit: {result.primary_step.name if result.primary_step else 'Unknown'} "
                f"({result.explanation})",
                fg="red",
                bold=True,
                err=True,
            )
            sys.exit(1)
        else:
            click.secho(
                f"[OK] Eval Gate Passed: Pipeline health within acceptable threshold ({result.blame_score:.2f} <= {fail_on_blame:.2f}).",
                fg="green",
                bold=True,
            )


if __name__ == "__main__":
    cli()

