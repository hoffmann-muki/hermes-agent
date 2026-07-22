"""Per-instance Hermes worker specialization for SWE-bench Pro."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from hermes_cli.benchmarks import swebench_verified_worker as shared
from hermes_cli.benchmarks.swebench_pro import (
    BENCHMARK,
    SweBenchProRow,
    format_problem_statement,
    parse_swebench_pro_row,
)


WORKTREE = "/app"

COORDINATOR_SYSTEM_PROMPT = """You are benchmark-coordinator for a SWE-bench Pro task.

You own the result, but three fresh specialists must work in the same foreground
workspace before you finish. Call swe_benchmark_phase exactly once for each role,
one call at a time, in this exact order:

1. navigator — investigate only; it is read-only and returns an evidence-backed plan.
2. patcher — implement the fix and run focused checks.
3. reviewer — independently inspect the diff/tests and make small corrective edits.

Wait for each call to return before making the next. Do not skip, repeat, parallelize,
or replace a phase with your own analysis. Do not invoke Hermes' general delegation.
After reviewer returns, reconcile all reports, inspect the final worktree, make only
necessary final corrections, run feasible verification, and provide a concise final
summary. The source edits in /app—not prose—are the benchmark answer.

Never seek a gold patch, hidden test patch, benchmark answer, or hidden grading data.
Do not modify tests or benchmark metadata unless the issue explicitly requires it.
"""

PHASE_SYSTEM_PROMPTS = {
    "navigator": """You are benchmark-navigator, a fresh read-only SWE specialist.
Investigate the issue and repository in /app. Trace the relevant implementation,
tests, and likely root cause. Do not modify files, install into the repository, or run
destructive commands. Return a concrete implementation and verification plan with
paths and symbols. You cannot delegate.""",
    "patcher": """You are benchmark-patcher, a fresh SWE implementation specialist.
Work directly in /app. Use the navigator evidence, inspect the code yourself,
implement the smallest correct fix, and run focused verification when feasible.
Do not seek benchmark answers or hidden tests. You cannot delegate. Leave all source
changes in the shared worktree and report changed paths, commands, and remaining risk.""",
    "reviewer": """You are benchmark-reviewer, a fresh independent code reviewer.
Inspect the issue, current /app worktree, and patcher report. Review the full diff
for correctness, regressions, style, and missing edge cases. Run focused checks and
make small corrective edits directly when justified. Do not rewrite a sound solution
for preference alone. You cannot delegate. Report findings, fixes, verification, and
residual risk.""",
}

PHASE_TOOL_SCHEMA: dict[str, Any] = {
    "description": (
        "Run the next required fresh SWE-bench Pro specialist synchronously in the "
        "shared /app worktree. Call navigator, then patcher, then reviewer, exactly "
        "once each and wait for every result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": list(shared.PHASE_ORDER),
                "description": "The next required benchmark phase.",
            }
        },
        "required": ["phase"],
        "additionalProperties": False,
    },
}


def phase_user_prompt(
    phase: str,
    row: SweBenchProRow,
    previous_records: Sequence[dict[str, Any]],
    *,
    include_hints: bool,
) -> str:
    del include_hints
    handoffs = []
    for record in previous_records:
        report = str(record.get("report") or "").strip()
        if report:
            handoffs.extend([
                f"### {record['phase'].title()} handoff",
                report[: shared.MAX_PHASE_REPORT_CHARS],
                "",
            ])
    issue = [
        f"Complete the {phase} phase for SWE-bench Pro instance {row.instance_id}.",
        f"Repository: {row.repo}",
        f"Base commit: {row.base_commit}",
        f"Repository language: {row.repo_language}",
        "Worktree: /app",
        "",
        "## Issue",
        format_problem_statement(row),
        "",
    ]
    return "\n".join([*issue, *handoffs])


@contextmanager
def configuration() -> Iterator[None]:
    with shared.worker_configuration(
        benchmark=BENCHMARK,
        worktree=WORKTREE,
        conda_activation="",
        row_parser=parse_swebench_pro_row,
        coordinator_system_prompt=COORDINATOR_SYSTEM_PROMPT,
        phase_system_prompts=PHASE_SYSTEM_PROMPTS,
        phase_tool_schema=PHASE_TOOL_SCHEMA,
        phase_prompt_builder=phase_user_prompt,
    ):
        yield


def run_worker(
    request: dict[str, Any],
    *,
    agent_factory: Callable[..., Any] = shared.default_agent_factory,
) -> dict[str, Any]:
    with configuration():
        return shared.run_worker(request, agent_factory=agent_factory)


def main(argv: Sequence[str] | None = None) -> int:
    with configuration():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
