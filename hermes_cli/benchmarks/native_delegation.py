"""Post-run auditing for benchmark workflows using native Hermes delegation."""

from __future__ import annotations

import json
from typing import Any, Sequence


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def audit_native_delegations(
    messages: Any,
    *,
    phase_order: Sequence[str],
    goal_markers: dict[str, str],
    subagent_budget: int,
    max_report_chars: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Summarize native delegate_task calls without controlling their order."""
    if not isinstance(messages, list):
        return [], ["Coordinator messages are unavailable"]
    records: list[dict[str, Any]] = []
    by_call_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if (
                    not isinstance(function, dict)
                    or function.get("name") != "delegate_task"
                ):
                    continue
                call_id = str(call.get("id") or f"delegate-{len(records)}")
                raw_arguments = function.get("arguments")
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    try:
                        arguments = json.loads(str(raw_arguments or "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                goal = arguments.get("goal") if isinstance(arguments, dict) else None
                lowered_goal = goal.lower() if isinstance(goal, str) else ""
                phase = next(
                    (
                        name
                        for name, marker in goal_markers.items()
                        if marker in lowered_goal
                    ),
                    None,
                )
                record = {
                    "phase": phase,
                    "budget": subagent_budget,
                    "status": "missing-result",
                    "freshAgent": True,
                    "nativeDelegation": True,
                    "role": arguments.get("role", "leaf")
                    if isinstance(arguments, dict)
                    else None,
                    "report": "",
                    "apiCalls": 0,
                    "error": None,
                }
                records.append(record)
                by_call_id[call_id] = record
                if phase is None:
                    errors.append(
                        "Native delegate_task goal is missing a benchmark role marker"
                    )
                if isinstance(arguments, dict) and arguments.get("tasks"):
                    errors.append("Native delegation used batch mode")
        if message.get("role") != "tool":
            continue
        record = by_call_id.get(str(message.get("tool_call_id") or ""))
        if record is None:
            continue
        try:
            payload = json.loads(_content_text(message.get("content")))
        except json.JSONDecodeError:
            record["status"] = "failed"
            record["error"] = "Native delegation returned invalid JSON"
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != 1:
            record["status"] = "failed"
            record["error"] = "Native delegation did not return exactly one child"
            continue
        child = results[0]
        if not isinstance(child, dict):
            record["status"] = "failed"
            record["error"] = "Native delegation child result is invalid"
            continue
        record.update(
            {
                "status": child.get("status"),
                "report": str(child.get("summary") or "")[:max_report_chars],
                "apiCalls": child.get("api_calls", 0),
                "error": child.get("error"),
                "durationSeconds": child.get("duration_seconds"),
                "liveTranscript": child.get("live_transcript"),
            }
        )

    observed = [record.get("phase") for record in records]
    if observed != list(phase_order):
        errors.append(
            f"Expected native delegation order {list(phase_order)!r}; "
            f"observed {observed!r}"
        )
    for record in records:
        if record.get("status") != "completed":
            errors.append(
                f"Native {record.get('phase') or 'unknown'} delegation did not complete"
            )
        if record.get("role") != "leaf":
            errors.append(
                f"Native {record.get('phase') or 'unknown'} delegation was not a leaf"
            )
    return records, list(dict.fromkeys(errors))
