"""Deterministic execution-tree projection for finalized benchmark traces."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from hermes_cli.benchmarks.tracing.runtime import TraceIdentity


JsonObject = dict[str, Any]
EXECUTION_TREE_FORMAT = "benchmark-trace/execution-tree-v1"


@dataclass(slots=True)
class _Node:
    node_id: str
    event_type: str
    event_family: str
    status: str
    span_id: str
    parent_span_id: str | None
    source_event_ids: list[str]
    source_sequences: list[int]
    started_at: str | None
    ended_at: str | None
    duration_ms: float | None
    timing_fidelity: str
    completeness: str
    actor: JsonObject
    boundaries: list[JsonObject]
    input: JsonObject | None = None
    output: JsonObject | None = None
    data: JsonObject | None = None
    overlaps_with: list[str] = field(default_factory=list)
    concurrency_group: str | None = None
    children: list["_Node"] = field(default_factory=list)

    @property
    def first_sequence(self) -> int:
        return min(self.source_sequences)

    @property
    def interval(self) -> tuple[datetime, datetime] | None:
        if self.started_at is None or self.ended_at is None:
            return None
        start = _timestamp(self.started_at)
        end = _timestamp(self.ended_at)
        if end <= start:
            return None
        return start, end

    def finish(self, event: JsonObject) -> None:
        self.source_event_ids.append(_string(event, "event_id"))
        self.source_sequences.append(_integer(event, "sequence"))
        self.ended_at = _string(event, "occurred_at")
        self.duration_ms = _event_duration(event, self.started_at, self.ended_at)
        self.timing_fidelity = _timing_fidelity(event)
        self.status = _string(event, "status")
        self.completeness = "complete"
        self.boundaries.append(_boundary(event))
        self.output = _content(event)

    def as_json(self) -> JsonObject:
        value: JsonObject = {
            "node_id": self.node_id,
            "kind": "activity"
            if self.completeness == "complete"
            else self.completeness,
            "event_type": self.event_type,
            "event_family": self.event_family,
            "status": self.status,
            "span_id": self.span_id,
            "source_event_ids": list(self.source_event_ids),
            "source_sequences": list(self.source_sequences),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "timing_fidelity": self.timing_fidelity,
            "completeness": self.completeness,
            "actor": dict(self.actor),
            "boundaries": [dict(boundary) for boundary in self.boundaries],
            "overlaps_with": list(self.overlaps_with),
            "children": [child.as_json() for child in self.children],
        }
        if self.parent_span_id is not None:
            value["parent_span_id"] = self.parent_span_id
        if self.input is not None:
            value["input"] = dict(self.input)
        if self.output is not None:
            value["output"] = dict(self.output)
        if self.data is not None:
            value["data"] = dict(self.data)
        if self.concurrency_group is not None:
            value["concurrency_group"] = self.concurrency_group
        return value


def build_execution_tree(
    events: Sequence[JsonObject],
    *,
    identity: TraceIdentity,
    schema_digest: str,
    events_content: bytes,
    generated_at: str | None = None,
) -> JsonObject:
    """Pair normalized span boundaries and project them into one ordered tree."""

    if not events:
        if generated_at is None:
            raise ValueError(
                "An empty execution tree requires a finalization timestamp"
            )
        _timestamp(generated_at)
        return {
            "schema_version": "benchmark-trace/v1",
            "schema_digest": schema_digest,
            "format": EXECUTION_TREE_FORMAT,
            **identity.event_fields(),
            "generated_at": generated_at,
            "complete": True,
            "source": {
                "path": "events.jsonl",
                "sha256": hashlib.sha256(events_content).hexdigest(),
                "event_count": 0,
                "represented_event_count": 0,
            },
            "root": {
                "node_id": "trace-root",
                "started_at": None,
                "ended_at": None,
                "duration_ms": 0.0,
                "children": [],
            },
            "warnings": [],
        }
    ordered = sorted(events, key=lambda event: _integer(event, "sequence"))
    pending: dict[str, list[_Node]] = {}
    nodes: list[_Node] = []
    warnings: list[JsonObject] = []

    for event in ordered:
        phase = _string(event, "phase")
        span_id = _string(event, "span_id")
        if phase == "start":
            node = _start_node(event)
            nodes.append(node)
            pending.setdefault(span_id, []).append(node)
            continue
        if phase == "end":
            node = next(
                (
                    candidate
                    for candidate in pending.get(span_id, [])
                    if candidate.completeness == "start_only"
                ),
                None,
            )
            if node is None:
                nodes.append(_end_node(event))
                warnings.append(
                    _warning(
                        "projection.unmatched_end",
                        "An end event had no matching start boundary",
                        event_id=_string(event, "event_id"),
                        span_id=span_id,
                    )
                )
                continue
            node.finish(event)
            continue
        nodes.append(_instant_node(event))

    for node in nodes:
        if node.completeness != "start_only":
            continue
        warnings.append(
            _warning(
                "projection.unmatched_start",
                "A start event had no matching end boundary",
                event_id=node.source_event_ids[0],
                span_id=node.span_id,
            )
        )

    roots = _attach_nodes(nodes, warnings)
    _sort_nodes(roots)
    _mark_concurrency(roots)
    represented = [event_id for node in nodes for event_id in node.source_event_ids]
    generated_at = max(_string(event, "recorded_at") for event in ordered)
    occurred = [_timestamp(_string(event, "occurred_at")) for event in ordered]
    started_at = min(occurred)
    ended_at = max(occurred)
    return {
        "schema_version": "benchmark-trace/v1",
        "schema_digest": schema_digest,
        "format": EXECUTION_TREE_FORMAT,
        **identity.event_fields(),
        "generated_at": generated_at,
        "complete": not warnings and len(represented) == len(ordered),
        "source": {
            "path": "events.jsonl",
            "sha256": hashlib.sha256(events_content).hexdigest(),
            "event_count": len(ordered),
            "represented_event_count": len(represented),
        },
        "root": {
            "node_id": "trace-root",
            "started_at": _format_timestamp(started_at),
            "ended_at": _format_timestamp(ended_at),
            "duration_ms": max(0.0, (ended_at - started_at).total_seconds() * 1000),
            "children": [node.as_json() for node in roots],
        },
        "warnings": warnings,
    }


def execution_tree_event_ids(document: JsonObject) -> tuple[str, ...]:
    root = document.get("root")
    if not isinstance(root, Mapping):
        raise ValueError("Execution tree root is not an object")
    children = root.get("children")
    if not isinstance(children, list):
        raise ValueError("Execution tree children are not an array")
    event_ids: list[str] = []
    stack = list(reversed(children))
    while stack:
        node = stack.pop()
        if not isinstance(node, Mapping):
            raise ValueError("Execution tree contains a non-object node")
        source_event_ids = node.get("source_event_ids")
        nested = node.get("children")
        if not isinstance(source_event_ids, list) or not isinstance(nested, list):
            raise ValueError("Execution tree node is incomplete")
        if not all(isinstance(value, str) for value in source_event_ids):
            raise ValueError("Execution tree event identifier is invalid")
        event_ids.extend(value for value in source_event_ids if isinstance(value, str))
        stack.extend(reversed(nested))
    return tuple(event_ids)


def _start_node(event: JsonObject) -> _Node:
    return _Node(
        node_id=_string(event, "event_id"),
        event_type=_string(event, "event_type"),
        event_family=_string(event, "event_family"),
        status=_string(event, "status"),
        span_id=_string(event, "span_id"),
        parent_span_id=_optional_string(event, "parent_span_id"),
        source_event_ids=[_string(event, "event_id")],
        source_sequences=[_integer(event, "sequence")],
        started_at=_string(event, "occurred_at"),
        ended_at=None,
        duration_ms=None,
        timing_fidelity=_timing_fidelity(event),
        completeness="start_only",
        actor=_actor(event),
        boundaries=[_boundary(event)],
        input=_content(event),
    )


def _end_node(event: JsonObject) -> _Node:
    ended_at = _string(event, "occurred_at")
    return _Node(
        node_id=_string(event, "event_id"),
        event_type=_string(event, "event_type"),
        event_family=_string(event, "event_family"),
        status=_string(event, "status"),
        span_id=_string(event, "span_id"),
        parent_span_id=_optional_string(event, "parent_span_id"),
        source_event_ids=[_string(event, "event_id")],
        source_sequences=[_integer(event, "sequence")],
        started_at=None,
        ended_at=ended_at,
        duration_ms=_event_duration(event, None, ended_at),
        timing_fidelity=_timing_fidelity(event),
        completeness="end_only",
        actor=_actor(event),
        boundaries=[_boundary(event)],
        output=_content(event),
    )


def _instant_node(event: JsonObject) -> _Node:
    occurred_at = _string(event, "occurred_at")
    return _Node(
        node_id=_string(event, "event_id"),
        event_type=_string(event, "event_type"),
        event_family=_string(event, "event_family"),
        status=_string(event, "status"),
        span_id=_string(event, "span_id"),
        parent_span_id=_optional_string(event, "parent_span_id"),
        source_event_ids=[_string(event, "event_id")],
        source_sequences=[_integer(event, "sequence")],
        started_at=occurred_at,
        ended_at=occurred_at,
        duration_ms=0.0,
        timing_fidelity=_timing_fidelity(event),
        completeness="instant",
        actor=_actor(event),
        boundaries=[_boundary(event)],
        data=_content(event),
    )


def _attach_nodes(nodes: list[_Node], warnings: list[JsonObject]) -> list[_Node]:
    primary_by_span: dict[str, _Node] = {}
    for node in nodes:
        current = primary_by_span.get(node.span_id)
        if current is None or (
            current.completeness != "complete" and node.completeness == "complete"
        ):
            primary_by_span[node.span_id] = node

    roots: list[_Node] = []
    for node in nodes:
        if node.parent_span_id is None:
            roots.append(node)
            continue
        parent = primary_by_span.get(node.parent_span_id)
        if parent is None:
            roots.append(node)
            warnings.append(
                _warning(
                    "projection.orphan_parent",
                    "A node referenced a parent span absent from the event stream",
                    event_id=node.source_event_ids[0],
                    span_id=node.span_id,
                )
            )
            continue
        if parent is node or _would_cycle(node, parent, primary_by_span):
            roots.append(node)
            warnings.append(
                _warning(
                    "projection.parent_cycle",
                    "A cyclic parent relationship was moved to the trace root",
                    event_id=node.source_event_ids[0],
                    span_id=node.span_id,
                )
            )
            continue
        parent.children.append(node)
    return roots


def _would_cycle(
    node: _Node,
    parent: _Node,
    primary_by_span: dict[str, _Node],
) -> bool:
    current: _Node | None = parent
    seen: set[str] = set()
    while current is not None:
        if current is node or current.node_id in seen:
            return True
        seen.add(current.node_id)
        current = (
            primary_by_span.get(current.parent_span_id)
            if current.parent_span_id is not None
            else None
        )
    return False


def _sort_nodes(nodes: list[_Node]) -> None:
    nodes.sort(
        key=lambda node: (
            _timestamp(node.started_at or node.ended_at or "1970-01-01T00:00:00Z"),
            node.first_sequence,
        )
    )
    for node in nodes:
        _sort_nodes(node.children)


def _mark_concurrency(roots: list[_Node]) -> None:
    next_group = 1
    stack: list[list[_Node]] = [roots]
    while stack:
        siblings = stack.pop()
        adjacency = {node.node_id: set[str]() for node in siblings}
        for index, left in enumerate(siblings):
            for right in siblings[index + 1 :]:
                if _overlap(left, right):
                    adjacency[left.node_id].add(right.node_id)
                    adjacency[right.node_id].add(left.node_id)
        visited: set[str] = set()
        by_id = {node.node_id: node for node in siblings}
        for node in siblings:
            if node.node_id in visited or not adjacency[node.node_id]:
                continue
            component: list[_Node] = []
            pending = [node.node_id]
            while pending:
                node_id = pending.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                component.append(by_id[node_id])
                pending.extend(sorted(adjacency[node_id], reverse=True))
            group = f"concurrency-{next_group:06d}"
            next_group += 1
            for member in component:
                member.concurrency_group = group
                member.overlaps_with = sorted(
                    adjacency[member.node_id],
                    key=lambda node_id: by_id[node_id].first_sequence,
                )
        stack.extend(reversed([node.children for node in siblings]))


def _overlap(left: _Node, right: _Node) -> bool:
    left_interval = left.interval
    right_interval = right.interval
    if left_interval is None or right_interval is None:
        return False
    return left_interval[0] < right_interval[1] and right_interval[0] < left_interval[1]


def _boundary(event: JsonObject) -> JsonObject:
    value: JsonObject = {
        "event_id": _string(event, "event_id"),
        "sequence": _integer(event, "sequence"),
        "event_type": _string(event, "event_type"),
        "phase": _string(event, "phase"),
        "status": _string(event, "status"),
        "occurred_at": _string(event, "occurred_at"),
        "recorded_at": _string(event, "recorded_at"),
        "origin": _object(event, "origin"),
        "timing": _object(event, "timing"),
    }
    relations = event.get("relations")
    if isinstance(relations, list):
        value["relations"] = relations
    return value


def _content(event: JsonObject) -> JsonObject:
    value: JsonObject = {
        "payload": _object(event, "payload"),
        "artifacts": _array(event, "artifacts"),
    }
    error = event.get("error")
    if isinstance(error, Mapping):
        value["error"] = dict(error)
    return value


def _actor(event: JsonObject) -> JsonObject:
    return {
        key: value
        for key in ("session_id", "agent_id", "parent_agent_id", "turn_id")
        if isinstance((value := event.get(key)), str)
    }


def _event_duration(
    event: JsonObject,
    started_at: str | None,
    ended_at: str,
) -> float | None:
    duration = _object(event, "timing").get("duration_ms")
    if (
        isinstance(duration, int | float)
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 0
    ):
        return float(duration)
    if started_at is None:
        return None
    return max(
        0.0,
        (_timestamp(ended_at) - _timestamp(started_at)).total_seconds() * 1000,
    )


def _timing_fidelity(event: JsonObject) -> str:
    return _string(_object(event, "timing"), "fidelity")


def _warning(
    code: str,
    message: str,
    *,
    event_id: str,
    span_id: str,
) -> JsonObject:
    return {
        "code": code,
        "message": message,
        "event_id": event_id,
        "span_id": span_id,
    }


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Execution tree contains an invalid timestamp") from exc
    return parsed.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"Execution tree source field {key} is not a string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"Execution tree source field {key} is invalid")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"Execution tree source field {key} is not an integer")
    return item


def _object(value: Mapping[str, Any], key: str) -> JsonObject:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"Execution tree source field {key} is not an object")
    return dict(item)


def _array(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"Execution tree source field {key} is not an array")
    return item
