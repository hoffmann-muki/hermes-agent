"""Durable ``benchmark-trace/v1`` recorder used by Hermes benchmark adapters."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast
from urllib.parse import quote
from uuid import uuid4


SCHEMA_VERSION = "benchmark-trace/v1"
CONTRACT_VERSION = "1.1.0"
SCHEMA_DIGEST = "12121cb7fbdb81b1637954eefab17b1faaf39ecdff1ed4fe0d67065941ca4b17"
NATIVE_CHUNK_MEDIA_TYPE = "application/vnd.benchmark-trace.native-records+jsonl+gzip"
NATIVE_JOURNAL_FORMAT = "benchmark-trace/native-journal-v1"
NATIVE_CHUNK_TARGET_BYTES = 1024 * 1024
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
CAPABILITY_CATEGORIES = (
    "agent.session",
    "model.turn",
    "provider.exchange",
    "tool.invocation",
    "tool.result",
    "tool.timing",
    "shell",
    "file",
    "search",
    "browser",
    "delegation",
    "context.compaction",
    "memory",
    "harness.lifecycle",
    "container.lifecycle",
    "evaluator.lifecycle",
    "patch",
    "native.evidence",
)
_EVENT_FAMILIES = {
    "run",
    "instance",
    "attempt",
    "harness",
    "container",
    "evaluator",
    "agent",
    "model",
    "provider",
    "tool",
    "shell",
    "file",
    "search",
    "browser",
    "delegation",
    "context",
    "memory",
    "patch",
    "trace",
}
_EVENT_STATUSES = {
    "started",
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "degraded",
    "unknown",
}
_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "degraded",
}
_CAPTURE_METHODS = {
    "native_hook",
    "native_stream",
    "native_export",
    "derived",
    "generic_harness",
}
_TIMING_FIDELITIES = {
    "native_monotonic",
    "native_wall",
    "derived",
    "not_available",
}
_RELATION_TYPES = {
    "caused_by",
    "contains",
    "derived_from",
    "native_evidence",
    "retry_of",
}

JsonObject = dict[str, Any]
TraceStatus = Literal["completed", "failed", "cancelled", "timeout", "degraded"]


class TraceInitializationError(RuntimeError):
    """Raised before agent work when private trace storage cannot be initialized."""


class TraceStorageError(RuntimeError):
    """Raised when durable trace storage cannot be read or written safely."""


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    run_id: str
    benchmark: str
    framework: str
    instance_id: str
    attempt: int

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        benchmark: str,
        framework: str,
        instance_id: str,
        attempt: int,
    ) -> "TraceIdentity":
        return cls(
            trace_id=f"trace-{uuid4().hex}",
            run_id=run_id,
            benchmark=benchmark,
            framework=framework,
            instance_id=instance_id,
            attempt=attempt,
        )

    def event_fields(self) -> JsonObject:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "framework": self.framework,
            "instance_id": self.instance_id,
            "attempt": self.attempt,
        }


@dataclass(frozen=True)
class Capability:
    category: str
    state: str
    coverage: str
    timing: str
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def as_json(self) -> JsonObject:
        return {
            "category": self.category,
            "state": self.state,
            "coverage": self.coverage,
            "timing": self.timing,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class TraceConfig:
    attempt_dir: Path
    identity: TraceIdentity
    producer: JsonObject
    provenance: JsonObject
    execution: JsonObject
    capabilities: tuple[Capability, ...]


@dataclass(frozen=True)
class TraceFinalization:
    trace_id: str
    attempt_dir: str
    health: str
    complete: bool


@dataclass(frozen=True)
class _RedactionResult:
    value: Any
    matches: int
    rules: tuple[str, ...]


@dataclass(frozen=True)
class _PatternRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]


_CREDENTIAL_FIELDS = {
    "apikey",
    "accesstoken",
    "authorization",
    "authorizationheader",
    "authtoken",
    "clientsecret",
    "cookie",
    "credentials",
    "githubtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretkey",
    "signedcredential",
}
_CREDENTIAL_FIELD_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesstoken",
    "authorization",
    "authorizationheader",
    "authtoken",
    "clientsecret",
    "cookie",
    "credentials",
    "githubtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretaccesskey",
    "secretkey",
    "signedcredential",
)
_ACCOUNTING_FIELDS = {
    "accumulatedcost",
    "accumulatedtokenusage",
    "cachereadtokens",
    "cachewritetokens",
    "cachedtokens",
    "completiontokens",
    "cost",
    "costusd",
    "currency",
    "estimatedcost",
    "estimatedcostusd",
    "inputtokens",
    "outputtokens",
    "price",
    "prompttokens",
    "reasoningtokens",
    "tokencount",
    "tokens",
    "totalcost",
    "totalcostusd",
    "totaltokens",
    "usage",
    "usagesummary",
    "usagetometrics",
}
_ACCOUNTING_FIELD_SUFFIXES = tuple(_ACCOUNTING_FIELDS)
_CREDENTIAL_NAME_PATTERN = (
    r"(?:[A-Za-z_][A-Za-z0-9_-]*?)?"
    r"(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
    r"authorization(?:[_-]?header)?|client[_-]?secret|cookie|credentials|"
    r"github[_-]?token|password|private[_-]?key|refresh[_-]?token|"
    r"secret(?:[_-]?access)?[_-]?key|secret|signed[_-]?credential)"
)
_PATTERN_RULES = (
    _PatternRule(
        "credential.private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "<redacted:private_key>",
    ),
    _PatternRule(
        "credential.authorization",
        re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
        "<redacted:authorization>",
    ),
    _PatternRule(
        "credential.model_api_key",
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"),
        "<redacted:model_api_key>",
    ),
    _PatternRule(
        "credential.github_token",
        re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
        "<redacted:github_token>",
    ),
    _PatternRule(
        "credential.cloud_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "<redacted:cloud_access_key>",
    ),
    _PatternRule(
        "credential.uri_password",
        re.compile(
            r"(?P<scheme>[a-z][a-z0-9+.-]*://)"
            r"(?P<username>[^:/@\s]+):(?!<redacted:)"
            r"(?P<password>[^/@\s]+)@",
            re.IGNORECASE,
        ),
        lambda match: (
            f"{match.group('scheme')}{match.group('username')}:<redacted:uri_password>@"
        ),
    ),
    _PatternRule(
        "credential.assignment",
        re.compile(
            rf"(?i)(?<![A-Za-z0-9_])(?P<option>--?)?"
            rf"(?P<name>{_CREDENTIAL_NAME_PATTERN})"
            r"\s*(?:=|\s)\s*(?!<redacted:)"
            r"(?P<quote>['\"]?)(?P<value>[^\s'\"]{4,})(?P=quote)"
        ),
        lambda match: (
            f"{match.group('option') or ''}{match.group('name')}=<redacted:assignment>"
        ),
    ),
)


class _Redactor:
    def sanitize(self, value: Any) -> _RedactionResult:
        rules: list[str] = []
        matches = [0]

        def add(rule: str, count: int = 1) -> None:
            if count < 1:
                return
            matches[0] += count
            if rule not in rules:
                rules.append(rule)

        def text(raw: str) -> str:
            result = raw
            for rule in _PATTERN_RULES:
                result, count = rule.pattern.subn(rule.replacement, result)
                add(rule.name, count)
            return result

        def walk(raw: Any) -> Any:
            if isinstance(raw, Mapping):
                result: JsonObject = {}
                for key, item in raw.items():
                    name = str(key)
                    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
                    if _matches_field(
                        normalized,
                        _CREDENTIAL_FIELDS,
                        _CREDENTIAL_FIELD_SUFFIXES,
                    ):
                        add("field.credential")
                        continue
                    if _matches_field(
                        normalized,
                        _ACCOUNTING_FIELDS,
                        _ACCOUNTING_FIELD_SUFFIXES,
                    ):
                        add("field.accounting")
                        continue
                    result[name] = walk(item)
                return result
            if isinstance(raw, (list, tuple)):
                return [walk(item) for item in raw]
            if isinstance(raw, str):
                return text(raw)
            if raw is None or isinstance(raw, (bool, int, float)):
                return raw
            return text(str(raw))

        return _RedactionResult(walk(value), matches[0], tuple(rules))

    def sanitize_text(self, value: str) -> _RedactionResult:
        result = self.sanitize(value)
        return _RedactionResult(str(result.value), result.matches, result.rules)


def _matches_field(
    normalized: str,
    exact: set[str],
    suffixes: tuple[str, ...],
) -> bool:
    return normalized in exact or any(
        normalized != suffix and normalized.endswith(suffix) for suffix in suffixes
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    try:
        content = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TraceStorageError("Trace value is not canonical JSON") from exc
    return content.encode("utf-8") + b"\n"


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 512


def _valid_slug(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) <= 128
        and re.fullmatch(r"[a-z][a-z0-9._-]*", value)
    )


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_event_contract_fields(
    *,
    event_type: object,
    event_family: object,
    phase: object,
    status: object,
    event_id: object,
    span_id: object,
    occurred_at: object,
    origin: object,
    timing: object,
    error: object,
    relations: Sequence[object],
    identifiers: Sequence[object],
) -> bool:
    if (
        not isinstance(event_type, str)
        or len(event_type) > 128
        or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", event_type)
        or not isinstance(event_family, str)
        or event_family not in _EVENT_FAMILIES
        or not isinstance(phase, str)
        or phase not in {"start", "end", "instant"}
        or not isinstance(status, str)
        or status not in _EVENT_STATUSES
        or (phase == "start" and status != "started")
        or (phase == "end" and status not in _TERMINAL_STATUSES)
        or (phase == "instant" and status == "started")
        or not _valid_identifier(event_id)
        or not _valid_identifier(span_id)
        or any(
            value is not None and not _valid_identifier(value) for value in identifiers
        )
        or not _valid_timestamp(occurred_at)
        or not _valid_origin(origin)
        or not _valid_timing(timing, phase)
        or not _valid_error(error)
        or not _valid_relations(relations)
    ):
        return False
    return True


def _valid_origin(value: object) -> bool:
    if not isinstance(value, Mapping) or not set(value).issubset({
        "component",
        "capture_method",
        "native_event_type",
        "native_event_id",
    }):
        return False
    origin = cast(Mapping[str, object], value)
    component = origin.get("component")
    capture_method = origin.get("capture_method")
    native_event_type = origin.get("native_event_type")
    native_event_id = origin.get("native_event_id")
    return bool(
        isinstance(component, str)
        and 0 < len(component) <= 256
        and isinstance(capture_method, str)
        and capture_method in _CAPTURE_METHODS
        and (
            "native_event_type" not in origin
            or isinstance(native_event_type, str)
            and 0 < len(native_event_type) <= 256
        )
        and ("native_event_id" not in origin or _valid_identifier(native_event_id))
    )


def _valid_timing(value: object, phase: object) -> bool:
    if not isinstance(value, Mapping) or not set(value).issubset({
        "fidelity",
        "clock_id",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_ms",
    }):
        return False
    timing = cast(Mapping[str, object], value)
    fidelity = timing.get("fidelity")
    clock_id = timing.get("clock_id")
    started = timing.get("started_monotonic_ns")
    ended = timing.get("ended_monotonic_ns")
    duration = timing.get("duration_ms")
    if not isinstance(fidelity, str) or fidelity not in _TIMING_FIDELITIES:
        return False
    if fidelity == "native_monotonic" and (
        not isinstance(clock_id, str) or not 0 < len(clock_id) <= 256
    ):
        return False
    if fidelity == "not_available" and len(timing) != 1:
        return False
    if started is not None and (
        isinstance(started, bool) or not isinstance(started, int) or started < 0
    ):
        return False
    if ended is not None and (
        isinstance(ended, bool) or not isinstance(ended, int) or ended < 0
    ):
        return False
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration < 0
    ):
        return False
    if started is not None and fidelity != "native_monotonic":
        return False
    if ended is not None and (
        fidelity != "native_monotonic" or started is None or duration is None
    ):
        return False
    if phase == "end" and fidelity != "not_available" and duration is None:
        return False
    return True


def _valid_error(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping) or not set(value).issubset({
        "code",
        "message",
        "artifact",
    }):
        return False
    error = cast(Mapping[str, object], value)
    code = error.get("code")
    return bool(
        isinstance(code, str)
        and re.fullmatch(r"[a-z][a-z0-9._-]*", code)
        and isinstance(error.get("message"), str)
    )


def _valid_relations(values: Sequence[object]) -> bool:
    for value in values:
        if not isinstance(value, Mapping):
            return False
        relation = cast(Mapping[str, object], value)
        if (
            set(relation) != {"type", "event_id"}
            or not isinstance(relation.get("type"), str)
            or relation.get("type") not in _RELATION_TYPES
            or not _valid_identifier(relation.get("event_id"))
        ):
            return False
    return True


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise TraceStorageError(f"Trace directory cannot be a symbolic link: {path}")
    path.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    if not path.is_dir():
        raise TraceStorageError(f"Trace path is not a directory: {path}")
    path.chmod(DIRECTORY_MODE)


def _create_private_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise TraceStorageError(f"Trace directory already exists: {path}")
    parent = path.parent
    _ensure_private_directory(parent)
    path.mkdir(mode=DIRECTORY_MODE)
    path.chmod(DIRECTORY_MODE)


def _atomic_write(path: Path, content: bytes) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, FILE_MODE)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("write returned zero bytes")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, FILE_MODE)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        path.chmod(FILE_MODE)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise TraceStorageError(f"Could not write trace file: {path}") from exc


def _replace_with_hard_link(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise TraceStorageError(f"Trace link source is not a safe file: {source}")
    temporary = target.parent / f".{target.name}.{uuid4().hex}.link"
    try:
        os.link(source, temporary, follow_symlinks=False)
        temporary.chmod(FILE_MODE)
        os.replace(temporary, target)
        target.chmod(FILE_MODE)
        descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        temporary.unlink(missing_ok=True)
        _atomic_write(target, source.read_bytes())


class _Journal:
    def __init__(self, path: Path) -> None:
        _ensure_private_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self._descriptor: int | None = os.open(path, flags, FILE_MODE)
            os.fchmod(self._descriptor, FILE_MODE)
        except OSError as exc:
            raise TraceStorageError(f"Could not create trace journal: {path}") from exc
        self.path = path
        self._lock = threading.Lock()

    def append(self, value: JsonObject) -> None:
        content = canonical_json(value)
        with self._lock:
            if self._descriptor is None:
                raise TraceStorageError(f"Trace journal is closed: {self.path}")
            try:
                view = memoryview(content)
                while view:
                    written = os.write(self._descriptor, view)
                    if written == 0:
                        raise OSError("write returned zero bytes")
                    view = view[written:]
                os.fsync(self._descriptor)
            except OSError as exc:
                raise TraceStorageError(
                    f"Could not append trace journal: {self.path}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._descriptor is None:
                return
            descriptor = self._descriptor
            self._descriptor = None
            try:
                os.fsync(descriptor)
                os.close(descriptor)
            except OSError as exc:
                raise TraceStorageError(
                    f"Could not close trace journal: {self.path}"
                ) from exc


class TraceRecorder:
    """One durable attempt recorder with fail-open post-initialization writes."""

    def __init__(self, config: TraceConfig) -> None:
        self._config = config
        self._redactor = _Redactor()
        self._created_at = utc_now()
        self._event_sequence = 0
        self._native_sequence = 0
        self._event_ids: set[str] = set()
        self._native_ids: set[str] = set()
        self._artifacts: dict[str, JsonObject] = {}
        self._issues: dict[str, JsonObject] = {}
        self._redactions = 0
        self._dropped_events = 0
        self._lock = threading.RLock()
        self._finalized: TraceFinalization | None = None
        self._events: _Journal | None = None
        self._native: _Journal | None = None
        try:
            self._prepare_config()
            _create_private_directory(config.attempt_dir)
            _ensure_private_directory(config.attempt_dir / "native")
            _ensure_private_directory(config.attempt_dir / "artifacts")
            _ensure_private_directory(config.attempt_dir / "artifacts" / "sha256")
            self._events = _Journal(config.attempt_dir / "journal.jsonl")
            self._native = _Journal(config.attempt_dir / "native" / "index.jsonl")
            self._write_preflight()
        except (OSError, TypeError, ValueError, TraceStorageError) as exc:
            for journal in (self._events, self._native):
                if journal is None:
                    continue
                try:
                    journal.close()
                except TraceStorageError:
                    pass
            raise TraceInitializationError(
                "Hermes benchmark tracing could not be initialized safely"
            ) from exc

    @property
    def identity(self) -> TraceIdentity:
        return self._config.identity

    @property
    def attempt_dir(self) -> Path:
        return self._config.attempt_dir

    def _prepare_config(self) -> None:
        identity = self.identity
        if (
            not all((
                identity.trace_id,
                identity.run_id,
                identity.benchmark,
                identity.framework,
                identity.instance_id,
            ))
            or identity.attempt < 1
            or not all(
                _valid_identifier(value)
                for value in (
                    identity.trace_id,
                    identity.run_id,
                    identity.instance_id,
                )
            )
            or not _valid_slug(identity.benchmark)
            or not _valid_slug(identity.framework)
        ):
            raise ValueError("Trace identity is incomplete")
        if self._redactor.sanitize(identity.event_fields()).matches:
            raise ValueError("Trace identity contains credential-like material")
        categories = [item.category for item in self._config.capabilities]
        if len(categories) != len(set(categories)) or set(categories) != set(
            CAPABILITY_CATEGORIES
        ):
            raise ValueError("Trace capability matrix must be exhaustive and unique")
        producer = self._redactor.sanitize(self._config.producer)
        provenance = self._redactor.sanitize(self._config.provenance)
        execution = self._redactor.sanitize(self._config.execution)
        self._producer = producer.value
        self._provenance = provenance.value
        self._execution = execution.value
        capabilities = [
            self._redactor.sanitize(item.as_json())
            for item in self._config.capabilities
        ]
        self._capabilities = tuple(item.value for item in capabilities)
        self._redactions += (
            producer.matches
            + provenance.matches
            + execution.matches
            + sum(item.matches for item in capabilities)
        )

    def report_issue(self, code: str, message: str, *, severity: str = "error") -> None:
        with self._lock:
            timestamp = utc_now()
            safe_code = (
                code
                if re.fullmatch(r"[a-z][a-z0-9._-]*", code)
                else "adapter.invalid_issue_code"
            )
            sanitized = self._redactor.sanitize_text(message)
            self._redactions += sanitized.matches
            existing = self._issues.get(safe_code)
            if existing is None:
                self._issues[safe_code] = {
                    "severity": "warning" if severity == "warning" else "error",
                    "code": safe_code,
                    "message": sanitized.value or "Trace adapter failure",
                    "first_seen_at": timestamp,
                    "last_seen_at": timestamp,
                    "count": 1,
                }
                return
            existing["last_seen_at"] = timestamp
            existing["count"] = int(existing["count"]) + 1
            existing["message"] = sanitized.value or existing["message"]
            if severity != "warning":
                existing["severity"] = "error"

    def update_capabilities(self, capabilities: tuple[Capability, ...]) -> bool:
        with self._lock:
            previous = self._capabilities
            try:
                categories = [item.category for item in capabilities]
                if len(categories) != len(set(categories)) or set(categories) != set(
                    CAPABILITY_CATEGORIES
                ):
                    raise ValueError("Capability matrix is incomplete")
                sanitized = [
                    self._redactor.sanitize(item.as_json()) for item in capabilities
                ]
                self._capabilities = tuple(item.value for item in sanitized)
                self._write_preflight()
                self._redactions += sum(item.matches for item in sanitized)
                return True
            except (OSError, TypeError, ValueError, TraceStorageError):
                self._capabilities = previous
                self.report_issue(
                    "capabilities.update_failed",
                    "Hermes submitted an invalid observed capability matrix",
                )
                return False

    def _write_preflight(self) -> None:
        _atomic_write(
            self.attempt_dir / "preflight.json",
            canonical_json({
                "format": "benchmark-trace/preflight-v1",
                "created_at": self._created_at,
                "identity": self.identity.event_fields(),
                "producer": dict(self._producer),
                "provenance": dict(self._provenance),
                "execution": dict(self._execution),
                "capabilities": [dict(capability) for capability in self._capabilities],
            }),
        )

    def store_text_artifact(
        self, value: str, *, role: str, media_type: str = "text/plain"
    ) -> JsonObject | None:
        sanitized = self._redactor.sanitize_text(value)
        return self._store_artifact(
            sanitized.value.encode("utf-8"),
            role=role,
            media_type=media_type,
            encoding="utf-8",
            redaction=sanitized,
        )

    def store_json_artifact(
        self, value: Any, *, role: str, media_type: str = "application/json"
    ) -> JsonObject | None:
        sanitized = self._redactor.sanitize(value)
        try:
            content = canonical_json(sanitized.value)
        except TraceStorageError:
            self.report_issue(
                "artifact.serialization_failed",
                "A Hermes trace artifact could not be serialized",
            )
            return None
        return self._store_artifact(
            content,
            role=role,
            media_type=media_type,
            encoding="utf-8",
            redaction=sanitized,
        )

    def _store_artifact(
        self,
        content: bytes,
        *,
        role: str,
        media_type: str,
        encoding: str,
        redaction: _RedactionResult,
        count_redactions: bool = True,
    ) -> JsonObject | None:
        with self._lock:
            if self._finalized is not None:
                self.report_issue(
                    "trace.record_after_finalize",
                    "An artifact was submitted after trace finalization",
                )
                return None
            try:
                digest = hashlib.sha256(content).hexdigest()
                relative = f"artifacts/sha256/{digest[:2]}/{digest}"
                path = self.attempt_dir / relative
                if path.is_symlink():
                    raise TraceStorageError("Artifact path is a symbolic link")
                if path.exists():
                    if path.read_bytes() != content:
                        raise TraceStorageError("Artifact digest collision")
                    path.chmod(FILE_MODE)
                else:
                    _atomic_write(path, content)
                reference = {
                    "sha256": digest,
                    "path": relative,
                    "size_bytes": len(content),
                    "media_type": media_type,
                    "encoding": encoding,
                    "role": role,
                    "redaction": {
                        "status": "applied" if redaction.matches else "not_required",
                        "matches": redaction.matches,
                        "rules": list(redaction.rules),
                    },
                }
                self._artifacts[relative] = reference
                if count_redactions:
                    self._redactions += redaction.matches
                return reference
            except (OSError, TraceStorageError):
                self.report_issue(
                    "artifact.write_failed",
                    "A Hermes trace artifact could not be retained",
                )
                return None

    def record_event(
        self,
        *,
        event_type: str,
        event_family: str,
        phase: str,
        status: str,
        span_id: str,
        origin: JsonObject,
        timing: JsonObject,
        payload: JsonObject | None = None,
        artifacts: Sequence[JsonObject] = (),
        occurred_at: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        turn_id: str | None = None,
        parent_span_id: str | None = None,
        error: JsonObject | None = None,
        relations: Sequence[JsonObject] = (),
    ) -> str | None:
        with self._lock:
            if self._finalized is not None:
                self._dropped_events += 1
                self.report_issue(
                    "trace.record_after_finalize",
                    "An event was submitted after trace finalization",
                )
                return None
            candidate = event_id or f"event-{uuid4().hex}"
            timestamp = occurred_at or utc_now()
            error_artifact = (
                error.get("artifact") if isinstance(error, Mapping) else None
            )
            if candidate in self._event_ids:
                self._dropped_events += 1
                self.report_issue(
                    "event.duplicate_id", "A duplicate Hermes trace event was rejected"
                )
                return None
            if not _valid_event_contract_fields(
                event_type=event_type,
                event_family=event_family,
                phase=phase,
                status=status,
                event_id=candidate,
                span_id=span_id,
                occurred_at=timestamp,
                origin=origin,
                timing=timing,
                error=error,
                relations=relations,
                identifiers=(
                    session_id,
                    agent_id,
                    parent_agent_id,
                    turn_id,
                    parent_span_id,
                ),
            ) or (
                isinstance(error, Mapping)
                and "artifact" in error
                and (
                    not isinstance(error_artifact, Mapping)
                    or self._artifacts.get(str(error_artifact.get("path")))
                    != error_artifact
                )
            ):
                self._dropped_events += 1
                self.report_issue(
                    "event.invalid", "An invalid normalized Hermes event was rejected"
                )
                return None
            sanitized_origin = self._redactor.sanitize(origin)
            sanitized_timing = self._redactor.sanitize(timing)
            sanitized_payload = self._redactor.sanitize(payload or {})
            sanitized_error = self._redactor.sanitize(error) if error else None
            sanitized_relations = [
                self._redactor.sanitize(relation) for relation in relations
            ]
            identifiers = {
                key: value
                for key, value in (
                    ("session_id", session_id),
                    ("agent_id", agent_id),
                    ("parent_agent_id", parent_agent_id),
                    ("turn_id", turn_id),
                    ("parent_span_id", parent_span_id),
                )
                if value is not None
            }
            identity_check = self._redactor.sanitize({
                "event_id": candidate,
                "span_id": span_id,
                **identifiers,
            })
            if identity_check.matches:
                self._dropped_events += 1
                self.report_issue(
                    "event.sensitive_identity",
                    "A Hermes trace event identity contained credential-like material",
                )
                return None
            if any(
                self._artifacts.get(str(reference.get("path"))) != reference
                for reference in artifacts
            ):
                self._dropped_events += 1
                self.report_issue(
                    "event.unknown_artifact",
                    "A Hermes event referenced an artifact not owned by this recorder",
                )
                return None
            event: JsonObject = {
                "schema_version": SCHEMA_VERSION,
                "schema_digest": SCHEMA_DIGEST,
                "event_id": candidate,
                "sequence": self._event_sequence + 1,
                **self.identity.event_fields(),
                **identifiers,
                "span_id": span_id,
                "occurred_at": timestamp,
                "recorded_at": utc_now(),
                "event_type": event_type,
                "event_family": event_family,
                "phase": phase,
                "status": status,
                "origin": sanitized_origin.value,
                "timing": sanitized_timing.value,
                "payload": sanitized_payload.value,
                "artifacts": [dict(item) for item in artifacts],
            }
            if sanitized_error is not None:
                event["error"] = sanitized_error.value
            if relations:
                event["relations"] = [item.value for item in sanitized_relations]
            try:
                if self._events is None:
                    raise TraceStorageError("Event journal is unavailable")
                self._events.append(event)
            except TraceStorageError:
                self._dropped_events += 1
                self.report_issue(
                    "journal.append_failed",
                    "A Hermes trace event could not be durably appended",
                )
                return None
            self._event_sequence += 1
            self._event_ids.add(candidate)
            self._redactions += (
                sanitized_origin.matches
                + sanitized_timing.matches
                + sanitized_payload.matches
                + (sanitized_error.matches if sanitized_error else 0)
                + sum(item.matches for item in sanitized_relations)
            )
            return candidate

    def record_native(
        self,
        *,
        source: str,
        content: Any,
        event_ids: Sequence[str] = (),
        role: str = "native.event",
        media_type: str = "application/json",
    ) -> str | None:
        with self._lock:
            if self._finalized is not None:
                return None
            sanitized_source = self._redactor.sanitize_text(source)
            if (
                not sanitized_source.value
                or len(sanitized_source.value) > 256
                or not media_type
                or len(media_type) > 128
                or not re.fullmatch(r"[a-z][a-z0-9._-]*", role)
                or len(role) > 128
            ):
                self.report_issue(
                    "native.invalid_metadata",
                    "Hermes native evidence metadata is invalid",
                )
                return None
            identity_check = self._redactor.sanitize(list(event_ids))
            if identity_check.matches or any(
                not event_id or len(event_id) > 512 for event_id in event_ids
            ):
                self.report_issue(
                    "native.sensitive_identity",
                    "A Hermes native record identity contained credential-like material",
                )
                return None
            candidate = f"native-{uuid4().hex}"
            if isinstance(content, str):
                retained = self._redactor.sanitize_text(content)
                retained_bytes = retained.value.encode("utf-8")
            else:
                retained = self._redactor.sanitize(content)
                try:
                    retained_bytes = canonical_json(retained.value)
                except TraceStorageError:
                    self.report_issue(
                        "native.serialization_failed",
                        "Hermes native evidence could not be serialized",
                    )
                    return None
            member = {
                "native_record_id": candidate,
                "content_sha256": hashlib.sha256(retained_bytes).hexdigest(),
                "size_bytes": len(retained_bytes),
                "media_type": media_type,
                "encoding": "utf-8",
                "role": role,
                "redaction": {
                    "status": "applied" if retained.matches else "not_required",
                    "matches": retained.matches,
                    "rules": list(retained.rules),
                },
                "content_base64": base64.b64encode(retained_bytes).decode("ascii"),
            }
            record = {
                "format": NATIVE_JOURNAL_FORMAT,
                "native_record_id": candidate,
                "sequence": self._native_sequence + 1,
                "trace_id": self.identity.trace_id,
                "framework": self.identity.framework,
                "recorded_at": utc_now(),
                "source": sanitized_source.value,
                "event_ids": list(dict.fromkeys(event_ids)),
                "member": member,
            }
            try:
                if self._native is None:
                    raise TraceStorageError("Native journal is unavailable")
                self._native.append(record)
            except TraceStorageError:
                self.report_issue(
                    "native.append_failed",
                    "Hermes native evidence could not be durably indexed",
                )
                return None
            self._native_sequence += 1
            self._native_ids.add(candidate)
            self._redactions += sanitized_source.matches + retained.matches
            return candidate

    def finalize(self) -> TraceFinalization:
        with self._lock:
            if self._finalized is not None:
                return self._finalized
            try:
                if self._events is None or self._native is None:
                    raise TraceStorageError("Trace journals are unavailable")
                self._events.close()
                self._native.close()
                journal = self.attempt_dir / "journal.jsonl"
                content = journal.read_bytes()
                if content and not content.endswith(b"\n"):
                    raise TraceStorageError("Hermes trace journal has a torn record")
                _atomic_write(self.attempt_dir / "events.jsonl", content)
                _replace_with_hard_link(
                    self.attempt_dir / "events.jsonl",
                    journal,
                )
                native_index = self._pack_native_journal()
                _atomic_write(
                    self.attempt_dir / "native" / "index.jsonl",
                    b"".join(canonical_json(record) for record in native_index),
                )
                finalized_at = utc_now()
                failed = any(
                    issue.get("severity") == "error" for issue in self._issues.values()
                )
                healthy = not self._issues and self._dropped_events == 0
                capabilities = {
                    "schema_version": SCHEMA_VERSION,
                    "schema_digest": SCHEMA_DIGEST,
                    "trace_id": self.identity.trace_id,
                    "framework": self.identity.framework,
                    "generated_at": finalized_at,
                    "capabilities": [dict(item) for item in self._capabilities],
                }
                health = {
                    "schema_version": SCHEMA_VERSION,
                    "schema_digest": SCHEMA_DIGEST,
                    "trace_id": self.identity.trace_id,
                    "generated_at": finalized_at,
                    "status": (
                        "healthy" if healthy else "failed" if failed else "degraded"
                    ),
                    "finalization": "clean" if healthy else "partial",
                    "failure_policy": "continue_agent_without_retry",
                    "agent_outcome_affected": False,
                    "benchmark_retry_triggered": False,
                    "counters": {
                        "events_written": self._event_sequence,
                        "artifacts_written": len(self._artifacts),
                        "artifact_bytes_written": sum(
                            int(item["size_bytes"]) for item in self._artifacts.values()
                        ),
                        "redactions_applied": self._redactions,
                        "dropped_events": self._dropped_events,
                        "sequence_gaps": 0,
                    },
                    "issues": list(self._issues.values()),
                }
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "contract": {
                        "name": "benchmark-trace",
                        "version": CONTRACT_VERSION,
                        "schema_digest": SCHEMA_DIGEST,
                    },
                    **self.identity.event_fields(),
                    "created_at": self._created_at,
                    "finalized_at": finalized_at,
                    "complete": healthy,
                    "producer": dict(self._producer),
                    "provenance": dict(self._provenance),
                    "execution": dict(self._execution),
                    "files": {
                        "journal": "journal.jsonl",
                        "events": "events.jsonl",
                        "capabilities": "capabilities.json",
                        "health": "health.json",
                        "native_index": "native/index.jsonl",
                        "artifacts": "artifacts/sha256",
                    },
                }
                _atomic_write(
                    self.attempt_dir / "capabilities.json",
                    canonical_json(capabilities),
                )
                _atomic_write(self.attempt_dir / "health.json", canonical_json(health))
                _atomic_write(
                    self.attempt_dir / "manifest.json", canonical_json(manifest)
                )
                self._finalized = TraceFinalization(
                    trace_id=self.identity.trace_id,
                    attempt_dir=str(self.attempt_dir),
                    health=str(health["status"]),
                    complete=bool(manifest["complete"]),
                )
                return self._finalized
            except (OSError, TraceStorageError) as exc:
                raise TraceStorageError(
                    "Hermes benchmark trace could not be finalized"
                ) from exc

    def _pack_native_journal(self) -> tuple[JsonObject, ...]:
        path = self.attempt_dir / "native" / "index.jsonl"
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise TraceStorageError("Native trace journal is unreadable") from exc
        if content and not content.endswith(b"\n"):
            raise TraceStorageError("Native trace journal has a torn record")
        records: list[JsonObject] = []
        for line in content.splitlines():
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TraceStorageError("Native trace journal is malformed") from exc
            if not isinstance(record, dict):
                raise TraceStorageError("Native trace journal record is not an object")
            _require_native_journal_record(record)
            records.append(record)

        chunks: list[list[JsonObject]] = []
        current: list[JsonObject] = []
        current_bytes = 0
        for record in records:
            member = _native_member(record)
            size = len(canonical_json(member))
            if current and current_bytes + size > NATIVE_CHUNK_TARGET_BYTES:
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(record)
            current_bytes += size
        if current:
            chunks.append(current)

        packed: list[JsonObject] = []
        for chunk in chunks:
            members = [_native_member(record) for record in chunk]
            compressed = gzip.compress(
                b"".join(canonical_json(member) for member in members),
                compresslevel=6,
                mtime=0,
            )
            redactions = [_native_redaction(member) for member in members]
            matches = sum(int(value["matches"]) for value in redactions)
            rules = tuple(
                dict.fromkeys(
                    str(rule)
                    for value in redactions
                    for rule in value["rules"]
                    if isinstance(rule, str)
                )
            )
            reference = self._store_artifact(
                compressed,
                role="native.chunk",
                media_type=NATIVE_CHUNK_MEDIA_TYPE,
                encoding="binary",
                redaction=_RedactionResult(
                    value=None,
                    matches=matches,
                    rules=rules,
                ),
                count_redactions=False,
            )
            if reference is None:
                raise TraceStorageError("Native evidence chunk could not be retained")
            packed.extend(
                {
                    "schema_version": SCHEMA_VERSION,
                    "schema_digest": SCHEMA_DIGEST,
                    "native_record_id": record["native_record_id"],
                    "sequence": record["sequence"],
                    "trace_id": record["trace_id"],
                    "framework": record["framework"],
                    "recorded_at": record["recorded_at"],
                    "source": record["source"],
                    "artifact": reference,
                    "event_ids": record["event_ids"],
                }
                for record in chunk
            )
        return tuple(packed)


def _require_native_journal_record(record: JsonObject) -> None:
    member = _native_member(record)
    event_ids = record.get("event_ids")
    sequence = record.get("sequence")
    native_record_id = record.get("native_record_id")
    trace_id = record.get("trace_id")
    framework = record.get("framework")
    recorded_at = record.get("recorded_at")
    source = record.get("source")
    if (
        record.get("format") != NATIVE_JOURNAL_FORMAT
        or not isinstance(native_record_id, str)
        or not native_record_id
        or len(native_record_id) > 512
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 1
        or not isinstance(trace_id, str)
        or not trace_id
        or len(trace_id) > 512
        or not isinstance(framework, str)
        or not re.fullmatch(r"[a-z][a-z0-9._-]*", framework)
        or not isinstance(recorded_at, str)
        or not recorded_at
        or not isinstance(source, str)
        or not source
        or len(source) > 256
        or not isinstance(event_ids, list)
        or any(
            not isinstance(value, str) or not value or len(value) > 512
            for value in event_ids
        )
        or member.get("native_record_id") != native_record_id
    ):
        raise TraceStorageError("Native trace journal record is malformed")


def _native_member(record: JsonObject) -> JsonObject:
    member = record.get("member")
    if not isinstance(member, dict):
        raise TraceStorageError("Native trace journal member is malformed")
    redaction = _native_redaction(member)
    content_base64 = member.get("content_base64")
    native_record_id = member.get("native_record_id")
    content_sha256 = member.get("content_sha256")
    size = member.get("size_bytes")
    media_type = member.get("media_type")
    role = member.get("role")
    if (
        not isinstance(native_record_id, str)
        or not native_record_id
        or len(native_record_id) > 512
        or not isinstance(content_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", content_sha256)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(media_type, str)
        or not media_type
        or len(media_type) > 128
        or member.get("encoding") not in {"utf-8", "binary"}
        or not isinstance(role, str)
        or not re.fullmatch(r"[a-z][a-z0-9._-]*", role)
        or len(role) > 128
        or not isinstance(content_base64, str)
    ):
        raise TraceStorageError("Native trace journal member is malformed")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise TraceStorageError("Native trace journal content is not base64") from exc
    if len(content) != size or hashlib.sha256(content).hexdigest() != content_sha256:
        raise TraceStorageError("Native trace journal content is corrupt")
    matches = redaction["matches"]
    assert isinstance(matches, int)
    if (redaction["status"] == "applied") != (matches > 0):
        raise TraceStorageError("Native redaction metadata is inconsistent")
    return member


def _native_redaction(member: JsonObject) -> JsonObject:
    redaction = member.get("redaction")
    if not isinstance(redaction, dict):
        raise TraceStorageError("Native redaction metadata is malformed")
    matches = redaction.get("matches")
    rules = redaction.get("rules")
    if (
        redaction.get("status") not in {"applied", "not_required"}
        or not isinstance(matches, int)
        or isinstance(matches, bool)
        or matches < 0
        or not isinstance(rules, list)
        or any(not isinstance(rule, str) for rule in rules)
        or (matches > 0) != bool(rules)
    ):
        raise TraceStorageError("Native redaction metadata is malformed")
    return redaction


def encode_instance_id(instance_id: str) -> str:
    return quote(
        instance_id,
        safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-",
    )


def attempt_directory(root: Path, instance_id: str, attempt: int) -> Path:
    return root / "instances" / encode_instance_id(instance_id) / f"attempt-{attempt}"


def write_run_index(
    *,
    root: Path,
    run_id: str,
    benchmark: str,
    framework: str,
    created_at: str,
    instance_ids: Sequence[str],
    selection_strategy: str,
) -> Path:
    attempts = []
    for instance_id in instance_ids:
        instance_root = root / "instances" / encode_instance_id(instance_id)
        attempt_dirs = sorted(
            (
                path
                for path in instance_root.glob("attempt-*")
                if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
            ),
            key=lambda path: int(path.name.removeprefix("attempt-")),
        )
        for attempt_dir in attempt_dirs:
            manifest = json.loads(
                (attempt_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(manifest, dict)
                or manifest.get("run_id") != run_id
                or manifest.get("benchmark") != benchmark
                or manifest.get("framework") != framework
                or manifest.get("instance_id") != instance_id
                or manifest.get("attempt")
                != int(attempt_dir.name.removeprefix("attempt-"))
            ):
                raise TraceStorageError(
                    f"Trace attempt identity does not match its run: {attempt_dir}"
                )
            events = [
                json.loads(line)
                for line in (attempt_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            status = next(
                (
                    event["status"]
                    for event in reversed(events)
                    if event.get("event_type") == "attempt.end"
                ),
                "degraded",
            )
            attempts.append({
                "trace_id": manifest["trace_id"],
                "instance_id": instance_id,
                "attempt": manifest["attempt"],
                "path": attempt_dir.relative_to(root).as_posix(),
                "status": status,
            })
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "name": "benchmark-trace",
            "version": CONTRACT_VERSION,
            "schema_digest": SCHEMA_DIGEST,
        },
        "run_id": run_id,
        "benchmark": benchmark,
        "framework": framework,
        "created_at": created_at,
        "finalized_at": utc_now(),
        "selection": {
            "strategy": selection_strategy,
            "requested_count": len(instance_ids),
            "instance_ids": list(instance_ids),
        },
        "attempts": attempts,
    }
    path = root / "run.json"
    _atomic_write(path, canonical_json(document))
    return path
