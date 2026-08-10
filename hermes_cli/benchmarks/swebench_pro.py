"""Local-Docker SWE-bench Pro runner for Hermes Agent.

Inference only consumes the public task fields. Official evaluator-only fields
are fetched after a completed predictions artifact has been frozen and verified.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import httpx

from hermes_cli.benchmarks import swebench_verified


BENCHMARK = "swe-bench-pro"
DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_SMOKE_INSTANCE_ID = (
    "instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-"
    "v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
)
DEFAULT_MODEL = "openrouter/qwen/qwen3-coder-next"
DEFAULT_IMAGE_PREFIX = "docker.io/jefzda/sweap-images"
DEFAULT_OUTPUT_DIR = ".benchmark-runs/swe-bench-pro"
DEFAULT_TRACE_DIR = swebench_verified.DEFAULT_TRACE_DIR
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
DEFAULT_AGENT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SETUP_TIMEOUT_SECONDS = 10 * 60
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 60 * 60
DEFAULT_EVALUATION_SETUP_GRACE_SECONDS = 10 * 60
DEFAULT_INFERENCE_WORKERS = 1
DEFAULT_EVALUATION_WORKERS = 1
DEFAULT_ATTEMPTS = 1
DEFAULT_INFRASTRUCTURE_RETRIES = 0
OFFICIAL_HARNESS_REPOSITORY = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
OFFICIAL_HARNESS_REF = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
DEFAULT_DOCKERHUB_USERNAME = "jefzda"
DATASET_PAGE_SIZE = 100
DATASET_FETCH_ATTEMPTS = 3
MANIFEST_SCHEMA_VERSION = 1
PUBLIC_DATASET_FIELDS = frozenset({
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
    "requirements",
    "interface",
    "repo_language",
    "dockerhub_tag",
})
EVALUATOR_LIST_FIELDS = (
    "selected_test_files_to_run",
    "fail_to_pass",
    "pass_to_pass",
)
MAX_EVALUATOR_LIST_FIELD_CHARS = 1_000_000
MAX_EVALUATOR_LIST_ITEMS = 10_000
EVALUATOR_RUNTIME_SENTINEL = "__HERMES_SWEBENCH_PRO_RUNTIME__="

BenchmarkError = swebench_verified.BenchmarkError
ControllerTerminated = swebench_verified.ControllerTerminated


@dataclass(frozen=True)
class SweBenchProRow:
    repo: str
    instance_id: str
    base_commit: str
    problem_statement: str
    requirements: str
    interface: str
    repo_language: str
    dockerhub_tag: str

    def public_dict(self) -> dict[str, str]:
        return {
            "repo": self.repo,
            "instance_id": self.instance_id,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "requirements": self.requirements,
            "interface": self.interface,
            "repo_language": self.repo_language,
            "dockerhub_tag": self.dockerhub_tag,
        }


@dataclass(frozen=True)
class InferenceOptions:
    run_id: str
    output_dir: Path
    instance_ids: tuple[str, ...]
    max_instances: int
    offset: int
    model: str
    image_prefix: str
    docker_platform: str
    agent_timeout_seconds: int
    setup_timeout_seconds: int
    restart: bool
    dry_run: bool
    trace_dir: Path | None = None
    source_identity: dict[str, Any] | None = None
    agent_topology: str = swebench_verified.DEFAULT_AGENT_TOPOLOGY

    @property
    def include_hints(self) -> bool:
        """SWE-bench Pro's public inference schema has no hints field."""
        return False


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    predictions: Path
    manifest: Path
    summary: Path
    instances: Path
    evaluation_manifest: Path


def default_run_id() -> str:
    return swebench_verified.default_run_id().replace("swe-verified-", "swe-pro-")


def parse_swebench_pro_row(value: Any) -> SweBenchProRow:
    """Project an upstream row onto the only fields inference may consume."""
    if not isinstance(value, dict):
        raise BenchmarkError("SWE-bench Pro dataset row must be an object")

    def required(name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise BenchmarkError(f"SWE-bench Pro dataset row has invalid {name!r}")
        return item

    def string(name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str):
            raise BenchmarkError(f"SWE-bench Pro dataset row has invalid {name!r}")
        return item

    instance_id = required("instance_id")
    swebench_verified.validate_instance_id(instance_id)
    base_commit = required("base_commit")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
        raise BenchmarkError(
            f"SWE-bench Pro instance {instance_id} has an unsafe base commit"
        )
    repo = required("repo")
    repo_parts = repo.split("/")
    if (
        len(repo_parts) != 2
        or any(part in {"", ".", ".."} for part in repo_parts)
        or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in repo_parts)
    ):
        raise BenchmarkError(f"SWE-bench Pro instance {instance_id} has an unsafe repo")
    row = SweBenchProRow(
        repo=repo,
        instance_id=instance_id,
        base_commit=base_commit,
        problem_statement=required("problem_statement"),
        requirements=string("requirements"),
        interface=string("interface"),
        repo_language=string("repo_language"),
        dockerhub_tag=required("dockerhub_tag"),
    )
    official_image(row.dockerhub_tag)
    return row


def public_row_sha256(row: SweBenchProRow) -> str:
    return swebench_verified.sha256_text(
        json.dumps(
            row.public_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def sanitize_evaluator_row(
    value: Any, selected_instance: dict[str, Any]
) -> dict[str, str]:
    """Project an evaluator row and neutralize the pinned harness's host eval()."""
    row = parse_swebench_pro_row(value)
    if row.instance_id != selected_instance.get("instanceId"):
        raise BenchmarkError("SWE-bench Pro evaluator row order does not match the run")
    if public_row_sha256(row) != selected_instance.get("publicRowSha256"):
        raise BenchmarkError(
            f"SWE-bench Pro public row changed for {row.instance_id}; "
            "refusing to evaluate a different task"
        )
    before_repo_set_cmd = value.get("before_repo_set_cmd")
    if (
        not isinstance(before_repo_set_cmd, str)
        or not before_repo_set_cmd.strip()
        or len(before_repo_set_cmd) > MAX_EVALUATOR_LIST_FIELD_CHARS
    ):
        raise BenchmarkError(
            f"SWE-bench Pro evaluator row {row.instance_id} has invalid "
            "'before_repo_set_cmd'"
        )
    if "\x00" in before_repo_set_cmd:
        raise BenchmarkError(
            f"SWE-bench Pro evaluator row {row.instance_id} contains a NUL byte"
        )
    projected = {
        "repo": row.repo,
        "instance_id": row.instance_id,
        "base_commit": row.base_commit,
        "before_repo_set_cmd": before_repo_set_cmd,
    }
    for field in EVALUATOR_LIST_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, str) or len(raw) > MAX_EVALUATOR_LIST_FIELD_CHARS:
            raise BenchmarkError(
                f"SWE-bench Pro evaluator row {row.instance_id} has invalid {field!r}"
            )
        try:
            items = ast.literal_eval(raw)
        except (RecursionError, SyntaxError, ValueError) as exc:
            raise BenchmarkError(
                f"SWE-bench Pro evaluator row {row.instance_id} has unsafe {field!r}"
            ) from exc
        if (
            type(items) is not list
            or len(items) > MAX_EVALUATOR_LIST_ITEMS
            or not all(type(item) is str and "\x00" not in item for item in items)
        ):
            raise BenchmarkError(
                f"SWE-bench Pro evaluator row {row.instance_id} has invalid {field!r}"
            )
        # JSON string arrays are also valid Python literals. Re-encoding removes
        # every expression form before the pinned harness evaluates the value.
        projected[field] = json.dumps(items, ensure_ascii=True, separators=(",", ":"))
    return projected


def official_image(dockerhub_tag: str, image_prefix: str = DEFAULT_IMAGE_PREFIX) -> str:
    tag = dockerhub_tag.strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise BenchmarkError(f"Invalid SWE-bench Pro dockerhub_tag: {dockerhub_tag}")
    prefix = image_prefix.strip().rstrip(":")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./_-]*", prefix):
        raise BenchmarkError(f"Invalid SWE-bench Pro image prefix: {image_prefix}")
    return f"{prefix}:{tag}"


def format_problem_statement(row: SweBenchProRow) -> str:
    return (
        f"{row.problem_statement}\n\nRequirements:\n{row.requirements}"
        f"\n\nNew interfaces introduced:\n{row.interface}"
    )


def build_prompt(
    row: SweBenchProRow,
    _include_hints: bool = False,
    *,
    agent_topology: str = swebench_verified.DEFAULT_AGENT_TOPOLOGY,
) -> str:
    lines = [
        "Resolve this SWE-bench Pro issue using Hermes Agent.",
        "",
        "You are running inside the official SWE-bench Pro task image at /app.",
        "Edit repository files directly; do not merely describe a patch.",
        "Do not seek or use gold patches, hidden tests, or benchmark answer artifacts.",
        "Do not modify tests or benchmark metadata unless the issue explicitly requires it.",
        "",
    ]
    if agent_topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY:
        lines.extend([
            "The benchmark coordinator must use Hermes-native delegation for one fresh",
            "leaf agent at a time in this order: navigator, patcher, reviewer. It then",
            "reconciles their reports and leaves final changes in the shared worktree.",
            "",
        ])
    elif agent_topology == swebench_verified.SINGLE_AGENT_TOPOLOGY:
        lines.extend([
            "You are the sole coding agent. Do not delegate or create child agents.",
            "Personally investigate the issue, implement the smallest complete fix,",
            "run focused verification, inspect the final diff, and correct any defects",
            "you find before returning the result.",
            "",
        ])
    else:
        raise BenchmarkError(f"Unsupported agent topology: {agent_topology}")
    lines.extend([
        "## Repository",
        "Worktree: /app",
        f"Repo: {row.repo}",
        f"Base commit: {row.base_commit}",
        f"Instance id: {row.instance_id}",
        f"Repository language: {row.repo_language}",
        "",
        "## Issue",
        format_problem_statement(row),
        "",
        "## Completion requirements",
        "- Leave the final source changes in the worktree.",
        "- Run relevant lightweight verification when feasible.",
        "- Inspect the final diff before answering.",
        "- Summarize changed files, verification commands, and residual risk.",
    ])
    return "\n".join(lines)


class DatasetRowsClient:
    """Retrying client that separates public inference rows from evaluator rows."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=60)
        self._owns_client = client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DatasetRowsClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def raw_page(self, offset: int, length: int) -> list[dict[str, Any]]:
        params = {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "offset": offset,
            "length": length,
        }
        for attempt in range(1, DATASET_FETCH_ATTEMPTS + 1):
            try:
                response = self._client.get(DATASET_ROWS_URL, params=params)
            except httpx.HTTPError as exc:
                if attempt == DATASET_FETCH_ATTEMPTS:
                    raise BenchmarkError(
                        f"Could not fetch SWE-bench Pro rows: {exc}"
                    ) from exc
                self._sleep(attempt)
                continue
            if response.status_code == 200:
                revision = response.headers.get("x-revision")
                if revision != DATASET_REVISION:
                    raise BenchmarkError(
                        "SWE-bench Pro dataset revision changed or was not reported; "
                        f"expected {DATASET_REVISION}, found {revision or 'missing'}"
                    )
                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        "SWE-bench Pro dataset response is not valid JSON"
                    ) from exc
                wrapped = payload.get("rows") if isinstance(payload, dict) else None
                if not isinstance(wrapped, list):
                    raise BenchmarkError(
                        "SWE-bench Pro dataset response has no rows array"
                    )
                rows = []
                for item in wrapped:
                    row = item.get("row") if isinstance(item, dict) else None
                    if not isinstance(row, dict):
                        raise BenchmarkError(
                            "SWE-bench Pro dataset response contains a malformed row"
                        )
                    rows.append(row)
                return rows
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == DATASET_FETCH_ATTEMPTS:
                raise BenchmarkError(
                    f"Could not fetch SWE-bench Pro rows ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            self._sleep(attempt)
        raise AssertionError("unreachable")

    def page(self, offset: int, length: int) -> list[SweBenchProRow]:
        return [parse_swebench_pro_row(row) for row in self.raw_page(offset, length)]

    def select(
        self,
        instance_ids: Sequence[str],
        *,
        offset: int,
        max_instances: int,
    ) -> list[SweBenchProRow]:
        if not instance_ids:
            selected = []
            page_offset = offset
            while len(selected) < max_instances:
                length = min(DATASET_PAGE_SIZE, max_instances - len(selected))
                rows = self.page(page_offset, length)
                selected.extend(rows)
                if len(rows) < length:
                    break
                page_offset += length
            return selected[:max_instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise BenchmarkError("Duplicate --instance-id values are not allowed")
        for instance_id in instance_ids:
            swebench_verified.validate_instance_id(instance_id)
        wanted = set(instance_ids)
        found: dict[str, SweBenchProRow] = {}
        page_offset = 0
        while len(found) < len(wanted):
            rows = self.page(page_offset, DATASET_PAGE_SIZE)
            if not rows:
                break
            found.update({
                row.instance_id: row for row in rows if row.instance_id in wanted
            })
            page_offset += DATASET_PAGE_SIZE
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in found
        ]
        if missing:
            raise BenchmarkError(
                f"Could not find SWE-bench Pro instance(s): {', '.join(missing)}"
            )
        return [found[instance_id] for instance_id in instance_ids]

    def evaluator_rows(self, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
        wanted = set(instance_ids)
        found: dict[str, dict[str, Any]] = {}
        page_offset = 0
        while len(found) < len(wanted):
            rows = self.raw_page(page_offset, DATASET_PAGE_SIZE)
            if not rows:
                break
            for row in rows:
                instance_id = row.get("instance_id")
                if isinstance(instance_id, str) and instance_id in wanted:
                    found[instance_id] = row
            page_offset += DATASET_PAGE_SIZE
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in found
        ]
        if missing:
            raise BenchmarkError(
                f"Could not fetch evaluator row(s): {', '.join(missing)}"
            )
        return [found[instance_id] for instance_id in instance_ids]


def build_paths(options: InferenceOptions) -> RunPaths:
    run_dir = options.output_dir.resolve() / "runs" / options.run_id
    return RunPaths(
        run_dir=run_dir,
        predictions=run_dir / "predictions.json",
        manifest=run_dir / "prediction-manifest.json",
        summary=run_dir / "summary.json",
        instances=run_dir / "instances.jsonl",
        evaluation_manifest=run_dir / "evaluation-manifest.json",
    )


def encode_predictions(predictions: Iterable[dict[str, str]]) -> str:
    return json.dumps(list(predictions), indent=2, sort_keys=True) + "\n"


def _prediction(
    row: SweBenchProRow, run_id: str, result: dict[str, Any]
) -> dict[str, str]:
    patch = result.get("modelPatch")
    swebench_verified.validate_run_id(run_id)
    return {
        "instance_id": row.instance_id,
        "patch": patch if isinstance(patch, str) else "",
        # Scale uses this value directly in output filenames.
        "prefix": run_id,
    }


def _instance_summary(
    row: SweBenchProRow,
    image: str,
    result: dict[str, Any],
    prediction: dict[str, str],
) -> dict[str, Any]:
    return {
        "instanceId": row.instance_id,
        "repo": row.repo,
        "baseCommit": row.base_commit,
        "image": image,
        "attemptsUsed": DEFAULT_ATTEMPTS,
        "infrastructureRetriesUsed": DEFAULT_INFRASTRUCTURE_RETRIES,
        "semanticRetriesUsed": 0,
        "timedOut": bool(result.get("timedOut")),
        "workflowComplete": bool(result.get("workflowComplete")),
        "agentTopology": result.get("agentTopology"),
        "primaryAgentRole": result.get("primaryAgentRole"),
        "agentSequence": result.get("agentSequence"),
        "delegationEnabled": result.get("delegationEnabled"),
        "delegationMode": result.get("delegationMode"),
        "generationSucceeded": (
            not result.get("error")
            and not result.get("timedOut")
            and bool(result.get("workflowComplete"))
            and bool(prediction["patch"].strip())
        ),
        "changedPaths": result.get("changedPaths") or [],
        "patchCaptureError": result.get("patchCaptureError"),
        "sandboxOwnershipError": result.get("sandboxOwnershipError"),
        "sandboxCleanupError": result.get("sandboxCleanupError"),
        "containerCleanupError": result.get("containerCleanupError"),
        "error": result.get("error"),
        "workerReturnCode": result.get("workerReturnCode"),
        "phases": result.get("phases") or [],
        "startedAt": result.get("controllerStartedAt"),
        "completedAt": result.get("controllerCompletedAt"),
    }


def _selected_instances(
    options: InferenceOptions, rows: Sequence[SweBenchProRow]
) -> list[dict[str, str]]:
    return [
        {
            "instanceId": row.instance_id,
            "repo": row.repo,
            "baseCommit": row.base_commit,
            "image": official_image(row.dockerhub_tag, options.image_prefix),
            "publicRowSha256": public_row_sha256(row),
        }
        for row in rows
    ]


def _agent_sequence(topology: str) -> list[str]:
    if topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY:
        return list(swebench_verified.DEFAULT_AGENT_SEQUENCE)
    if topology == swebench_verified.SINGLE_AGENT_TOPOLOGY:
        return ["agent"]
    raise BenchmarkError(f"Unsupported agent topology: {topology}")


def _agent_budgets(topology: str) -> dict[str, int]:
    if topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY:
        return {
            "coordinator": swebench_verified.DEFAULT_COORDINATOR_BUDGET,
            "nativeSubagent": swebench_verified.DEFAULT_NATIVE_SUBAGENT_BUDGET,
            "nativeSubagentCount": swebench_verified.DEFAULT_NATIVE_SUBAGENT_COUNT,
        }
    if topology == swebench_verified.SINGLE_AGENT_TOPOLOGY:
        return {"singleAgent": swebench_verified.DEFAULT_COORDINATOR_BUDGET}
    raise BenchmarkError(f"Unsupported agent topology: {topology}")


def _delegation_mode(topology: str) -> str:
    if topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY:
        return swebench_verified.DEFAULT_DELEGATION_MODE
    if topology == swebench_verified.SINGLE_AGENT_TOPOLOGY:
        return "disabled"
    raise BenchmarkError(f"Unsupported agent topology: {topology}")


def _phase_budget_reference(topology: str) -> dict[str, int] | None:
    if topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY:
        return swebench_verified.DEFAULT_PHASE_BUDGETS
    if topology == swebench_verified.SINGLE_AGENT_TOPOLOGY:
        return None
    raise BenchmarkError(f"Unsupported agent topology: {topology}")


def _manifest(
    options: InferenceOptions,
    rows: Sequence[SweBenchProRow],
    predictions: Sequence[dict[str, str]],
    *,
    complete: bool,
) -> dict[str, Any]:
    content = encode_predictions(predictions)
    instances_content = swebench_verified.encode_jsonl(
        row.public_dict() for row in rows
    )
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "dataset": DATASET_NAME,
        "datasetConfig": DATASET_CONFIG,
        "datasetSplit": DATASET_SPLIT,
        "datasetRevision": DATASET_REVISION,
        "runId": options.run_id,
        "framework": "hermes-agent",
        "hermesVersion": swebench_verified.hermes_version(),
        "sourceIdentity": swebench_verified._source_identity_for_options(options),
        "model": swebench_verified.canonical_model(options.model),
        "temperature": 0.1,
        "inferenceRuntime": "official-swebench-pro-instance-image",
        "imagePrefix": options.image_prefix,
        "containerWorktree": "/app",
        "dockerPlatform": options.docker_platform,
        "inferenceWorkers": DEFAULT_INFERENCE_WORKERS,
        "attemptsPerInstance": DEFAULT_ATTEMPTS,
        "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
        "apiMaxRetries": swebench_verified.DEFAULT_API_MAX_RETRIES,
        "codingContext": swebench_verified.DEFAULT_CODING_CONTEXT,
        "agentTimeoutSeconds": options.agent_timeout_seconds,
        "setupTimeoutSeconds": options.setup_timeout_seconds,
        "agentTopology": options.agent_topology,
        "primaryAgentRole": (
            "coordinator"
            if options.agent_topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY
            else "agent"
        ),
        "delegationEnabled": (
            options.agent_topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY
        ),
        "agentSequence": _agent_sequence(options.agent_topology),
        "agentBudgets": _agent_budgets(options.agent_topology),
        "delegationMode": _delegation_mode(options.agent_topology),
        "peerPhaseBudgetReference": _phase_budget_reference(options.agent_topology),
        "selectedInstances": _selected_instances(options, rows),
        "instancesSha256": swebench_verified.sha256_text(instances_content),
        "completedInstanceIds": [item["instance_id"] for item in predictions],
        "complete": complete,
        "predictionCount": len(predictions),
        "nonEmptyPatchCount": sum(bool(item["patch"].strip()) for item in predictions),
        "predictionsSha256": swebench_verified.sha256_text(content),
        "generatedAt": swebench_verified.utc_now(),
    }


def _write_progress(
    options: InferenceOptions,
    paths: RunPaths,
    rows: Sequence[SweBenchProRow],
    summaries: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, str]],
    *,
    complete: bool,
) -> None:
    content = encode_predictions(predictions)
    manifest = _manifest(options, rows, predictions, complete=complete)
    swebench_verified.atomic_write_text(paths.predictions, content)
    swebench_verified.atomic_write_json(paths.manifest, manifest)
    swebench_verified.atomic_write_json(
        paths.summary,
        {
            "runId": options.run_id,
            "benchmark": BENCHMARK,
            "dataset": DATASET_NAME,
            "model": swebench_verified.canonical_model(options.model),
            "agentTopology": options.agent_topology,
            "delegationMode": _delegation_mode(options.agent_topology),
            "selectedCount": len(rows),
            "completedCount": len(predictions),
            "generationSucceededCount": sum(
                summary.get("generationSucceeded") is True for summary in summaries
            ),
            "predictionCount": len(predictions),
            "nonEmptyPatchCount": sum(
                bool(prediction["patch"].strip()) for prediction in predictions
            ),
            "complete": complete,
            "predictionsPath": str(paths.predictions),
            "predictionManifestPath": str(paths.manifest),
            "predictionsSha256": manifest["predictionsSha256"],
            "summaries": list(summaries),
        },
    )


def _parse_predictions(content: str) -> list[dict[str, str]]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("SWE-bench Pro predictions are not valid JSON") from exc
    if not isinstance(value, list):
        raise BenchmarkError("SWE-bench Pro predictions must be a JSON array")
    predictions = []
    for index, item in enumerate(value, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"instance_id", "patch", "prefix"}
            or not all(isinstance(item.get(key), str) for key in item)
        ):
            raise BenchmarkError(f"Invalid SWE-bench Pro prediction row {index}")
        swebench_verified.validate_instance_id(item["instance_id"])
        swebench_verified.validate_run_id(item["prefix"])
        predictions.append(item)
    return predictions


def _expected_resume_contract(
    options: InferenceOptions, rows: Sequence[SweBenchProRow]
) -> dict[str, Any]:
    instances_content = swebench_verified.encode_jsonl(
        row.public_dict() for row in rows
    )
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "dataset": DATASET_NAME,
        "datasetConfig": DATASET_CONFIG,
        "datasetSplit": DATASET_SPLIT,
        "datasetRevision": DATASET_REVISION,
        "runId": options.run_id,
        "framework": "hermes-agent",
        "hermesVersion": swebench_verified.hermes_version(),
        "sourceIdentity": swebench_verified._source_identity_for_options(options),
        "model": swebench_verified.canonical_model(options.model),
        "temperature": 0.1,
        "inferenceRuntime": "official-swebench-pro-instance-image",
        "imagePrefix": options.image_prefix,
        "containerWorktree": "/app",
        "dockerPlatform": options.docker_platform,
        "inferenceWorkers": DEFAULT_INFERENCE_WORKERS,
        "attemptsPerInstance": DEFAULT_ATTEMPTS,
        "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
        "apiMaxRetries": swebench_verified.DEFAULT_API_MAX_RETRIES,
        "codingContext": swebench_verified.DEFAULT_CODING_CONTEXT,
        "agentTimeoutSeconds": options.agent_timeout_seconds,
        "setupTimeoutSeconds": options.setup_timeout_seconds,
        "agentTopology": options.agent_topology,
        "primaryAgentRole": (
            "coordinator"
            if options.agent_topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY
            else "agent"
        ),
        "delegationEnabled": (
            options.agent_topology == swebench_verified.DEFAULT_AGENT_TOPOLOGY
        ),
        "agentSequence": _agent_sequence(options.agent_topology),
        "agentBudgets": _agent_budgets(options.agent_topology),
        "delegationMode": _delegation_mode(options.agent_topology),
        "peerPhaseBudgetReference": _phase_budget_reference(options.agent_topology),
        "selectedInstances": _selected_instances(options, rows),
        "instancesSha256": swebench_verified.sha256_text(instances_content),
    }


def _load_resume(
    options: InferenceOptions,
    paths: RunPaths,
    rows: Sequence[SweBenchProRow],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not paths.manifest.exists():
        if paths.predictions.exists() or paths.summary.exists():
            raise BenchmarkError(
                f"Run {options.run_id} has incomplete metadata; use --restart"
            )
        return [], []
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        content = paths.predictions.read_text(encoding="utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise BenchmarkError("Could not read the incomplete Pro run") from exc
    if not isinstance(manifest, dict):
        raise BenchmarkError("Incomplete Pro run manifest must be an object")
    try:
        instances_content = paths.instances.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkError(
            "Could not read the incomplete Pro dataset artifact"
        ) from exc
    if swebench_verified.sha256_text(instances_content) != manifest.get(
        "instancesSha256"
    ):
        raise BenchmarkError(
            "Incomplete Pro dataset artifact does not match its manifest"
        )
    mismatches = [
        key
        for key, expected in _expected_resume_contract(options, rows).items()
        if manifest.get(
            key,
            (
                swebench_verified.DEFAULT_AGENT_TOPOLOGY
                if key == "agentTopology"
                else None
            ),
        )
        != expected
    ]
    if mismatches:
        raise BenchmarkError(
            f"Existing run differs in {', '.join(mismatches)}; use --restart"
        )
    if manifest.get("complete"):
        raise BenchmarkError(
            f"Run {options.run_id} is already complete; use a new run id or --restart"
        )
    if swebench_verified.sha256_text(content) != manifest.get("predictionsSha256"):
        raise BenchmarkError("Incomplete predictions do not match their manifest")
    predictions = _parse_predictions(content)
    completed_ids = [prediction["instance_id"] for prediction in predictions]
    if len(predictions) != manifest.get("predictionCount"):
        raise BenchmarkError("Incomplete prediction count does not match its manifest")
    if sum(bool(item["patch"].strip()) for item in predictions) != manifest.get(
        "nonEmptyPatchCount"
    ):
        raise BenchmarkError(
            "Incomplete non-empty prediction count does not match its manifest"
        )
    if any(prediction["prefix"] != options.run_id for prediction in predictions):
        raise BenchmarkError("Incomplete prediction prefix does not match its run id")
    if completed_ids != [row.instance_id for row in rows[: len(predictions)]]:
        raise BenchmarkError(
            "Incomplete predictions are not an ordered instance prefix"
        )
    if completed_ids != manifest.get("completedInstanceIds"):
        raise BenchmarkError("Incomplete prediction IDs do not match their manifest")
    summaries = []
    for row, prediction in zip(rows[: len(predictions)], predictions, strict=True):
        instance_dir = paths.run_dir / "instances" / row.instance_id
        try:
            checkpoint_prediction = json.loads(
                (instance_dir / "prediction.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (instance_dir / "run.json").read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            raise BenchmarkError(
                f"Could not read resume checkpoint for {row.instance_id}"
            ) from exc
        if checkpoint_prediction != prediction or not isinstance(summary, dict):
            raise BenchmarkError(f"Resume checkpoint differs for {row.instance_id}")
        summaries.append(summary)
    return summaries, predictions


def run_inference(
    options: InferenceOptions,
    *,
    client: DatasetRowsClient | None = None,
) -> RunPaths:
    swebench_verified.validate_run_id(options.run_id)
    swebench_verified.validate_docker_platform(options.docker_platform)
    swebench_verified.canonical_model(options.model)
    _agent_sequence(options.agent_topology)
    if options.max_instances <= 0 or options.offset < 0:
        raise BenchmarkError("Instance selection values are invalid")
    owned_client = client is None
    dataset = client or DatasetRowsClient()
    try:
        rows = dataset.select(
            options.instance_ids,
            offset=options.offset,
            max_instances=options.max_instances,
        )
    finally:
        if owned_client:
            dataset.close()
    if not rows:
        raise BenchmarkError("SWE-bench Pro selection returned no instances")
    paths = build_paths(options)
    if options.dry_run:
        print(
            json.dumps(
                {
                    "mode": "inference",
                    "runId": options.run_id,
                    "benchmark": BENCHMARK,
                    "dataset": DATASET_NAME,
                    "datasetRevision": DATASET_REVISION,
                    "instanceIds": [row.instance_id for row in rows],
                    "model": swebench_verified.canonical_model(options.model),
                    "temperature": 0.1,
                    "inferenceWorkers": DEFAULT_INFERENCE_WORKERS,
                    "attemptsPerInstance": DEFAULT_ATTEMPTS,
                    "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
                    "apiMaxRetries": swebench_verified.DEFAULT_API_MAX_RETRIES,
                    "agentTimeoutSeconds": options.agent_timeout_seconds,
                    "setupTimeoutSeconds": options.setup_timeout_seconds,
                    "agentTopology": options.agent_topology,
                    "primaryAgentRole": (
                        "coordinator"
                        if options.agent_topology
                        == swebench_verified.DEFAULT_AGENT_TOPOLOGY
                        else "agent"
                    ),
                    "delegationEnabled": (
                        options.agent_topology
                        == swebench_verified.DEFAULT_AGENT_TOPOLOGY
                    ),
                    "sequence": _agent_sequence(options.agent_topology),
                    "delegationMode": _delegation_mode(options.agent_topology),
                    "agentBudget": swebench_verified.DEFAULT_COORDINATOR_BUDGET,
                    **(
                        {
                            "coordinatorBudget": (
                                swebench_verified.DEFAULT_COORDINATOR_BUDGET
                            ),
                            "nativeSubagentBudget": (
                                swebench_verified.DEFAULT_NATIVE_SUBAGENT_BUDGET
                            ),
                            "nativeSubagentCount": (
                                swebench_verified.DEFAULT_NATIVE_SUBAGENT_COUNT
                            ),
                            "nativeSubagentTotalBudget": (
                                swebench_verified.DEFAULT_NATIVE_SUBAGENT_BUDGET
                                * swebench_verified.DEFAULT_NATIVE_SUBAGENT_COUNT
                            ),
                            "peerPhaseBudgetReference": (
                                swebench_verified.DEFAULT_PHASE_BUDGETS
                            ),
                        }
                        if options.agent_topology
                        == swebench_verified.DEFAULT_AGENT_TOPOLOGY
                        else {}
                    ),
                    "images": [
                        official_image(row.dockerhub_tag, options.image_prefix)
                        for row in rows
                    ],
                    "output": str(paths.run_dir),
                    "traceBase": (
                        str(options.trace_dir.resolve())
                        if options.trace_dir is not None
                        else None
                    ),
                },
                indent=2,
            )
        )
        return paths
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise BenchmarkError("OPENROUTER_API_KEY is required for inference")
    docker_metadata = swebench_verified.require_docker()
    if options.restart and paths.run_dir.exists():
        shutil.rmtree(paths.run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    frozen_options = (
        options
        if options.source_identity is not None
        else replace(
            options, source_identity=swebench_verified.hermes_source_identity()
        )
    )
    summaries, predictions = _load_resume(frozen_options, paths, rows)
    if options.trace_dir is not None and predictions:
        raise BenchmarkError(
            "Tracing cannot resume an inference run with completed instances; "
            "use a fresh run id"
        )
    swebench_verified.atomic_write_text(
        paths.instances,
        swebench_verified.encode_jsonl(row.public_dict() for row in rows),
    )
    swebench_verified.require_unchanged_source(frozen_options)
    trace_run = None
    if options.trace_dir is not None:
        source_identity = swebench_verified._source_identity_for_options(frozen_options)
        if (
            source_identity["dirty"] is not False
            or not isinstance(source_identity["commit"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", source_identity["commit"])
        ):
            raise BenchmarkError(
                "Tracing requires a clean Hermes checkout at an exact Git revision"
            )
        from hermes_cli.benchmarks.tracing import create_trace_run

        trace_run = create_trace_run(
            options.trace_dir,
            benchmark=BENCHMARK,
            framework="hermes",
        )
        swebench_verified.require_unchanged_source(frozen_options)
        print(f"[trace] output: {trace_run.root}")
    _write_progress(frozen_options, paths, rows, summaries, predictions, complete=False)
    for row in rows[len(predictions) :]:
        instance_dir = paths.run_dir / "instances" / row.instance_id
        image = official_image(row.dockerhub_tag, options.image_prefix)
        image_metadata: dict[str, Any] = {"image": image}
        try:
            image_metadata = swebench_verified.ensure_image(
                image, options.docker_platform, options.setup_timeout_seconds
            )
            image_metadata["docker"] = docker_metadata
            swebench_verified.require_unchanged_source(frozen_options)
            result = swebench_verified._run_worker(
                frozen_options,
                row,
                instance_dir,
                image,
                image_metadata,
                benchmark=BENCHMARK,
                worker_module="hermes_cli.benchmarks.swebench_pro_worker",
                prompt_builder=lambda selected_row, include_hints: build_prompt(
                    selected_row,
                    include_hints,
                    agent_topology=options.agent_topology,
                ),
                worktree="/app",
                evaluation_timeout_seconds=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
                trace_run=trace_run,
                agent_topology=options.agent_topology,
            )
        except Exception as exc:
            result = {
                "modelPatch": "",
                "changedPaths": [],
                "workflowComplete": False,
                "agentTopology": options.agent_topology,
                "delegationMode": _delegation_mode(options.agent_topology),
                "timedOut": False,
                "error": f"{type(exc).__name__}: {exc}",
                "controllerStartedAt": swebench_verified.utc_now(),
                "controllerCompletedAt": swebench_verified.utc_now(),
            }
        swebench_verified.require_unchanged_source(frozen_options)
        prediction = _prediction(row, options.run_id, result)
        summary = _instance_summary(row, image, result, prediction)
        instance_dir.mkdir(parents=True, exist_ok=True)
        swebench_verified.atomic_write_text(
            instance_dir / "model.patch", prediction["patch"]
        )
        swebench_verified.atomic_write_json(
            instance_dir / "prediction.json", prediction
        )
        swebench_verified.atomic_write_json(instance_dir / "run.json", summary)
        predictions.append(prediction)
        summaries.append(summary)
        _write_progress(
            frozen_options,
            paths,
            rows,
            summaries,
            predictions,
            complete=len(predictions) == len(rows),
        )
        status = "ok" if summary["generationSucceeded"] else "failed"
        print(f"[{len(predictions)}/{len(rows)}] {row.instance_id}: {status}")
    swebench_verified.require_unchanged_source(frozen_options)
    if trace_run is not None:
        from hermes_cli.benchmarks.tracing import (
            DirectTraceHarness,
            TraceSelection,
            finalize_trace_run,
        )

        try:
            finalize_trace_run(
                trace_run,
                DirectTraceHarness(
                    TraceSelection(
                        instance_ids=tuple(row.instance_id for row in rows),
                        strategy=(
                            "explicit_ids" if options.instance_ids else "ordered_window"
                        ),
                    )
                ),
            )
        except Exception as exc:
            print(
                "[trace] Hermes trace run index could not be finalized; "
                f"benchmark outputs remain valid: {type(exc).__name__}",
                file=sys.stderr,
            )
    return paths


def verify_prediction_artifact(
    predictions_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    try:
        content = predictions_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (
        UnicodeDecodeError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise BenchmarkError(f"Could not read prediction artifact: {exc}") from exc
    if not isinstance(manifest, dict) or any((
        manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION,
        manifest.get("benchmark") != BENCHMARK,
        manifest.get("dataset") != DATASET_NAME,
        manifest.get("datasetConfig") != DATASET_CONFIG,
        manifest.get("datasetSplit") != DATASET_SPLIT,
        manifest.get("datasetRevision") != DATASET_REVISION,
        manifest.get("framework") != "hermes-agent",
        not manifest.get("complete"),
    )):
        raise BenchmarkError("Prediction manifest is not a complete SWE-bench Pro run")
    digest = swebench_verified.sha256_text(content)
    if digest != manifest.get("predictionsSha256"):
        raise BenchmarkError("Predictions changed after inference; refusing evaluation")
    predictions = _parse_predictions(content)
    prediction_ids = [prediction["instance_id"] for prediction in predictions]
    selected = manifest.get("selectedInstances")
    selected_keys = {
        "instanceId",
        "repo",
        "baseCommit",
        "image",
        "publicRowSha256",
    }
    if not isinstance(selected, list) or not all(
        isinstance(item, dict)
        and set(item) == selected_keys
        and all(isinstance(item.get(key), str) for key in selected_keys)
        and re.fullmatch(r"[0-9a-f]{64}", item["publicRowSha256"]) is not None
        for item in selected
    ):
        raise BenchmarkError("Prediction manifest has invalid selected instances")
    if prediction_ids != [item.get("instanceId") for item in selected]:
        raise BenchmarkError("Prediction IDs or ordering do not match the manifest")
    if prediction_ids != manifest.get("completedInstanceIds"):
        raise BenchmarkError("Completed prediction IDs do not match the manifest")
    if any(prediction["prefix"] != manifest.get("runId") for prediction in predictions):
        raise BenchmarkError("Prediction prefixes do not match the manifest run id")
    if len(predictions) != manifest.get("predictionCount"):
        raise BenchmarkError("Prediction count does not match the manifest")
    if sum(bool(item["patch"].strip()) for item in predictions) != manifest.get(
        "nonEmptyPatchCount"
    ):
        raise BenchmarkError("Non-empty prediction count does not match the manifest")
    return manifest, predictions, digest


def validate_pinned_harness(harness_dir: Path) -> Path:
    required_files = (
        harness_dir / "swe_bench_pro_eval.py",
        harness_dir / "helper_code" / "image_uri.py",
    )
    required_directories = (
        harness_dir / "run_scripts",
        harness_dir / "dockerfiles" / "base_dockerfile",
        harness_dir / "dockerfiles" / "instance_dockerfile",
    )
    if not all(path.is_file() for path in required_files) or not all(
        path.is_dir() for path in required_directories
    ):
        raise BenchmarkError("SWE-bench Pro harness checkout is incomplete")
    env = swebench_verified._evaluation_environment()
    head = swebench_verified.run_command(
        ["git", "-C", str(harness_dir), "rev-parse", "HEAD"],
        timeout=30,
        env=env,
    )
    if head.returncode != 0 or head.stdout.strip() != OFFICIAL_HARNESS_REF:
        raise BenchmarkError(
            f"SWE-bench Pro harness must be pinned to {OFFICIAL_HARNESS_REF}; "
            f"found {head.stdout.strip() or 'unknown'}"
        )
    status = swebench_verified.run_command(
        [
            "git",
            "-C",
            str(harness_dir),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        timeout=30,
        env=env,
    )
    if status.returncode != 0:
        raise BenchmarkError("Could not verify SWE-bench Pro harness worktree")
    if status.stdout.strip():
        raise BenchmarkError(
            "SWE-bench Pro harness has local or ignored files; "
            "use a clean pinned checkout"
        )
    return harness_dir


def validate_harness_instances(harness_dir: Path, instance_ids: Sequence[str]) -> None:
    for instance_id in instance_ids:
        required = (
            harness_dir / "run_scripts" / instance_id / "run_script.sh",
            harness_dir / "run_scripts" / instance_id / "parser.py",
            harness_dir
            / "dockerfiles"
            / "base_dockerfile"
            / instance_id
            / "Dockerfile",
            harness_dir
            / "dockerfiles"
            / "instance_dockerfile"
            / instance_id
            / "Dockerfile",
        )
        if not all(path.is_file() for path in required):
            raise BenchmarkError(
                f"Pinned SWE-bench Pro harness is missing files for {instance_id}"
            )


def ensure_pinned_harness(harness_dir: str | None = None) -> Path:
    if harness_dir:
        return validate_pinned_harness(Path(harness_dir).expanduser().resolve())
    destination = (
        Path.home()
        / ".cache"
        / "hermes-agent"
        / "benchmarks"
        / "swe-bench-pro"
        / OFFICIAL_HARNESS_REF
    )
    if destination.is_symlink():
        raise BenchmarkError("Refusing a symlinked SWE-bench Pro harness cache")
    if destination.exists():
        try:
            return validate_pinned_harness(destination)
        except BenchmarkError:
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".harness-", dir=destination.parent))
    checkout = temporary / "checkout"
    try:
        clone = swebench_verified.run_command(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                OFFICIAL_HARNESS_REPOSITORY,
                str(checkout),
            ],
            timeout=DEFAULT_SETUP_TIMEOUT_SECONDS,
            env=swebench_verified._evaluation_environment(),
        )
        if clone.returncode != 0:
            raise BenchmarkError(
                f"Could not clone SWE-bench Pro harness: {clone.stderr.strip()}"
            )
        pin = swebench_verified.run_command(
            [
                "git",
                "-C",
                str(checkout),
                "checkout",
                "--quiet",
                "--detach",
                OFFICIAL_HARNESS_REF,
            ],
            timeout=120,
            env=swebench_verified._evaluation_environment(),
        )
        if pin.returncode != 0:
            raise BenchmarkError(
                f"Could not pin SWE-bench Pro harness: {pin.stderr.strip()}"
            )
        validate_pinned_harness(checkout)
        os.replace(checkout, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return validate_pinned_harness(destination)


def build_evaluation_command(
    python: str,
    harness_dir: Path,
    raw_sample_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    dockerhub_username: str,
    docker_platform: str,
    block_network: bool,
    redo: bool,
) -> list[str]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,254}", dockerhub_username):
        raise BenchmarkError(
            f"Invalid Docker Hub username for SWE-bench Pro: {dockerhub_username}"
        )
    command = [
        python,
        str(harness_dir / "swe_bench_pro_eval.py"),
        f"--raw_sample_path={raw_sample_path}",
        f"--patch_path={predictions_path}",
        f"--output_dir={output_dir}",
        f"--scripts_dir={harness_dir / 'run_scripts'}",
        f"--num_workers={DEFAULT_EVALUATION_WORKERS}",
        f"--dockerhub_username={dockerhub_username}",
        "--use_local_docker",
        f"--docker_platform={docker_platform}",
    ]
    if block_network:
        command.append("--block_network")
    if redo:
        command.append("--redo")
    return command


def _require_evaluator_python(python: str, env: dict[str, str]) -> None:
    result = swebench_verified.run_command(
        [python, "-c", "import docker, pandas, tqdm"], timeout=30, env=env
    )
    if result.returncode != 0:
        raise BenchmarkError(
            "Evaluator Python must provide docker, pandas, and tqdm: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def evaluator_runtime_metadata(python: str, env: dict[str, str]) -> dict[str, Any]:
    script = (
        "import importlib.metadata as m,json,platform,sys;"
        "versions={d.metadata['Name'].lower():d.version for d in m.distributions() "
        "if d.metadata['Name']};"
        "value={'pythonImplementation':sys.implementation.name,"
        "'pythonVersion':platform.python_version(),"
        "'packages':{name:versions.get(name) for name in ('docker','pandas','tqdm')}};"
        f"print('{EVALUATOR_RUNTIME_SENTINEL}'+json.dumps(value,sort_keys=True))"
    )
    result = swebench_verified.run_command([python, "-c", script], timeout=30, env=env)
    values = [
        line.removeprefix(EVALUATOR_RUNTIME_SENTINEL)
        for line in result.stdout.splitlines()
        if line.startswith(EVALUATOR_RUNTIME_SENTINEL)
    ]
    if result.returncode != 0 or not values:
        raise BenchmarkError(
            "Could not inspect the SWE-bench Pro evaluator Python: "
            + (result.stderr.strip() or result.stdout.strip() or "no metadata returned")
        )
    try:
        metadata = json.loads(values[-1])
    except json.JSONDecodeError as exc:
        raise BenchmarkError("Evaluator Python returned invalid metadata") from exc
    packages = metadata.get("packages") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("pythonImplementation"), str)
        or not isinstance(metadata.get("pythonVersion"), str)
        or not isinstance(packages, dict)
        or set(packages) != {"docker", "pandas", "tqdm"}
        or not all(
            value is None or isinstance(value, str) for value in packages.values()
        )
    ):
        raise BenchmarkError("Evaluator Python returned invalid metadata")
    return metadata


def cleanup_evaluation_containers(
    output_dir: Path,
    instance_ids: Sequence[str],
    *,
    env: dict[str, str],
) -> list[str]:
    """Remove only evaluator containers mounted to this run's workspaces."""
    expected_sources = {
        (output_dir.resolve() / instance_id / "workspace").resolve()
        for instance_id in instance_ids
    }
    try:
        listing = swebench_verified.run_command(
            ["docker", "ps", "-aq"], timeout=30, env=env
        )
    except BenchmarkError as exc:
        return [str(exc)]
    if listing.returncode != 0:
        return [
            listing.stderr.strip()
            or listing.stdout.strip()
            or "could not list evaluator containers"
        ]
    container_ids = [
        item
        for item in listing.stdout.splitlines()
        if re.fullmatch(r"[0-9a-f]{12,64}", item)
    ]
    if not container_ids:
        return []
    errors = []
    owned_ids = []
    for listed_id in container_ids:
        try:
            inspected = swebench_verified.run_command(
                ["docker", "inspect", listed_id], timeout=30, env=env
            )
        except BenchmarkError as exc:
            errors.append(str(exc))
            continue
        if inspected.returncode != 0:
            detail = inspected.stderr.strip() or inspected.stdout.strip()
            if "no such" not in detail.lower():
                errors.append(detail or f"could not inspect container {listed_id}")
            continue
        try:
            metadata = json.loads(inspected.stdout)
        except json.JSONDecodeError:
            errors.append(f"Docker returned invalid metadata for {listed_id}")
            continue
        if not isinstance(metadata, list) or len(metadata) != 1:
            errors.append(f"Docker returned invalid metadata for {listed_id}")
            continue
        item = metadata[0]
        if not isinstance(item, dict):
            errors.append(f"Docker returned invalid metadata for {listed_id}")
            continue
        container_id = item.get("Id")
        mounts = item.get("Mounts")
        if not isinstance(container_id, str) or not re.fullmatch(
            r"[0-9a-f]{12,64}", container_id
        ):
            continue
        if not isinstance(mounts, list):
            continue
        if any(
            isinstance(mount, dict)
            and mount.get("Type") == "bind"
            and mount.get("Destination") == "/workspace"
            and isinstance(mount.get("Source"), str)
            and Path(mount["Source"]).resolve() in expected_sources
            for mount in mounts
        ):
            owned_ids.append(container_id)
    for container_id in owned_ids:
        try:
            removed = swebench_verified.run_command(
                ["docker", "rm", "-f", container_id], timeout=60, env=env
            )
        except BenchmarkError as exc:
            errors.append(str(exc))
            continue
        if removed.returncode != 0:
            detail = removed.stderr.strip() or removed.stdout.strip()
            if "no such" not in detail.lower():
                errors.append(detail or f"could not remove container {container_id}")
    return errors


def load_evaluation_results(
    path: Path, instance_ids: Sequence[str]
) -> tuple[dict[str, bool], str]:
    try:
        content = path.read_text(encoding="utf-8")
        value = json.loads(content)
    except (
        UnicodeDecodeError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise BenchmarkError(
            f"Could not read official SWE-bench Pro evaluation results: {exc}"
        ) from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and type(passed) is bool for key, passed in value.items()
    ):
        raise BenchmarkError("Official SWE-bench Pro evaluation results are invalid")
    if set(value) != set(instance_ids):
        raise BenchmarkError(
            "Official SWE-bench Pro evaluation results do not cover the requested instances"
        )
    return {instance_id: value[instance_id] for instance_id in instance_ids}, content


def evaluator_python_command(value: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or os.sep in expanded:
        return str(Path(expanded).absolute())
    if os.altsep and os.altsep in expanded:
        return str(Path(expanded).absolute())
    return expanded


def evaluation_cache_contract(
    *,
    evaluation_predictions_sha256: str,
    evaluation_instances_sha256: str,
    instance_ids: Sequence[str],
    dockerhub_username: str,
    docker_platform: str,
    block_network: bool,
    evaluator_runtime: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = {
        "schemaVersion": 1,
        "officialHarnessRef": OFFICIAL_HARNESS_REF,
        "evaluationPredictionsSha256": evaluation_predictions_sha256,
        "evaluationInstancesSha256": evaluation_instances_sha256,
        "instanceIds": list(instance_ids),
        "dockerhubUsername": dockerhub_username,
        "dockerPlatform": docker_platform,
        "blockNetwork": block_network,
        "evaluatorRuntime": evaluator_runtime,
        "useLocalDocker": True,
        "maxWorkers": DEFAULT_EVALUATION_WORKERS,
    }
    encoded = json.dumps(contract, separators=(",", ":"), sort_keys=True)
    return contract, swebench_verified.sha256_text(encoded)


@contextmanager
def exclusive_evaluation_cache(lock_path: Path) -> Iterator[None]:
    """Hold a process lock so identical evaluations cannot tear each other down."""
    if lock_path.is_symlink():
        raise BenchmarkError("Refusing a symlinked SWE-bench Pro evaluation lock")
    lock_path.touch(mode=0o600, exist_ok=True)
    lock_path.chmod(0o600)
    handle = lock_path.open("r+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BenchmarkError(
                    "An evaluation is already using this SWE-bench Pro cache"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BenchmarkError(
                    "An evaluation is already using this SWE-bench Pro cache"
                ) from exc
        locked = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode())
        handle.flush()
        yield
    finally:
        if not handle.closed and locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _cached_output_path(output_dir: Path, run_id: str, instance_id: str) -> Path:
    return output_dir / instance_id / f"{run_id}_output.json"


def _valid_cached_output(path: Path) -> tuple[bool, str | None]:
    if path.is_symlink():
        return False, None
    try:
        content = path.read_text(encoding="utf-8")
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return False, None
    tests = value.get("tests") if isinstance(value, dict) else None
    if not isinstance(tests, list) or not all(
        isinstance(test, dict)
        and isinstance(test.get("name"), str)
        and isinstance(test.get("status"), str)
        for test in tests
    ):
        return False, None
    return True, swebench_verified.sha256_text(content)


def prepare_evaluation_cache(
    output_dir: Path,
    run_id: str,
    instance_ids: Sequence[str],
    cache_key: str,
    *,
    redo: bool,
) -> None:
    marker_path = output_dir / ".hermes-cache.json"
    marker: Any = None
    if not redo:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            marker = None
    expected = marker.get("instanceOutputs") if isinstance(marker, dict) else None
    expected_outputs = expected if isinstance(expected, dict) else {}
    marker_valid = (
        isinstance(marker, dict)
        and marker.get("schemaVersion") == 1
        and marker.get("cacheKey") == cache_key
        and isinstance(expected, dict)
        and set(expected_outputs).issubset(instance_ids)
        and all(
            isinstance(instance_id, str)
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for instance_id, digest in expected_outputs.items()
        )
    )
    for instance_id in instance_ids:
        instance_dir = output_dir / instance_id
        if instance_dir.is_symlink():
            try:
                instance_dir.unlink()
            except OSError as exc:
                raise BenchmarkError(
                    f"Could not clear stale evaluator cache for {instance_id}"
                ) from exc
            continue
        path = _cached_output_path(output_dir, run_id, instance_id)
        valid, digest = _valid_cached_output(path)
        if marker_valid and expected_outputs.get(instance_id) == digest and valid:
            continue
        try:
            if instance_dir.exists():
                shutil.rmtree(instance_dir)
        except OSError as exc:
            raise BenchmarkError(
                f"Could not clear stale evaluator cache for {instance_id}"
            ) from exc
    if not marker_valid or redo:
        try:
            marker_path.unlink(missing_ok=True)
        except OSError as exc:
            raise BenchmarkError("Could not clear the evaluator cache marker") from exc


def write_evaluation_cache_marker(
    output_dir: Path,
    run_id: str,
    instance_ids: Sequence[str],
    cache_key: str,
    evaluation_results_sha256: str,
) -> None:
    outputs = {}
    for instance_id in instance_ids:
        path = _cached_output_path(output_dir, run_id, instance_id)
        valid, digest = _valid_cached_output(path)
        if path.exists() and not valid:
            raise BenchmarkError(
                f"Official evaluator wrote invalid output for {instance_id}"
            )
        if not valid:
            continue
        if digest is not None:
            outputs[instance_id] = digest
    try:
        swebench_verified.atomic_write_json(
            output_dir / ".hermes-cache.json",
            {
                "schemaVersion": 1,
                "cacheKey": cache_key,
                "instanceOutputs": outputs,
                "evaluationResultsSha256": evaluation_results_sha256,
                "completedAt": swebench_verified.utc_now(),
            },
        )
    except OSError as exc:
        raise BenchmarkError("Could not write the evaluator cache marker") from exc


def execute_evaluation(
    *,
    record: dict[str, Any],
    command: Sequence[str],
    harness_dir: Path,
    run_dir: Path,
    evaluation_manifest: Path,
    evaluation_output: Path,
    requested_ids: Sequence[str],
    evaluation_env: dict[str, str],
    process_timeout_seconds: int,
    python: str,
    cache_key: str,
    run_id: str,
    redo: bool,
) -> None:
    swebench_verified.require_docker(env=evaluation_env)
    _require_evaluator_python(python, evaluation_env)
    pre_cleanup_errors = cleanup_evaluation_containers(
        evaluation_output, requested_ids, env=evaluation_env
    )
    if pre_cleanup_errors:
        error = (
            "Could not clean prior SWE-bench Pro evaluator containers: "
            + "; ".join(pre_cleanup_errors)
        )
        swebench_verified.atomic_write_json(
            evaluation_manifest,
            {
                **record,
                "status": "failed",
                "containerCleanupErrors": pre_cleanup_errors,
                "error": error,
                "completedAt": swebench_verified.utc_now(),
            },
        )
        raise BenchmarkError(error)
    prepare_evaluation_cache(
        evaluation_output,
        run_id,
        requested_ids,
        cache_key,
        redo=redo,
    )
    results_path = evaluation_output / "eval_results.json"
    try:
        results_path.unlink(missing_ok=True)
    except OSError as exc:
        raise BenchmarkError(
            "Could not clear the prior SWE-bench Pro aggregate result"
        ) from exc
    swebench_verified.atomic_write_json(
        evaluation_manifest,
        {
            **record,
            "status": "running",
            "preRunContainerCleanupErrors": [],
            "startedAt": swebench_verified.utc_now(),
        },
    )
    result = None
    execution_error: BaseException | None = None
    try:
        result = swebench_verified.run_command(
            command,
            cwd=harness_dir,
            timeout=process_timeout_seconds,
            env=evaluation_env,
            umask=0o077,
        )
    except BaseException as exc:
        execution_error = exc
    final: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    with swebench_verified.defer_cleanup_signals() as deferred_signals:
        cleanup_errors = cleanup_evaluation_containers(
            evaluation_output, requested_ids, env=evaluation_env
        )
        evaluation_results = None
        evaluation_results_content = None
        if execution_error is None and result is not None and result.returncode == 0:
            try:
                evaluation_results, evaluation_results_content = (
                    load_evaluation_results(results_path, requested_ids)
                )
            except BenchmarkError as exc:
                execution_error = exc
        if (
            execution_error is None
            and evaluation_results_content is not None
            and not cleanup_errors
        ):
            try:
                write_evaluation_cache_marker(
                    evaluation_output,
                    run_id,
                    requested_ids,
                    cache_key,
                    swebench_verified.sha256_text(evaluation_results_content),
                )
            except BenchmarkError as exc:
                execution_error = exc
        status = (
            "completed"
            if evaluation_results is not None
            and execution_error is None
            and not cleanup_errors
            else "failed"
        )
        if isinstance(execution_error, (KeyboardInterrupt, ControllerTerminated)):
            status = "interrupted"
        swebench_verified.atomic_write_text(
            run_dir / "evaluation.stdout.log", result.stdout if result else ""
        )
        swebench_verified.atomic_write_text(
            run_dir / "evaluation.stderr.log", result.stderr if result else ""
        )
        final = {
            **record,
            "status": status,
            "exitCode": result.returncode if result else None,
            "preRunContainerCleanupErrors": [],
            "containerCleanupErrors": cleanup_errors,
            "completedAt": swebench_verified.utc_now(),
        }
        if evaluation_results is not None and evaluation_results_content is not None:
            final.update({
                "evaluationResultsPath": str(results_path),
                "evaluationResultsSha256": swebench_verified.sha256_text(
                    evaluation_results_content
                ),
                "resolvedInstanceIds": [
                    instance_id
                    for instance_id, passed in evaluation_results.items()
                    if passed
                ],
                "unresolvedInstanceIds": [
                    instance_id
                    for instance_id, passed in evaluation_results.items()
                    if not passed
                ],
            })
        if execution_error is not None:
            final["error"] = f"{type(execution_error).__name__}: {execution_error}"
        swebench_verified.atomic_write_json(evaluation_manifest, final)
    if deferred_signals and execution_error is None:
        execution_error = (
            ControllerTerminated(deferred_signals[0])
            if deferred_signals[0] == signal.SIGTERM
            else KeyboardInterrupt()
        )
        final["status"] = "interrupted"
        final["error"] = f"{type(execution_error).__name__}: {execution_error}"
        swebench_verified.atomic_write_json(evaluation_manifest, final)
    if execution_error is not None:
        raise execution_error
    if result is None or result.returncode != 0:
        raise BenchmarkError(
            "Official SWE-bench Pro evaluation failed"
            + (f" with exit code {result.returncode}" if result else "")
        )
    if cleanup_errors:
        raise BenchmarkError(
            "Official SWE-bench Pro evaluation left containers behind: "
            + "; ".join(cleanup_errors)
        )


def run_evaluation(args: argparse.Namespace) -> None:
    swebench_verified.validate_run_id(args.run_id)
    swebench_verified.validate_docker_platform(args.docker_platform)
    run_dir = Path(args.output_dir).resolve() / "runs" / args.run_id
    predictions_path = (
        Path(args.predictions_path).resolve()
        if args.predictions_path
        else run_dir / "predictions.json"
    )
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else run_dir / "prediction-manifest.json"
    )
    manifest, predictions, digest = verify_prediction_artifact(
        predictions_path, manifest_path
    )
    if manifest.get("runId") != args.run_id:
        raise BenchmarkError("Prediction manifest run id does not match --run-id")
    available_ids = [prediction["instance_id"] for prediction in predictions]
    requested_ids = args.instance_id or available_ids
    if len(set(requested_ids)) != len(requested_ids):
        raise BenchmarkError("Duplicate --instance-id values are not allowed")
    missing = [
        instance_id for instance_id in requested_ids if instance_id not in available_ids
    ]
    if missing:
        raise BenchmarkError(
            f"Prediction artifact does not contain: {', '.join(missing)}"
        )
    harness_dir = ensure_pinned_harness(args.harness_dir)
    validate_harness_instances(harness_dir, requested_ids)
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_manifest = run_dir / "evaluation-manifest.json"
    with DatasetRowsClient() as client:
        raw_rows = client.evaluator_rows(requested_ids)
    selected_by_id = {
        item["instanceId"]: item for item in manifest["selectedInstances"]
    }
    rows = [
        sanitize_evaluator_row(raw_row, selected_by_id[instance_id])
        for raw_row, instance_id in zip(raw_rows, requested_ids, strict=True)
    ]
    rows_content = swebench_verified.encode_jsonl(rows)
    predictions_by_id = {
        prediction["instance_id"]: prediction for prediction in predictions
    }
    selected_predictions = [
        predictions_by_id[instance_id] for instance_id in requested_ids
    ]
    prediction_content = encode_predictions(selected_predictions)
    evaluation_predictions_sha256 = swebench_verified.sha256_text(prediction_content)
    evaluation_instances_sha256 = swebench_verified.sha256_text(rows_content)
    python = evaluator_python_command(args.python)
    evaluation_env = swebench_verified._evaluation_environment()
    evaluation_env["PYTHONDONTWRITEBYTECODE"] = "1"
    evaluator_runtime = evaluator_runtime_metadata(python, evaluation_env)
    cache_contract, cache_key = evaluation_cache_contract(
        evaluation_predictions_sha256=evaluation_predictions_sha256,
        evaluation_instances_sha256=evaluation_instances_sha256,
        instance_ids=requested_ids,
        dockerhub_username=args.dockerhub_username,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
        evaluator_runtime=evaluator_runtime,
    )
    evaluation_root = run_dir / "evaluation"
    if evaluation_root.is_symlink():
        raise BenchmarkError("Refusing a symlinked SWE-bench Pro evaluation directory")
    evaluation_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evaluation_root.chmod(0o700)
    evaluation_output = evaluation_root / cache_key
    if evaluation_output.is_symlink():
        raise BenchmarkError("Refusing a symlinked SWE-bench Pro evaluation cache")
    evaluation_output.mkdir(mode=0o700, exist_ok=True)
    evaluation_output.chmod(0o700)
    evaluation_instances = evaluation_output / "evaluation-instances.jsonl"
    evaluation_predictions = evaluation_output / "evaluation-predictions.json"
    command = build_evaluation_command(
        python,
        harness_dir,
        evaluation_instances,
        evaluation_predictions,
        evaluation_output,
        dockerhub_username=args.dockerhub_username,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
        redo=args.redo,
    )
    process_timeout_seconds = len(requested_ids) * (
        DEFAULT_EVALUATION_TIMEOUT_SECONDS + DEFAULT_EVALUATION_SETUP_GRACE_SECONDS
    )
    record = {
        "schemaVersion": 1,
        "benchmark": BENCHMARK,
        "dataset": DATASET_NAME,
        "datasetRevision": DATASET_REVISION,
        "runId": args.run_id,
        "predictionsPath": str(predictions_path),
        "predictionManifestPath": str(manifest_path),
        "predictionsSha256": digest,
        "evaluationPredictionsPath": str(evaluation_predictions),
        "evaluationPredictionsSha256": evaluation_predictions_sha256,
        "officialHarnessRepository": OFFICIAL_HARNESS_REPOSITORY,
        "officialHarnessRef": OFFICIAL_HARNESS_REF,
        "officialHarnessDir": str(harness_dir),
        "evaluationInstancesPath": str(evaluation_instances),
        "evaluationInstancesSha256": evaluation_instances_sha256,
        "evaluationOutput": str(evaluation_output),
        "evaluationCacheContract": cache_contract,
        "evaluationCacheKey": cache_key,
        "evaluatorRuntime": evaluator_runtime,
        "instanceIds": requested_ids,
        "maxWorkers": DEFAULT_EVALUATION_WORKERS,
        "useLocalDocker": True,
        "blockNetwork": args.block_network,
        "redo": args.redo,
        "evaluationProcessTimeoutSeconds": process_timeout_seconds,
        "command": command,
        "plannedAt": swebench_verified.utc_now(),
    }
    with exclusive_evaluation_cache(evaluation_root / ".hermes.lock"):
        swebench_verified.atomic_write_text(evaluation_instances, rows_content)
        swebench_verified.atomic_write_text(evaluation_predictions, prediction_content)
        if args.dry_run:
            swebench_verified.atomic_write_json(
                evaluation_manifest, {**record, "status": "dry-run"}
            )
            print(json.dumps({**record, "status": "dry-run"}, indent=2))
            return
        execute_evaluation(
            record=record,
            command=command,
            harness_dir=harness_dir,
            run_dir=run_dir,
            evaluation_manifest=evaluation_manifest,
            evaluation_output=evaluation_output,
            requested_ids=requested_ids,
            evaluation_env=evaluation_env,
            process_timeout_seconds=process_timeout_seconds,
            python=python,
            cache_key=cache_key,
            run_id=args.run_id,
            redo=args.redo,
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser(
    agent_topology: str = swebench_verified.DEFAULT_AGENT_TOPOLOGY,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "hermes-swebench-pro-single"
            if agent_topology == swebench_verified.SINGLE_AGENT_TOPOLOGY
            else "hermes-swebench-pro"
        ),
        description="Run Hermes Agent on SWE-bench Pro with local Docker evaluation.",
    )
    parser.set_defaults(agent_topology=agent_topology)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="Generate SWE-bench Pro predictions")
    infer.add_argument("--run-id", default=default_run_id())
    infer.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    infer.add_argument("--instance-id", action="append", default=[])
    infer.add_argument("--max-instances", type=_positive_int)
    infer.add_argument("--offset", type=_nonnegative_int)
    infer.add_argument(
        "--model",
        default=(
            swebench_verified.SINGLE_AGENT_DEFAULT_MODEL
            if agent_topology == swebench_verified.SINGLE_AGENT_TOPOLOGY
            else DEFAULT_MODEL
        ),
    )
    infer.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX)
    infer.add_argument("--docker-platform", default=DEFAULT_DOCKER_PLATFORM)
    infer.add_argument(
        "--agent-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
    )
    infer.add_argument(
        "--setup-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_SETUP_TIMEOUT_SECONDS,
    )
    infer.add_argument("--restart", action="store_true")
    infer.add_argument("--dry-run", action="store_true")
    tracing = infer.add_mutually_exclusive_group()
    tracing.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIR,
        help=(
            "Override the benchmark-trace/v1 output base "
            f"(default: {DEFAULT_TRACE_DIR})"
        ),
    )
    tracing.add_argument(
        "--no-trace",
        action="store_const",
        const=None,
        dest="trace_dir",
        help="Disable benchmark tracing for this inference run",
    )

    evaluate = subparsers.add_parser(
        "evaluate", aliases=["eval"], help="Run Scale's pinned local Docker evaluator"
    )
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--predictions-path")
    evaluate.add_argument("--manifest-path")
    evaluate.add_argument("--instance-id", action="append", default=[])
    evaluate.add_argument("--python", default=sys.executable)
    evaluate.add_argument("--harness-dir")
    evaluate.add_argument("--dockerhub-username", default=DEFAULT_DOCKERHUB_USERNAME)
    evaluate.add_argument("--docker-platform", default=DEFAULT_DOCKER_PLATFORM)
    evaluate.add_argument("--block-network", action="store_true")
    evaluate.add_argument("--redo", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")

    listing = subparsers.add_parser("list", help="List selected public instances")
    listing.add_argument("--instance-id", action="append", default=[])
    listing.add_argument("--max-instances", type=_positive_int, default=1)
    listing.add_argument("--offset", type=_nonnegative_int, default=0)
    return parser


def options_from_args(args: argparse.Namespace) -> InferenceOptions:
    explicit_selection = (
        bool(args.instance_id)
        or args.max_instances is not None
        or args.offset is not None
    )
    return InferenceOptions(
        run_id=args.run_id,
        output_dir=Path(args.output_dir),
        instance_ids=(
            tuple(args.instance_id)
            if explicit_selection
            else (DEFAULT_SMOKE_INSTANCE_ID,)
        ),
        max_instances=args.max_instances if args.max_instances is not None else 1,
        offset=args.offset if args.offset is not None else 0,
        model=swebench_verified.canonical_model(args.model),
        image_prefix=args.image_prefix,
        docker_platform=args.docker_platform,
        agent_timeout_seconds=args.agent_timeout_seconds,
        setup_timeout_seconds=args.setup_timeout_seconds,
        restart=args.restart,
        dry_run=args.dry_run,
        trace_dir=args.trace_dir,
        agent_topology=args.agent_topology,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    agent_topology: str = swebench_verified.DEFAULT_AGENT_TOPOLOGY,
) -> int:
    parser = build_parser(agent_topology)
    args = parser.parse_args(argv)

    def terminate(signum: int, _frame: Any) -> None:
        raise ControllerTerminated(signum)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        if args.command == "infer":
            paths = run_inference(options_from_args(args))
            if not args.dry_run:
                print(f"Predictions: {paths.predictions}")
            return 0
        if args.command in {"evaluate", "eval"}:
            run_evaluation(args)
            return 0
        with DatasetRowsClient() as client:
            rows = client.select(
                tuple(args.instance_id),
                offset=args.offset,
                max_instances=args.max_instances,
            )
        print(swebench_verified.encode_jsonl(row.public_dict() for row in rows), end="")
        return 0
    except BenchmarkError as exc:
        parser.exit(2, f"error: {exc}\n")
    except ControllerTerminated as exc:
        print(
            "Benchmark interrupted; cleanup was attempted and partial artifacts "
            "were preserved.",
            file=sys.stderr,
        )
        return 128 + exc.signum
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; cleanup was attempted and partial artifacts "
            "were preserved.",
            file=sys.stderr,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def swebenchpro_single_main(argv: Sequence[str] | None = None) -> int:
    return main(argv, agent_topology=swebench_verified.SINGLE_AGENT_TOPOLOGY)


if __name__ == "__main__":
    raise SystemExit(main())
