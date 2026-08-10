from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hermes_cli.benchmarks import swebench_pro as benchmark
from hermes_cli.benchmarks import swebench_pro_worker as pro_worker
from hermes_cli.benchmarks import swebench_verified as shared
from hermes_cli.benchmarks import swebench_verified_worker as shared_worker


INSTANCE = benchmark.DEFAULT_SMOKE_INSTANCE_ID
BASE_COMMIT = "a" * 40
DOCKERHUB_TAG = (
    "qutebrowser.qutebrowser-qutebrowser__qutebrowser-"
    "5fdc83e5da6222fe61163395baaad7ae57fa2cb4-"
    "v363c8a7e5ccdf6968fc7ab84a2053ac780366"
)
SOURCE_IDENTITY = {"commit": "test", "dirty": False, "fingerprint": "test"}
EVALUATOR_RUNTIME = {
    "pythonImplementation": "cpython",
    "pythonVersion": "3.13.0",
    "packages": {"docker": "7.1.0", "pandas": "2.3.0", "tqdm": "4.67.0"},
}


def make_raw_row(**overrides):
    value = {
        "repo": "qutebrowser/qutebrowser",
        "instance_id": INSTANCE,
        "base_commit": BASE_COMMIT,
        "problem_statement": "Fix the browser regression.",
        "requirements": "Preserve existing behavior.",
        "interface": "No public API changes.",
        "repo_language": "python",
        "dockerhub_tag": DOCKERHUB_TAG,
        "before_repo_set_cmd": (
            f"git reset --hard {BASE_COMMIT}\n"
            f"git checkout {'b' * 40} -- tests/test_browser.py"
        ),
        "selected_test_files_to_run": json.dumps(["tests/test_browser.py"]),
        "fail_to_pass": json.dumps(["test_browser_regression"]),
        "pass_to_pass": json.dumps([]),
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET HIDDEN TEST",
    }
    value.update(overrides)
    return value


def make_options(tmp_path: Path, **overrides):
    value = {
        "run_id": "pro-test-run",
        "output_dir": tmp_path,
        "instance_ids": (INSTANCE,),
        "max_instances": 1,
        "offset": 0,
        "model": benchmark.DEFAULT_MODEL,
        "image_prefix": benchmark.DEFAULT_IMAGE_PREFIX,
        "docker_platform": benchmark.DEFAULT_DOCKER_PLATFORM,
        "agent_timeout_seconds": benchmark.DEFAULT_AGENT_TIMEOUT_SECONDS,
        "setup_timeout_seconds": benchmark.DEFAULT_SETUP_TIMEOUT_SECONDS,
        "restart": False,
        "dry_run": False,
        "source_identity": SOURCE_IDENTITY,
    }
    value.update(overrides)
    return benchmark.InferenceOptions(**value)


def make_harness(path: Path, *, instance_id: str = INSTANCE) -> Path:
    files = (
        path / "swe_bench_pro_eval.py",
        path / "helper_code" / "image_uri.py",
        path / "run_scripts" / instance_id / "run_script.sh",
        path / "run_scripts" / instance_id / "parser.py",
        path / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile",
        path / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile",
    )
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")
    return path


def test_cli_defaults_match_peer_frameworks(tmp_path):
    args = benchmark.build_parser().parse_args(["infer", "--output-dir", str(tmp_path)])
    options = benchmark.options_from_args(args)

    assert options.instance_ids == (INSTANCE,)
    assert options.model == "openrouter/qwen/qwen3-coder-next"
    assert options.agent_timeout_seconds == 1800
    assert options.setup_timeout_seconds == 600
    assert options.agent_topology == shared.DEFAULT_AGENT_TOPOLOGY
    assert benchmark.DEFAULT_INFERENCE_WORKERS == 1
    assert benchmark.DEFAULT_EVALUATION_WORKERS == 1
    assert benchmark.DEFAULT_ATTEMPTS == 1
    assert benchmark.DEFAULT_INFRASTRUCTURE_RETRIES == 0
    assert shared.DEFAULT_API_MAX_RETRIES == 1
    assert shared.DEFAULT_AGENT_SEQUENCE == (
        "coordinator",
        "navigator",
        "patcher",
        "reviewer",
    )
    assert shared.DEFAULT_COORDINATOR_BUDGET == 24
    assert shared.DEFAULT_NATIVE_SUBAGENT_BUDGET == 13
    assert shared.DEFAULT_NATIVE_SUBAGENT_COUNT == 3
    assert shared.DEFAULT_DELEGATION_MODE == "native"
    assert shared.DEFAULT_PHASE_BUDGETS == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }


def test_single_agent_cli_defaults_preserve_pro_budget(tmp_path):
    args = benchmark.build_parser(shared.SINGLE_AGENT_TOPOLOGY).parse_args([
        "infer",
        "--output-dir",
        str(tmp_path),
    ])
    options = benchmark.options_from_args(args)

    assert options.agent_topology == "single-agent"
    assert options.instance_ids == (INSTANCE,)
    assert options.agent_timeout_seconds == 1800
    assert options.model == "openrouter/poolside/laguna-s-2.1:free"
    assert options.trace_dir == benchmark.DEFAULT_TRACE_DIR


def test_explicit_instance_ids_and_window_disable_smoke_default():
    ids = ["example__first-1", INSTANCE]
    args = benchmark.build_parser().parse_args([
        "infer",
        "--instance-id",
        ids[0],
        "--instance-id",
        ids[1],
    ])
    assert benchmark.options_from_args(args).instance_ids == tuple(ids)

    window = benchmark.build_parser().parse_args([
        "infer",
        "--max-instances",
        "2",
    ])
    assert benchmark.options_from_args(window).instance_ids == ()


def test_dataset_projection_never_retains_evaluator_fields():
    row = benchmark.parse_swebench_pro_row(make_raw_row())

    assert set(row.public_dict()) == benchmark.PUBLIC_DATASET_FIELDS
    serialized = json.dumps(row.public_dict())
    assert "SECRET GOLD PATCH" not in serialized
    assert "SECRET HIDDEN TEST" not in serialized
    assert "FAIL_TO_PASS" not in serialized


def test_evaluator_projection_is_minimal_and_neutralizes_host_eval(tmp_path):
    raw = make_raw_row(
        selected_test_files_to_run="['a.py', 'quote\\'s.py', 'snowman-☃.py']",
        fail_to_pass='["test one", "test\\\\two"]',
        pass_to_pass="[]",
    )
    row = benchmark.parse_swebench_pro_row(raw)
    selected = benchmark._selected_instances(make_options(tmp_path), [row])[0]

    projected = benchmark.sanitize_evaluator_row(raw, selected)

    assert set(projected) == {
        "repo",
        "instance_id",
        "base_commit",
        "before_repo_set_cmd",
        *benchmark.EVALUATOR_LIST_FIELDS,
    }
    assert json.loads(projected["selected_test_files_to_run"]) == [
        "a.py",
        "quote's.py",
        "snowman-☃.py",
    ]
    assert json.loads(projected["fail_to_pass"]) == ["test one", "test\\two"]
    assert json.loads(projected["pass_to_pass"]) == []
    assert "patch" not in projected
    assert "test_patch" not in projected

    marker = tmp_path / "executed"
    malicious = make_raw_row(
        fail_to_pass=(f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')")
    )
    with pytest.raises(benchmark.BenchmarkError, match="unsafe 'fail_to_pass'"):
        benchmark.sanitize_evaluator_row(malicious, selected)
    assert not marker.exists()


def test_evaluator_projection_rejects_public_row_drift_and_oversized_lists(tmp_path):
    raw = make_raw_row()
    row = benchmark.parse_swebench_pro_row(raw)
    selected = benchmark._selected_instances(make_options(tmp_path), [row])[0]

    with pytest.raises(benchmark.BenchmarkError, match="public row changed"):
        benchmark.sanitize_evaluator_row(
            make_raw_row(problem_statement="A different task"), selected
        )

    with pytest.raises(benchmark.BenchmarkError, match="invalid 'pass_to_pass'"):
        benchmark.sanitize_evaluator_row(
            make_raw_row(
                pass_to_pass=" " * (benchmark.MAX_EVALUATOR_LIST_FIELD_CHARS + 1)
            ),
            selected,
        )


def test_dataset_client_requires_pinned_revision():
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-revision": "b" * 40},
            json={"rows": [{"row": make_raw_row()}]},
        )

    client = benchmark.DatasetRowsClient(
        client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(benchmark.BenchmarkError, match="dataset revision changed"):
        client.raw_page(0, 1)


def test_dataset_client_preserves_requested_order_and_can_fetch_private_rows():
    second_id = "example__issue-2"
    first = make_raw_row()
    second = make_raw_row(instance_id=second_id, repo="example/example")

    def respond(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        rows = [first, second] if offset == 0 else []
        return httpx.Response(
            200,
            headers={"x-revision": benchmark.DATASET_REVISION},
            json={"rows": [{"row": row} for row in rows]},
        )

    client = benchmark.DatasetRowsClient(
        client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    selected = client.select([second_id, INSTANCE], offset=0, max_instances=1)
    evaluator = client.evaluator_rows([INSTANCE])

    assert [row.instance_id for row in selected] == [second_id, INSTANCE]
    assert set(selected[0].public_dict()) == benchmark.PUBLIC_DATASET_FIELDS
    assert evaluator[0]["patch"] == "SECRET GOLD PATCH"
    with pytest.raises(benchmark.BenchmarkError, match="Duplicate"):
        client.select([INSTANCE, INSTANCE], offset=0, max_instances=1)


def test_dataset_window_selection_paginates():
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        requests.append((offset, length))
        rows = [
            make_raw_row(
                instance_id=f"example__issue-{index}",
                repo="example/example",
            )
            for index in range(offset, offset + length)
        ]
        return httpx.Response(
            200,
            headers={"x-revision": benchmark.DATASET_REVISION},
            json={"rows": [{"row": row} for row in rows]},
        )

    client = benchmark.DatasetRowsClient(
        client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    rows = client.select([], offset=25, max_instances=225)

    assert len(rows) == 225
    assert rows[0].instance_id == "example__issue-25"
    assert rows[-1].instance_id == "example__issue-249"
    assert requests == [(25, 100), (125, 100), (225, 25)]


def test_problem_format_prompt_and_official_image_match_pro_contract():
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    formatted = benchmark.format_problem_statement(row)
    prompt = benchmark.build_prompt(row)

    assert formatted == (
        "Fix the browser regression.\n\nRequirements:\nPreserve existing behavior."
        "\n\nNew interfaces introduced:\nNo public API changes."
    )
    assert "Worktree: /app" in prompt
    assert "Repository language: python" in prompt
    assert formatted in prompt
    assert benchmark.official_image(row.dockerhub_tag) == (
        f"docker.io/jefzda/sweap-images:{DOCKERHUB_TAG}"
    )
    with pytest.raises(benchmark.BenchmarkError, match="dockerhub_tag"):
        benchmark.official_image("../unsafe")


def test_pro_worker_configuration_uses_app_and_restores_verified_globals(tmp_path):
    request = {
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(DOCKERHUB_TAG),
        "dockerPlatform": "linux/amd64",
    }
    verified_values = (
        shared_worker.WORKER_BENCHMARK,
        shared_worker.WORKTREE,
        shared_worker.CONDA_ACTIVATION,
        shared_worker.ROW_PARSER,
    )

    with pro_worker.configuration():
        config = shared_worker._terminal_config(request)
        assert shared_worker.WORKER_BENCHMARK == benchmark.BENCHMARK
        assert shared_worker.WORKTREE == "/app"
        assert shared_worker.CONDA_ACTIVATION == ""
        assert config["terminal"]["cwd"] == "/app"
        assert config["terminal"]["container_memory"] == 0
        assert shared_worker.ROW_PARSER is benchmark.parse_swebench_pro_row
        single_config = shared_worker._terminal_config({
            **request,
            "agentTopology": shared.SINGLE_AGENT_TOPOLOGY,
        })
        assert "delegation" not in single_config

    assert (
        shared_worker.WORKER_BENCHMARK,
        shared_worker.WORKTREE,
        shared_worker.CONDA_ACTIVATION,
        shared_worker.ROW_PARSER,
    ) == verified_values


def test_pro_worker_setup_resets_app_without_assuming_conda(monkeypatch):
    commands = []
    overrides = []
    fake_environment = object()

    def terminal_tool(**kwargs):
        commands.append(kwargs)
        return json.dumps({"exit_code": 0, "output": ""})

    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda *args: overrides.append(args),
    )
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", terminal_tool)
    monkeypatch.setattr(
        "tools.terminal_tool.get_active_env", lambda _task_id: fake_environment
    )
    with pro_worker.configuration():
        result = shared_worker.setup_environment({
            "taskId": "pro-setup",
            "image": benchmark.official_image(DOCKERHUB_TAG),
            "row": {"base_commit": BASE_COMMIT},
        })

    assert result is fake_environment
    assert commands[0]["workdir"] == "/app"
    assert "git -C /app reset --hard" in commands[0]["command"]
    assert "conda" not in commands[0]["command"]
    assert overrides == [("pro-setup", {"cwd": "/app"})]


def test_pro_prompt_and_native_delegation_contract_are_complete():
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    prompt = benchmark.build_prompt(row)
    system = pro_worker.COORDINATOR_SYSTEM_PROMPT

    assert "SWE-bench Pro" in prompt
    assert "Worktree: /app" in prompt
    assert benchmark.format_problem_statement(row) in prompt
    assert "native delegate_task" in system
    assert "[benchmark-navigator]" in system
    assert "[benchmark-patcher]" in system
    assert "[benchmark-reviewer]" in system
    assert "prior handoffs" in system


def test_single_agent_prompt_assigns_the_complete_pro_workflow():
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    prompt = benchmark.build_prompt(
        row,
        agent_topology=shared.SINGLE_AGENT_TOPOLOGY,
    )

    assert "sole coding agent" in prompt
    assert "Do not delegate" in prompt
    assert benchmark.format_problem_statement(row) in prompt
    assert "navigator, patcher, reviewer" not in prompt


def test_prediction_schema_and_manifest_record_parity(tmp_path):
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    options = make_options(tmp_path)
    prediction = benchmark._prediction(
        row, options.run_id, {"modelPatch": "diff --git"}
    )
    manifest = benchmark._manifest(options, [row], [prediction], complete=True)

    assert set(prediction) == {"instance_id", "patch", "prefix"}
    assert prediction["prefix"] == options.run_id
    assert manifest["containerWorktree"] == "/app"
    assert manifest["inferenceWorkers"] == 1
    assert manifest["attemptsPerInstance"] == 1
    assert manifest["maxInfrastructureRetries"] == 0
    assert manifest["agentTimeoutSeconds"] == 1800
    assert manifest["agentSequence"] == [
        "coordinator",
        "navigator",
        "patcher",
        "reviewer",
    ]
    assert manifest["agentBudgets"] == {
        "coordinator": 24,
        "nativeSubagent": 13,
        "nativeSubagentCount": 3,
    }
    assert manifest["delegationMode"] == "native"
    assert manifest["peerPhaseBudgetReference"] == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }
    assert manifest["datasetRevision"] == benchmark.DATASET_REVISION
    assert manifest["selectedInstances"][0]["publicRowSha256"] == (
        benchmark.public_row_sha256(row)
    )
    assert manifest["instancesSha256"]


def test_single_agent_manifest_disables_delegation_metadata(tmp_path):
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    options = make_options(tmp_path, agent_topology=shared.SINGLE_AGENT_TOPOLOGY)
    manifest = benchmark._manifest(options, [row], [], complete=False)

    assert manifest["agentTopology"] == "single-agent"
    assert manifest["primaryAgentRole"] == "agent"
    assert manifest["delegationEnabled"] is False
    assert manifest["agentSequence"] == ["agent"]
    assert manifest["agentBudgets"] == {"singleAgent": 24}
    assert manifest["delegationMode"] == "disabled"
    assert manifest["peerPhaseBudgetReference"] is None


def write_complete_artifact(tmp_path: Path):
    row = benchmark.parse_swebench_pro_row(make_raw_row())
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    prediction = benchmark._prediction(row, options.run_id, {"modelPatch": "patch"})
    content = benchmark.encode_predictions([prediction])
    paths.run_dir.mkdir(parents=True)
    shared.atomic_write_text(paths.predictions, content)
    shared.atomic_write_json(
        paths.manifest, benchmark._manifest(options, [row], [prediction], complete=True)
    )
    return options, paths, row, prediction


def test_prediction_artifact_rejects_mutation(tmp_path):
    _options, paths, _row, _prediction = write_complete_artifact(tmp_path)
    manifest, predictions, digest = benchmark.verify_prediction_artifact(
        paths.predictions, paths.manifest
    )
    assert manifest["complete"] is True
    assert predictions[0]["instance_id"] == INSTANCE
    assert digest == manifest["predictionsSha256"]

    paths.predictions.write_text("[]\n", encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="changed after inference"):
        benchmark.verify_prediction_artifact(paths.predictions, paths.manifest)


def test_evaluation_command_is_pinned_to_local_docker_and_one_worker(tmp_path):
    command = benchmark.build_evaluation_command(
        "/venv/bin/python",
        tmp_path / "harness",
        tmp_path / "rows.jsonl",
        tmp_path / "predictions.json",
        tmp_path / "output",
        dockerhub_username="jefzda",
        docker_platform="linux/amd64",
        block_network=True,
        redo=True,
    )

    assert command[0] == "/venv/bin/python"
    assert "--num_workers=1" in command
    assert "--dockerhub_username=jefzda" in command
    assert "--use_local_docker" in command
    assert "--docker_platform=linux/amd64" in command
    assert "--block_network" in command
    assert "--redo" in command
    assert not any("modal" in argument for argument in command)
    with pytest.raises(benchmark.BenchmarkError, match="Docker Hub username"):
        benchmark.build_evaluation_command(
            "/venv/bin/python",
            tmp_path / "harness",
            tmp_path / "rows.jsonl",
            tmp_path / "predictions.json",
            tmp_path / "output",
            dockerhub_username="../unsafe",
            docker_platform="linux/amd64",
            block_network=False,
            redo=False,
        )


def test_evaluator_python_and_cache_contract_are_stable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launcher = tmp_path / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to("/usr/bin/python3")
    assert benchmark.evaluator_python_command(".venv/bin/python") == str(
        launcher.absolute()
    )
    assert benchmark.evaluator_python_command(str(launcher)) == str(launcher)
    assert benchmark.evaluator_python_command("python") == "python"

    values = {
        "evaluation_predictions_sha256": "a" * 64,
        "evaluation_instances_sha256": "b" * 64,
        "instance_ids": [INSTANCE],
        "dockerhub_username": "jefzda",
        "docker_platform": "linux/amd64",
        "block_network": False,
        "evaluator_runtime": EVALUATOR_RUNTIME,
    }
    contract, key = benchmark.evaluation_cache_contract(**values)
    assert benchmark.evaluation_cache_contract(**values) == (contract, key)
    _, blocked_key = benchmark.evaluation_cache_contract(**{
        **values,
        "block_network": True,
    })
    assert blocked_key != key


def test_evaluator_runtime_metadata_is_versioned(monkeypatch):
    output = benchmark.EVALUATOR_RUNTIME_SENTINEL + json.dumps(EVALUATOR_RUNTIME)
    monkeypatch.setattr(
        shared,
        "run_command",
        lambda args, **_kwargs: shared.CommandResult(tuple(args), 0, output, ""),
    )

    assert benchmark.evaluator_runtime_metadata("python", {}) == EVALUATOR_RUNTIME


def test_evaluation_cache_lock_rejects_a_concurrent_run(tmp_path):
    lock = tmp_path / ".hermes.lock"
    with benchmark.exclusive_evaluation_cache(lock):
        with pytest.raises(benchmark.BenchmarkError, match="already using"):
            with benchmark.exclusive_evaluation_cache(lock):
                pass


def test_evaluation_cache_preserves_only_marker_bound_outputs(tmp_path):
    output_dir = tmp_path / "cache"
    output_path = output_dir / INSTANCE / "pro-test-run_output.json"
    output_path.parent.mkdir(parents=True)
    content = json.dumps({"tests": [{"name": "test", "status": "PASSED"}]})
    output_path.write_text(content, encoding="utf-8")
    marker = {
        "schemaVersion": 1,
        "cacheKey": "a" * 64,
        "instanceOutputs": {INSTANCE: shared.sha256_text(content)},
    }
    shared.atomic_write_json(output_dir / ".hermes-cache.json", marker)

    benchmark.prepare_evaluation_cache(
        output_dir,
        "pro-test-run",
        [INSTANCE],
        "a" * 64,
        redo=False,
    )
    assert output_path.exists()

    (output_path.parent / "workspace").mkdir()
    (output_path.parent / "workspace" / "output.json").write_text(
        "stale", encoding="utf-8"
    )
    shared.atomic_write_json(
        output_dir / ".hermes-cache.json",
        {**marker, "instanceOutputs": {INSTANCE: 1}},
    )
    benchmark.prepare_evaluation_cache(
        output_dir,
        "pro-test-run",
        [INSTANCE],
        "a" * 64,
        redo=False,
    )
    assert not output_path.parent.exists()
    assert not (output_dir / ".hermes-cache.json").exists()


def test_cleanup_evaluation_containers_removes_only_owned_workspaces(
    tmp_path, monkeypatch
):
    owned = "a" * 64
    foreign = "b" * 64
    commands = []

    def run_command(args, **_kwargs):
        commands.append(args)
        if args[:3] == ["docker", "ps", "-aq"]:
            return shared.CommandResult(tuple(args), 0, f"{owned}\n{foreign}\n", "")
        if args[:2] == ["docker", "inspect"]:
            metadata = {
                owned: {
                    "Id": owned,
                    "Mounts": [
                        {
                            "Type": "bind",
                            "Source": str(tmp_path / INSTANCE / "workspace"),
                            "Destination": "/workspace",
                        }
                    ],
                },
                foreign: {
                    "Id": foreign,
                    "Mounts": [
                        {
                            "Type": "bind",
                            "Source": str(tmp_path / "someone-else" / "workspace"),
                            "Destination": "/workspace",
                        }
                    ],
                },
            }
            return shared.CommandResult(
                tuple(args), 0, json.dumps([metadata[args[2]]]), ""
            )
        return shared.CommandResult(tuple(args), 0, owned, "")

    monkeypatch.setattr(shared, "run_command", run_command)

    assert benchmark.cleanup_evaluation_containers(tmp_path, [INSTANCE], env={}) == []
    assert commands[-1] == ["docker", "rm", "-f", owned]


def test_validate_harness_requires_exact_revision(tmp_path, monkeypatch):
    make_harness(tmp_path)

    def clean_harness(args, **_kwargs):
        stdout = f"{benchmark.OFFICIAL_HARNESS_REF}\n" if "rev-parse" in args else ""
        return shared.CommandResult(tuple(args), 0, stdout, "")

    monkeypatch.setattr(shared, "run_command", clean_harness)
    assert benchmark.validate_pinned_harness(tmp_path) == tmp_path

    bad = SimpleNamespace(returncode=0, stdout=f"{'b' * 40}\n", stderr="")
    monkeypatch.setattr(shared, "run_command", lambda *_args, **_kwargs: bad)
    with pytest.raises(benchmark.BenchmarkError, match="must be pinned"):
        benchmark.validate_pinned_harness(tmp_path)

    def dirty_harness(args, **_kwargs):
        stdout = (
            f"{benchmark.OFFICIAL_HARNESS_REF}\n"
            if "rev-parse" in args
            else " M swe_bench_pro_eval.py\n"
        )
        return shared.CommandResult(tuple(args), 0, stdout, "")

    monkeypatch.setattr(shared, "run_command", dirty_harness)
    with pytest.raises(benchmark.BenchmarkError, match="local or ignored files"):
        benchmark.validate_pinned_harness(tmp_path)


def test_validate_harness_requires_each_instance_contract(tmp_path):
    harness = make_harness(tmp_path)
    benchmark.validate_harness_instances(harness, [INSTANCE])

    (harness / "run_scripts" / INSTANCE / "parser.py").unlink()
    with pytest.raises(benchmark.BenchmarkError, match="missing files"):
        benchmark.validate_harness_instances(harness, [INSTANCE])


def test_evaluation_results_require_exact_boolean_instance_coverage(tmp_path):
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps({INSTANCE: True}), encoding="utf-8")

    results, content = benchmark.load_evaluation_results(path, [INSTANCE])

    assert results == {INSTANCE: True}
    assert json.loads(content) == {INSTANCE: True}

    path.write_text(json.dumps({INSTANCE: 1}), encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="invalid"):
        benchmark.load_evaluation_results(path, [INSTANCE])

    path.write_text(json.dumps({"example__other-1": False}), encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="do not cover"):
        benchmark.load_evaluation_results(path, [INSTANCE])


def test_inference_dry_run_resolves_defaults_without_docker_or_api(tmp_path, capsys):
    class Client:
        def select(self, *_args, **_kwargs):
            return [benchmark.parse_swebench_pro_row(make_raw_row())]

    options = make_options(tmp_path, dry_run=True, source_identity=None)
    benchmark.run_inference(options, client=Client())
    output = json.loads(capsys.readouterr().out)

    assert output["instanceIds"] == [INSTANCE]
    assert output["datasetRevision"] == benchmark.DATASET_REVISION
    assert output["inferenceWorkers"] == 1
    assert output["attemptsPerInstance"] == 1
    assert output["maxInfrastructureRetries"] == 0
    assert output["agentTimeoutSeconds"] == 1800
    assert output["delegationMode"] == "native"
    assert output["coordinatorBudget"] == 24
    assert output["nativeSubagentBudget"] == 13
    assert output["nativeSubagentCount"] == 3
    assert output["nativeSubagentTotalBudget"] == 39
    assert output["images"] == [benchmark.official_image(DOCKERHUB_TAG)]


def test_single_agent_dry_run_reports_disabled_delegation(tmp_path, capsys):
    class Client:
        def select(self, *_args, **_kwargs):
            return [benchmark.parse_swebench_pro_row(make_raw_row())]

    options = make_options(
        tmp_path,
        dry_run=True,
        source_identity=None,
        agent_topology=shared.SINGLE_AGENT_TOPOLOGY,
    )
    benchmark.run_inference(options, client=Client())
    output = json.loads(capsys.readouterr().out)

    assert output["agentTopology"] == "single-agent"
    assert output["benchmark"] == "swe-bench-pro"
    assert output["primaryAgentRole"] == "agent"
    assert output["delegationEnabled"] is False
    assert output["sequence"] == ["agent"]
    assert output["delegationMode"] == "disabled"
    assert output["agentBudget"] == 24
    assert "nativeSubagentBudget" not in output


def test_inference_dispatches_pro_worker_and_app_capture(tmp_path, monkeypatch):
    class Client:
        def select(self, *_args, **_kwargs):
            return [benchmark.parse_swebench_pro_row(make_raw_row())]

    observed = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(shared, "require_docker", lambda: {"serverVersion": "test"})
    monkeypatch.setattr(shared, "ensure_image", lambda *args: {"image": args[0]})
    monkeypatch.setattr(shared, "hermes_source_identity", lambda: SOURCE_IDENTITY)
    monkeypatch.setattr(shared, "require_unchanged_source", lambda _options: None)

    def run_worker(_options, _row, _instance_dir, _image, _metadata, **kwargs):
        observed.update(kwargs)
        return {
            "modelPatch": "diff --git a/a b/a",
            "workflowComplete": True,
            "timedOut": False,
            "changedPaths": ["a"],
        }

    monkeypatch.setattr(shared, "_run_worker", run_worker)
    paths = benchmark.run_inference(make_options(tmp_path), client=Client())

    assert observed["benchmark"] == benchmark.BENCHMARK
    assert observed["worker_module"] == "hermes_cli.benchmarks.swebench_pro_worker"
    assert observed["worktree"] == "/app"
    assert observed["evaluation_timeout_seconds"] == 3600
    assert observed["trace_run"] is None
    assert observed["agent_topology"] == shared.DEFAULT_AGENT_TOPOLOGY
    prompt = observed["prompt_builder"](
        benchmark.parse_swebench_pro_row(make_raw_row()), False
    )
    assert "navigator, patcher, reviewer" in prompt
    predictions = json.loads(paths.predictions.read_text(encoding="utf-8"))
    assert predictions[0]["patch"] == "diff --git a/a b/a"


def test_evaluation_verifies_artifact_before_fetching_private_rows(
    tmp_path, monkeypatch
):
    options, paths, _row, _prediction = write_complete_artifact(tmp_path)
    paths.predictions.write_text("[]\n", encoding="utf-8")
    touched = False

    class ForbiddenClient:
        def __init__(self):
            nonlocal touched
            touched = True

    monkeypatch.setattr(benchmark, "DatasetRowsClient", ForbiddenClient)
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(tmp_path),
        predictions_path=None,
        manifest_path=None,
        instance_id=[],
        docker_platform="linux/amd64",
        harness_dir=None,
        python="python",
        dockerhub_username="jefzda",
        block_network=False,
        redo=False,
        dry_run=True,
    )

    with pytest.raises(benchmark.BenchmarkError, match="changed after inference"):
        benchmark.run_evaluation(args)
    assert touched is False


def test_evaluation_dry_run_materializes_only_selected_evaluator_rows(
    tmp_path, monkeypatch, capsys
):
    options, paths, _row, _prediction = write_complete_artifact(tmp_path)
    harness = make_harness(tmp_path / "harness")
    monkeypatch.setattr(benchmark, "ensure_pinned_harness", lambda _path: harness)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def evaluator_rows(self, instance_ids):
            assert instance_ids == [INSTANCE]
            return [make_raw_row()]

    monkeypatch.setattr(benchmark, "DatasetRowsClient", Client)
    monkeypatch.setattr(
        benchmark,
        "evaluator_runtime_metadata",
        lambda *_args: EVALUATOR_RUNTIME,
    )
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(tmp_path),
        predictions_path=None,
        manifest_path=None,
        instance_id=[INSTANCE],
        docker_platform="linux/amd64",
        harness_dir=None,
        python="/eval/bin/python",
        dockerhub_username="jefzda",
        block_network=False,
        redo=False,
        dry_run=True,
    )
    benchmark.run_evaluation(args)
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "dry-run"
    assert output["maxWorkers"] == 1
    assert output["useLocalDocker"] is True
    assert output["datasetRevision"] == benchmark.DATASET_REVISION
    assert output["evaluationProcessTimeoutSeconds"] == 4200
    assert "--use_local_docker" in output["command"]
    evaluator_row = json.loads(
        Path(output["evaluationInstancesPath"]).read_text(encoding="utf-8")
    )
    assert set(evaluator_row) == {
        "repo",
        "instance_id",
        "base_commit",
        "before_repo_set_cmd",
        "selected_test_files_to_run",
        "fail_to_pass",
        "pass_to_pass",
    }
    assert "SECRET GOLD PATCH" not in json.dumps(evaluator_row)
    assert "SECRET HIDDEN TEST" not in json.dumps(evaluator_row)
    assert not paths.instances.exists()


@pytest.mark.parametrize("write_results", [True, False])
def test_evaluation_only_completes_with_valid_official_results(
    tmp_path, monkeypatch, write_results
):
    options, paths, _row, _prediction = write_complete_artifact(tmp_path)
    harness = make_harness(tmp_path / "harness")
    monkeypatch.setattr(benchmark, "ensure_pinned_harness", lambda _path: harness)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def evaluator_rows(self, instance_ids):
            assert instance_ids == [INSTANCE]
            return [make_raw_row()]

    monkeypatch.setattr(benchmark, "DatasetRowsClient", Client)
    monkeypatch.setattr(shared, "require_docker", lambda **_kwargs: {})
    monkeypatch.setattr(benchmark, "_require_evaluator_python", lambda *_args: None)
    monkeypatch.setattr(
        benchmark,
        "evaluator_runtime_metadata",
        lambda *_args: EVALUATOR_RUNTIME,
    )
    evaluator_kwargs = {}

    def run_evaluator(args, **kwargs):
        is_evaluator = any(str(arg).endswith("swe_bench_pro_eval.py") for arg in args)
        if is_evaluator:
            evaluator_kwargs.update(kwargs)
        if write_results and is_evaluator:
            output_dir = next(
                Path(arg.removeprefix("--output_dir="))
                for arg in args
                if str(arg).startswith("--output_dir=")
            )
            output = output_dir / "eval_results.json"
            output.write_text(json.dumps({INSTANCE: True}), encoding="utf-8")
        stdout = "evaluation output" if is_evaluator else ""
        return shared.CommandResult(tuple(args), 0, stdout, "")

    monkeypatch.setattr(shared, "run_command", run_evaluator)
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(tmp_path),
        predictions_path=None,
        manifest_path=None,
        instance_id=[],
        docker_platform="linux/amd64",
        harness_dir=None,
        python="/eval/bin/python",
        dockerhub_username="jefzda",
        block_network=False,
        redo=False,
        dry_run=False,
    )

    if write_results:
        benchmark.run_evaluation(args)
    else:
        with pytest.raises(benchmark.BenchmarkError, match="Could not read official"):
            benchmark.run_evaluation(args)

    manifest = json.loads(
        (paths.run_dir / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == ("completed" if write_results else "failed")
    assert evaluator_kwargs["timeout"] == 4200
    assert evaluator_kwargs["umask"] == 0o077
    if write_results:
        assert manifest["resolvedInstanceIds"] == [INSTANCE]
        assert manifest["unresolvedInstanceIds"] == []
        assert manifest["evaluationResultsSha256"]
        marker = Path(manifest["evaluationOutput"]) / ".hermes-cache.json"
        assert (
            json.loads(marker.read_text(encoding="utf-8"))["cacheKey"]
            == manifest["evaluationCacheKey"]
        )


def test_evaluation_timeout_is_finalized_and_cleaned(tmp_path, monkeypatch):
    options, paths, _row, _prediction = write_complete_artifact(tmp_path)
    harness = make_harness(tmp_path / "harness")
    monkeypatch.setattr(benchmark, "ensure_pinned_harness", lambda _path: harness)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def evaluator_rows(self, _instance_ids):
            return [make_raw_row()]

    monkeypatch.setattr(benchmark, "DatasetRowsClient", Client)
    monkeypatch.setattr(shared, "require_docker", lambda **_kwargs: {})
    monkeypatch.setattr(benchmark, "_require_evaluator_python", lambda *_args: None)
    monkeypatch.setattr(
        benchmark,
        "evaluator_runtime_metadata",
        lambda *_args: EVALUATOR_RUNTIME,
    )
    cleanup_calls = 0

    def run_evaluator(args, **_kwargs):
        nonlocal cleanup_calls
        if args[:3] == ["docker", "ps", "-aq"]:
            cleanup_calls += 1
            return shared.CommandResult(tuple(args), 0, "", "")
        raise benchmark.BenchmarkError("Command timed out: evaluator")

    monkeypatch.setattr(shared, "run_command", run_evaluator)
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(tmp_path),
        predictions_path=None,
        manifest_path=None,
        instance_id=[],
        docker_platform="linux/amd64",
        harness_dir=None,
        python="/eval/bin/python",
        dockerhub_username="jefzda",
        block_network=False,
        redo=False,
        dry_run=False,
    )

    with pytest.raises(benchmark.BenchmarkError, match="timed out"):
        benchmark.run_evaluation(args)

    manifest = json.loads(
        (paths.run_dir / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert "timed out" in manifest["error"]
    assert cleanup_calls == 2
