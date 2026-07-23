# SWE-bench Pro

Hermes runs SWE-bench Pro inside Scale's official per-instance images and
scores frozen predictions with the pinned official local-Docker evaluator.
Inference and evaluation are separate commands; inference never starts scoring
automatically.

## Parity defaults

The experiment-defining CLI defaults match the OpenCode and OpenHands
integrations in this workspace:

| Setting | Default |
| --- | --- |
| Dataset | `ScaleAI/SWE-bench_Pro` (`default`, `test`), revision `7ab5114912baf22bb098818e604c02fe7ad2c11f` |
| Smoke instance | `instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d` |
| Model | `openrouter/qwen/qwen3-coder-next` |
| Temperature | `0.1` |
| Inference/evaluation workers | `1` / `1` |
| Attempts/retries | `1` / `0` |
| Agent deadline | 1,800 seconds |
| Setup/teardown guard | 600 seconds |
| Agent sequence | coordinator → navigator → patcher → reviewer |
| Iteration budgets | Coordinator `24`; three native children at `13` each |
| Inference image prefix | `docker.io/jefzda/sweap-images` |
| Container worktree | `/app` |
| Evaluator | `scaleapi/SWE-bench_Pro-os` at `0c64e26f00b9c190432de7fc520c8ceed5c25518` |
| Evaluation mode | local Docker, username `jefzda`, network enabled |

The coordinator uses Hermes' native `delegate_task` tool for one fresh leaf
agent at a time. The one-shot worker's stateless delivery declaration makes
each native delegation return synchronously before the next role begins. All
four agents share one `/app` container.

Hermes natively exposes one delegated-child iteration cap, so each specialist
gets 13 iterations (39 combined), the closest lower-cost match to the peers'
`10` / `18` / `12` split (40 combined). Prompts preserve the navigator →
patcher → reviewer semantics and tell the navigator not to modify state. A
post-run audit checks the observed native calls, leaf roles, order, and
completion status without intercepting or mechanically enforcing orchestration.

Parity here means the same selected instances, model/temperature, one
benchmark-level attempt, zero outer retries, one worker, agent deadline,
topology/budgets, image family, and official evaluator flags. Provider clients
have different internal retry behavior, and each framework has its own
setup/teardown and worktree architecture. Hermes pins the dataset revision that
the peer configurations currently resolve instead of silently accepting later
dataset changes.

## Prerequisites

- A working local Docker daemon. Hermes runs instances serially by default and
  applies no per-container CPU, memory, or disk cap.
- `OPENROUTER_API_KEY` exported in the shell that starts inference.
- For evaluation, a dedicated Python environment providing the official
  harness requirements, including `docker`, `pandas`, and `tqdm`.

The API key is passed only to the isolated worker process. It is not accepted as
a command-line argument, written to artifacts, forwarded into the task
container, or included in the worker request. User memory, context files,
checkpoints, plugins, configured volumes, and credential mounts are disabled for
the attempt.

## Run inference

From the Hermes repository:

```bash
export OPENROUTER_API_KEY='...'
hermes-swebench-pro infer --run-id qwen-next-pro-smoke
```

Use repeatable explicit IDs to run a chosen set in that exact order:

```bash
hermes-swebench-pro infer \
  --run-id qwen-next-pro \
  --instance-id instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
```

Alternatively, select a dataset window with `--offset` and
`--max-instances`. Any explicit selection disables the implicit smoke
instance. `--dry-run` resolves the public rows, image tags, budgets, and output
paths without Docker or an API call.

Inference receives only the public fields `repo`, `instance_id`, `base_commit`,
`problem_statement`, `requirements`, `interface`, `repo_language`, and
`dockerhub_tag`. Hidden tests, gold patches, and evaluator-only fields are not
placed in prompts, requests, or inference artifacts.

The task container retains network access by default to match the peer runs.
The public-field projection therefore protects Hermes-created prompts and
artifacts; it is not an air-gap against agent code independently downloading a
public dataset.

The runner pulls the row's official
`docker.io/jefzda/sweap-images:<dockerhub_tag>` image when it is not already
available; it does not build an agent image. Each instance gets a fresh worker
subprocess and container, one outer attempt, and no benchmark-level retry. The
controller restarts the container before capturing tracked and untracked changes
from `/app`, then restores bind-mount ownership and removes the container.

Artifacts are written below:

```text
.benchmark-runs/swe-bench-pro/runs/<run-id>/
```

They include public selected rows, prompts, redacted worker logs,
native-delegation audit records, patches, Scale-format `predictions.json`, summaries, and a SHA-256-bound
completion manifest. The manifest also binds resume to the exact Hermes source
identity, pinned dataset revision, complete public-row hashes, and
configuration. Use `--restart` to replace a matching run.

## Run local evaluation

Create a dedicated evaluator environment from the pinned harness, then point
Hermes at its Python executable:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /tmp/swe-bench-pro
git -C /tmp/swe-bench-pro checkout 0c64e26f00b9c190432de7fc520c8ceed5c25518
uv venv --python .venv/bin/python \
  ~/.cache/hermes-agent/benchmarks/swe-bench-pro/evaluator-venv
~/.cache/hermes-agent/benchmarks/swe-bench-pro/evaluator-venv/bin/pip install \
  docker==7.2.0 pandas==3.0.3 tqdm==4.69.0

hermes-swebench-pro evaluate \
  --run-id qwen-next-pro-smoke \
  --python ~/.cache/hermes-agent/benchmarks/swe-bench-pro/evaluator-venv/bin/python \
  --harness-dir /tmp/swe-bench-pro
```

Those are the only optional packages imported by the pinned local-Docker path;
Modal and dataset-loader dependencies are not required for local evaluation.

When `--harness-dir` is omitted, Hermes clones and validates the exact pinned
revision under `~/.cache/hermes-agent/benchmarks/swe-bench-pro/`. Before
evaluation it requires a clean harness worktree and verifies the completed
manifest, run ID, prediction order/count, schema, and SHA-256 digest. It only
then fetches evaluator rows. Hermes verifies each row against its frozen public
hash and writes only the seven fields consumed by the harness. Gold patches,
hidden test patches, prompts, and issue metadata are excluded. The three list
fields are parsed as literals and re-encoded as canonical JSON arrays before
the pinned upstream harness reads them; this prevents its `eval()` calls from
executing downloaded expressions on the host.

Evaluation always passes `--use_local_docker`, one worker, the configured Docker
platform, and Docker Hub username `jefzda`. Supply repeatable `--instance-id`
flags to evaluate a subset already present in the prediction artifact. Optional
`--block-network` and `--redo` flags map directly to the official harness.
For untrusted submissions, prefer `--block-network`; the network-enabled
default exists for peer parity. A non-default inference `--image-prefix` and
evaluation `--dockerhub-username` should refer to byte-identical mirrors.

`--dry-run` validates and materializes the evaluator inputs, records the Python
and `docker`/`pandas`/`tqdm` versions, and prints the exact command without
starting Docker evaluation. Evaluation outputs use a private,
content-addressed cache keyed by the rows, predictions, runtime, harness,
platform, network mode, and ordered IDs, so changed inputs cannot reuse stale
scores. `--redo` forces the official harness to recompute an otherwise matching
cache entry. The evaluator process is bounded to 3,600 seconds plus a 600-second
setup allowance per instance at the fixed one-worker setting. Hermes removes
only containers mounted to that cache both before and after the evaluator.

A real evaluation is marked complete only when the freshly generated official
`eval_results.json` contains one Boolean result for every requested instance;
its digest and resolved/unresolved IDs are added to the evaluation manifest.
