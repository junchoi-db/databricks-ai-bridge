# Mason Genie Web Search Report Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, deploy, and validate a DF1 OpenAI Agents SDK app that uses OBO Slack, native Genie Agent, web search, and sandbox tools to create a cited report in a Unity Catalog Volume.

**Architecture:** A current `ab init --framework openai` managed-server scaffold hosts one orchestrator. Managed bindings provide all four request-user tools; focused Python modules enforce fixed source/output contracts and citation policy, while lifecycle scripts provision governed DF1 resources, deploy the App, invoke it, and verify the artifact.

**Tech Stack:** Python 3.11+, Agent Bricks CLI/AgentKit from repository `main`, OpenAI Agents SDK, Databricks SDK, pytest, Unity Catalog, Genie Chat API, managed MCP services, Databricks Apps.

**Spec:** `docs/superpowers/specs/2026-09-25-mason-genie-web-search-report-demo-design.md`

## Global Constraints

- Base the demo on upstream `main` commit `81c427266af394216f327fb5c0159db332f1ae64` or a newer fetched `origin/main` that retains request-user Genie support.
- Use one OpenAI Agents SDK orchestrator; do not add agent fan-out.
- Configure `system.ai.slack`, native `genie_agent`, `system.ai.web_search`, and `system.ai.sandbox` with explicit `auth = "user"`.
- Limit sandbox to the single reports Volume with `permission = "read_write"`.
- Accept official citations only from `docs.databricks.com` or Azure Databricks pages under `learn.microsoft.com`.
- Reuse Genie `conversation_id` and `message_id` for polling; never blindly resubmit an indeterminate request.
- Never grant the App service principal `SELECT`, `READ_VOLUME`, or `WRITE_VOLUME` on demo data.
- Never persist forwarded tokens, credentials, or credential-derived headers.
- Leave a successful DF1 deployment running; cleanup happens only through the targeted teardown command.

---

### Task 1: Scaffold the Current OpenAI Agent Project

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/agent.toml`
- Create: `integrations/agentbricks/examples/genie-web-search-report/pyproject.toml`
- Create: `integrations/agentbricks/examples/genie-web-search-report/app.yaml`
- Create: `integrations/agentbricks/examples/genie-web-search-report/agent/agent.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/runtime/adapter.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/runtime/main.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_manifest.py`

**Interfaces:**
- Consumes: current `agent-openai` template from `integrations/agentbricks/src/databricks_agentbricks/templates/agent-openai/`.
- Produces: importable package `agent`, managed `agentbricks` runtime, and a manifest with explicit sentinel space/Volume values replaced by deployment.

- [ ] **Step 1: Generate the current scaffold**

Run from `integrations/agentbricks`:

```bash
python -m venv .venv-demo-cli
.venv-demo-cli/bin/pip install -e .
.venv-demo-cli/bin/ab init examples/genie-web-search-report --framework openai --no-interactive
```

Expected: a managed OpenAI project with `[agent].server = "agentbricks"`.

- [ ] **Step 2: Write the failing manifest test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_all_managed_tools_use_request_user_auth():
    manifest = tomllib.loads((ROOT / "agent.toml").read_text())
    tools = {item["id"]: item for item in manifest["tools"]}
    assert set(tools) == {"slack", "genie_assets", "web_search", "report_sandbox"}
    assert {item["auth"] for item in tools.values()} == {"user"}
    assert tools["slack"]["source"] == {"kind": "mcp", "service": "system.ai.slack"}
    assert tools["web_search"]["source"] == {
        "kind": "mcp",
        "service": "system.ai.web_search",
    }
    assert tools["genie_assets"]["source"]["kind"] == "genie_agent"
    assert tools["report_sandbox"]["policy"]["downscope"][0]["permission"] == "read_write"
```

- [ ] **Step 3: Run the test and observe the missing bindings**

Run: `cd integrations/agentbricks/examples/genie-web-search-report && uv run pytest tests/test_manifest.py -v`

Expected: FAIL because the scaffold does not yet have the four bindings.

- [ ] **Step 4: Add the explicit managed bindings**

Use explicit non-deployable sentinel values that deploy replaces deterministically:

```toml
[[tools]]
id = "slack"
auth = "user"
source = { kind = "mcp", service = "system.ai.slack" }

[[tools]]
id = "genie_assets"
auth = "user"
source = { kind = "genie_agent", space_id = "00000000000000000000000000000000" }

[[tools]]
id = "web_search"
auth = "user"
source = { kind = "mcp", service = "system.ai.web_search" }

[[tools]]
id = "report_sandbox"
auth = "user"
source = { kind = "sandbox", service = "system.ai.sandbox" }
policy = { downscope = [{ resource = "volume:supervisor_agent.not_deployed.reports", permission = "read_write" }] }
```

- [ ] **Step 5: Run the manifest test**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the scaffold**

```bash
git add integrations/agentbricks/examples/genie-web-search-report
git commit -m "Add Genie report demo scaffold"
```

### Task 2: Implement the Report and Citation Contract

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/agent/report_contract.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_report_contract.py`

**Interfaces:**
- Consumes: fixed Slack URL and Volume paths injected through `REPORT_SCHEMA_NAME`.
- Produces: `EvidenceItem`, `ReportResult`, `canonical_official_url(url)`, `validate_evidence(items)`, and `volume_paths(schema_name)`.

- [ ] **Step 1: Write failing URL and path tests**

```python
import pytest
from agent.report_contract import canonical_official_url, volume_paths


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://docs.databricks.com/aws/en/generative-ai/mcp/", "https://docs.databricks.com/aws/en/generative-ai/mcp"),
        ("https://sub.docs.databricks.com/page", "https://sub.docs.databricks.com/page"),
        ("https://learn.microsoft.com/en-us/azure/databricks/generative-ai/", "https://learn.microsoft.com/en-us/azure/databricks/generative-ai"),
    ],
)
def test_accepts_official_documentation(url, expected):
    assert canonical_official_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.databricks.com/page",
        "https://docs.databricks.com.evil.example/page",
        "https://learn.microsoft.com/en-us/windows/page",
        "https://user:password@docs.databricks.com/page",
    ],
)
def test_rejects_non_official_or_unsafe_urls(url):
    with pytest.raises(ValueError):
        canonical_official_url(url)


def test_volume_paths_are_fixed_to_demo_schema():
    paths = volume_paths("mason_genie_web_search_demo_ab12cd34")
    assert paths.report.endswith("/reports/databricks_web_search_report.md")
    assert paths.evidence.endswith("/reports/evidence.json")
```

- [ ] **Step 2: Run tests to verify import failure**

Run: `uv run pytest tests/test_report_contract.py -v`

Expected: FAIL with `ModuleNotFoundError: agent.report_contract`.

- [ ] **Step 3: Implement typed contracts and validation**

Implement frozen dataclasses with JSON-safe `to_dict()` methods:

```python
@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_kind: Literal["slack", "genie_asset", "official_web"]
    title: str
    canonical_uri: str
    excerpt: str
    retrieved_at: str
    source_metadata: dict[str, str]
    authority: Literal[
        "internal_field_report", "governed_asset_catalog", "official_documentation"
    ]


@dataclass(frozen=True)
class VolumePaths:
    report: str
    evidence: str
```

`canonical_official_url()` must parse with `urllib.parse.urlsplit`, require HTTPS/no user info,
allow exact `docs.databricks.com` or its subdomains, and allow `learn.microsoft.com` only when the
normalized path begins `/en-us/azure/databricks/`.

- [ ] **Step 4: Run contract tests**

Run: `uv run pytest tests/test_report_contract.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add integrations/agentbricks/examples/genie-web-search-report/agent/report_contract.py integrations/agentbricks/examples/genie-web-search-report/tests/test_report_contract.py
git commit -m "Add report evidence contract"
```

### Task 3: Add Deterministic Evidence and Report Tools

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/agent/tools/report_tools.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_report_tools.py`

**Interfaces:**
- Consumes: `EvidenceItem`, official URL validator, and JSON strings returned by managed tools.
- Produces: OpenAI `@function_tool` functions `record_slack_evidence`, `record_genie_assets`, `record_web_evidence`, and `validate_report`; run-local `EvidenceLedger` keyed by `contextvars.ContextVar`.

- [ ] **Step 1: Write failing evidence-ledger tests**

```python
from agent.tools.report_tools import EvidenceLedger


def test_web_evidence_separates_rejected_domains():
    ledger = EvidenceLedger(run_id="run-1")
    accepted, rejected = ledger.add_web_results(
        [
            {"title": "Official", "url": "https://docs.databricks.com/aws/en/generative-ai/mcp", "excerpt": "Managed MCP"},
            {"title": "Other", "url": "https://example.com/post", "excerpt": "Unofficial"},
        ]
    )
    assert [item.evidence_id for item in accepted] == ["web-001"]
    assert rejected == [{"title": "Other", "url": "https://example.com/post", "reason": "unapproved host"}]


def test_report_rejects_unknown_evidence_ids():
    ledger = EvidenceLedger(run_id="run-1")
    errors = ledger.validate_report("Finding [web-999]\n\n## Sources\n")
    assert errors == ["unknown evidence id: web-999"]
```

- [ ] **Step 2: Run tests to verify missing implementation**

Run: `uv run pytest tests/test_report_tools.py -v`

Expected: FAIL on missing `EvidenceLedger`.

- [ ] **Step 3: Implement ledger behavior and function tools**

Use deterministic prefixes/counters, bound excerpts to 2,000 characters, strip token-like fields
(`token`, `authorization`, `cookie`, and `headers`) from metadata, and require these report sections:

```python
REQUIRED_SECTIONS = (
    "## Executive summary",
    "## Internal field report",
    "## Cataloged assets",
    "## Official documentation findings",
    "## Discrepancies and limitations",
    "## Sources",
)
```

The decorated functions return JSON containing accepted IDs, rejected results, and validation
errors; they do not write files or invoke external services.

- [ ] **Step 4: Run evidence tests**

Run: `uv run pytest tests/test_report_tools.py -v`

Expected: PASS.

- [ ] **Step 5: Commit evidence tools**

```bash
git add integrations/agentbricks/examples/genie-web-search-report/agent/tools/report_tools.py integrations/agentbricks/examples/genie-web-search-report/tests/test_report_tools.py
git commit -m "Add governed report evidence tools"
```

### Task 4: Wire the Single Orchestrator and Safe Tool Protocol

**Files:**
- Modify: `integrations/agentbricks/examples/genie-web-search-report/agent/agent.py`
- Modify: `integrations/agentbricks/examples/genie-web-search-report/agent/tools/__init__.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_agent_protocol.py`

**Interfaces:**
- Consumes: managed tools from `mcp_tools()` and `genie_tools()`, local report tools, and template `run_agent(input, context)` contract.
- Produces: one configured `Agent`, strict developer instructions, and terminal `ReportResult` JSON.

- [ ] **Step 1: Write failing prompt/protocol tests**

```python
from agent.agent import INSTRUCTIONS, SLACK_THREAD_URL


def test_prompt_fixes_source_order_and_slack_thread():
    assert SLACK_THREAD_URL in INSTRUCTIONS
    assert INSTRUCTIONS.index("Slack") < INSTRUCTIONS.index("Genie") < INSTRUCTIONS.index("web search")


def test_prompt_forbids_blind_genie_resubmission():
    assert "same conversation_id and message_id" in INSTRUCTIONS
    assert "never resubmit" in INSTRUCTIONS.lower()


def test_prompt_requires_sandbox_artifacts():
    assert "databricks_web_search_report.md" in INSTRUCTIONS
    assert "evidence.json" in INSTRUCTIONS
```

- [ ] **Step 2: Run tests and observe prompt mismatch**

Run: `uv run pytest tests/test_agent_protocol.py -v`

Expected: FAIL because the scaffold has generic instructions.

- [ ] **Step 3: Implement tool composition and instructions**

Compose:

```python
tools = [
    *await mcp_tools(workspace_client_for=workspace_client_for),
    *genie_tools(workspace_client_for=workspace_client_for),
    *all_tools(),
]
```

The instructions must name the actual managed tool IDs, prescribe Slack → Genie ask/poll/query
result → web search → ledger validation → sandbox write, require `allowed_domains` when exposed by
the web tool, and require post-filtering through `record_web_evidence`. They must tell sandbox to
write exact UTF-8 Markdown/JSON, then read back both paths and check the run ID.

- [ ] **Step 4: Add a mocked tool-trace test**

Create a fake trace with an in-progress `genie_assets_ask` response, then assert the next Genie call
is `genie_assets_poll` with the returned IDs and that no second ask exists:

```python
assert genie_calls == [
    ("genie_assets_ask", {"question": ANY}),
    ("genie_assets_poll", {"conversation_id": "c-1", "message_id": "m-1"}),
]
```

- [ ] **Step 5: Run all local app tests**

Run: `uv run pytest -v`

Expected: PASS.

- [ ] **Step 6: Commit orchestrator wiring**

```bash
git add integrations/agentbricks/examples/genie-web-search-report/agent integrations/agentbricks/examples/genie-web-search-report/tests
git commit -m "Wire Genie report orchestrator"
```

### Task 5: Implement Idempotent DF1 Setup

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/scripts/common.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/scripts/setup_df1.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_setup_df1.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/.gitignore`

**Interfaces:**
- Consumes: Databricks profile `df1`, warehouse `59d2ebcf58480621`, catalog `supervisor_agent`.
- Produces: `.demo-state/setup-state.json` with schema/table/Volume/paths/space IDs and token-free evidence under `.demo-state/evidence/`.

- [ ] **Step 1: Write failing pure lifecycle tests**

```python
from scripts.setup_df1 import asset_rows, build_serialized_space, resource_names


def test_resource_names_share_one_suffix():
    names = resource_names("ab12cd34")
    assert names.schema == "supervisor_agent.mason_genie_web_search_demo_ab12cd34"
    assert names.table == names.schema + ".web_search_assets"
    assert names.volume == names.schema + ".reports"


def test_seed_assets_are_official_and_cover_required_topics():
    rows = asset_rows()
    assert {row["topic"] for row in rows} >= {
        "web_search", "managed_mcp", "obo", "audit", "apps_oauth"
    }
    assert all(row["url"].startswith("https://docs.databricks.com/") for row in rows)


def test_genie_space_uses_only_asset_table():
    payload = build_serialized_space("supervisor_agent.schema.web_search_assets")
    assert payload["data_sources"] == {
        "tables": [{"identifier": "supervisor_agent.schema.web_search_assets"}]
    }
```

- [ ] **Step 2: Run tests and observe missing setup module**

Run: `uv run pytest tests/test_setup_df1.py -v`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement state-safe provisioning**

Implement `StateStore.save()` with a temporary file plus `Path.replace()`, state mode `0600`, and
save after every created resource. Use SDK schema/Volume/Genie APIs and Statement Execution for:

```sql
CREATE TABLE <table> (
  asset_id STRING, topic STRING, title STRING, url STRING,
  description STRING, keywords ARRAY<STRING>
) USING DELTA
```

Insert fixed rows using parameterized SQL or safely JSON-escaped literals. Confirm the current user
is `jun.choi@databricks.com`. Refuse to overwrite a state file belonging to a different workspace or
principal.

- [ ] **Step 4: Run setup tests**

Run: `uv run pytest tests/test_setup_df1.py -v`

Expected: PASS.

- [ ] **Step 5: Commit setup lifecycle**

```bash
git add integrations/agentbricks/examples/genie-web-search-report/scripts integrations/agentbricks/examples/genie-web-search-report/tests/test_setup_df1.py integrations/agentbricks/examples/genie-web-search-report/.gitignore
git commit -m "Add DF1 Genie report setup"
```

### Task 6: Implement Deploy, Invoke/Verify, and Targeted Teardown

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/scripts/deploy_df1.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/scripts/invoke_e2e.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/scripts/teardown_df1.py`
- Create: `integrations/agentbricks/examples/genie-web-search-report/tests/test_lifecycle.py`

**Interfaces:**
- Consumes: `setup-state.json`, current Git commit, `ab` CLI, Databricks SDK, fixed report contract.
- Produces: deployed App state, E2E `result.json`, downloaded report/evidence snapshots, and explicit teardown.

- [ ] **Step 1: Write failing manifest-render and verification tests**

```python
from scripts.deploy_df1 import render_manifest
from scripts.invoke_e2e import verify_report


def test_render_manifest_injects_exact_space_and_volume(setup_state):
    manifest = render_manifest(setup_state)
    assert setup_state["space_id"] in manifest
    assert f'volume:{setup_state["volume"]}' in manifest
    assert manifest.count('auth = "user"') == 4


def test_verify_report_requires_all_source_kinds(valid_report, valid_evidence):
    broken = [item for item in valid_evidence if item["source_kind"] != "slack"]
    result = verify_report(valid_report, broken, run_id="run-1")
    assert result.passed is False
    assert "missing source kind: slack" in result.errors
```

- [ ] **Step 2: Run lifecycle tests and observe missing modules**

Run: `uv run pytest tests/test_lifecycle.py -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement deployment**

Render only the two dynamic manifest values, verify the four bindings afterward with `tomllib`, run
`ab --profile df1 deploy <app-name> --source . --allow-user-scope-update`, then poll the App until
its deployment is `SUCCEEDED` or terminal failure. Record effective scopes and assert
`{"ai-gateway", "genie", "files"}` is a subset.

Grant App `CAN_USE` to `jun.choi@databricks.com`; grant only schema resolution and Genie `CAN_READ`
to the App service principal when required. Query effective table/Volume grants and fail if the App
principal has data privileges.

- [ ] **Step 4: Implement invocation and semantic verification**

Invoke `/api/invocations` with a UUID run/invocation ID and the exact report request. Poll status and
events with a bounded 20-minute deadline. Emit a one-line timestamped heartbeat no less often than
every 60 seconds. On failure, capture invocation events and `ab deployments logs`.

Download both Volume files through the Files API and verify required sections, run ID, source kinds,
Genie identifiers, Slack timestamps, accepted host policy, no unknown evidence IDs, App scopes, and
negative App-principal data grants. Write `.demo-state/result.json` with no secrets.

- [ ] **Step 5: Implement teardown**

Require the recorded workspace host and current principal to match. Delete only the exact recorded
App, Genie space, and schema, in that order; tolerate already-absent targets and update state after
each deletion. Do not discover by prefix and do not delete the catalog or warehouse.

- [ ] **Step 6: Run lifecycle tests**

Run: `uv run pytest tests/test_lifecycle.py -v`

Expected: PASS.

- [ ] **Step 7: Commit lifecycle scripts**

```bash
git add integrations/agentbricks/examples/genie-web-search-report/scripts integrations/agentbricks/examples/genie-web-search-report/tests/test_lifecycle.py
git commit -m "Add Genie report deploy and E2E lifecycle"
```

### Task 7: Document and Verify the Local Demo

**Files:**
- Create: `integrations/agentbricks/examples/genie-web-search-report/README.md`
- Create: `integrations/agentbricks/examples/genie-web-search-report/.env.example`
- Modify: `integrations/agentbricks/examples/genie-web-search-report/tests/test_manifest.py`

**Interfaces:**
- Consumes: all app and lifecycle commands.
- Produces: copy-paste setup/deploy/invoke/inspect/teardown instructions and a locally verified demo.

- [ ] **Step 1: Add documentation assertions**

Assert README contains `setup_df1.py`, `deploy_df1.py`, `invoke_e2e.py`, `teardown_df1.py`, all four
tool names, the exact Slack URL, the three required OAuth scopes, and the warning that successful
E2E leaves resources deployed.

- [ ] **Step 2: Run the assertion and observe missing README content**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: FAIL on missing documentation requirements.

- [ ] **Step 3: Write the operator README and environment example**

Document:

```bash
uv sync --all-extras
uv run python scripts/setup_df1.py
uv run python scripts/deploy_df1.py
uv run python scripts/invoke_e2e.py
uv run python scripts/teardown_df1.py  # explicit cleanup only
```

Include how to inspect `.demo-state/setup-state.json`, `.demo-state/result.json`, deployment logs,
and both Volume paths. Explain current `ab` naming, OBO consent/re-consent, and least-privilege
negative grant checks.

- [ ] **Step 4: Run format, lint, type, and test gates**

Run from the example:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -v
```

Run relevant Agent Bricks tests from `integrations/agentbricks`:

```bash
uv run pytest tests/unit_tests/genie_adapters_test.py tests/unit_tests/deploy_auth_test.py tests/unit_tests/runtime_app_test.py -v
```

Expected: all commands PASS.

- [ ] **Step 5: Commit documentation**

```bash
git add integrations/agentbricks/examples/genie-web-search-report
git commit -m "Document Genie web search report demo"
```

### Task 8: Provision DF1, Deploy, and Iterate to a Passing Invocation

**Files:**
- Create at runtime: `integrations/agentbricks/examples/genie-web-search-report/.demo-state/setup-state.json`
- Create at runtime: `integrations/agentbricks/examples/genie-web-search-report/.demo-state/result.json`
- Create at runtime: `integrations/agentbricks/examples/genie-web-search-report/.demo-state/evidence/*`
- Modify as diagnosed: app/config/lifecycle files from Tasks 1–7.

**Interfaces:**
- Consumes: verified local implementation and DF1 credentials/profile.
- Produces: one live DF1 App and governed resources with a fully passing deployed invocation.

- [ ] **Step 1: Run setup and inspect recorded resources**

Run: `uv run python scripts/setup_df1.py`

Expected: schema, table, Volume, paths, and a 32-character Genie space ID are recorded; seeded table
query succeeds.

- [ ] **Step 2: Deploy and actively monitor**

Run: `uv run python scripts/deploy_df1.py`

Poll once per minute until success or terminal failure. On failure, collect build/runtime logs and
use the systematic-debugging workflow to identify the failing boundary before editing code.

- [ ] **Step 3: Invoke and verify end to end**

Run: `uv run python scripts/invoke_e2e.py`

Expected: `.demo-state/result.json` has `"passed": true`; the report and evidence files exist in the
recorded Volume; result checks prove Slack, Genie, official web, sandbox write, scopes, and negative
App-principal grants.

- [ ] **Step 4: Iterate only from captured evidence**

For each failure, record the exact failed phase and evidence, write or extend a failing local test,
make one minimal correction, run the focused test, redeploy if runtime source changed, and repeat the
same E2E invocation contract until it passes.

- [ ] **Step 5: Run final verification**

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check .
git diff --check origin/main...HEAD
git status --short
```

Verify the live App remains `RUNNING`, `.demo-state/result.json` is passing, and teardown has not
been run.

- [ ] **Step 6: Commit any evidence-driven fixes**

```bash
git add integrations/agentbricks/examples/genie-web-search-report
git commit -m "Validate Genie report demo on DF1"
```
