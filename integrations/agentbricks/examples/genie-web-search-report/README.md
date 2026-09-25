# Mason Genie web search report demo

This DF1 demo uses the latest Mason codebase, now named the Agent Bricks CLI (`ab`) and AgentKit, to
deploy one OpenAI Agents SDK orchestrator. One invocation:

1. reads a fixed Slack thread through request-user `system.ai.slack`;
2. searches a governed documentation-asset table through request-user native `genie-agent`;
3. validates the selected topics through request-user `system.ai.web_search`; and
4. writes a cited Markdown report and evidence JSON through request-user `system.ai.sandbox`.

The App service principal is deliberately denied table and Volume data privileges. A successful
Genie query and sandbox write therefore exercise the invoking user's OBO identity, not an App
fallback.

## Fixed input and output

The source thread is:

`https://databricks.slack.com/archives/C088VN8U4E5/p1790353973998359?thread_ts=1789056476.567349&cid=C088VN8U4E5`

Setup creates a unique schema named
`supervisor_agent.mason_genie_web_search_demo_<8-hex-suffix>` with:

- table `web_search_assets`;
- managed Volume `reports`;
- report `databricks_web_search_report.md` in that Volume;
- evidence file `evidence.json` in that Volume;
- Genie space `Mason web search docs <suffix>` on warehouse `59d2ebcf58480621`; and
- App deployment `agent-bricks-mason-genie-report-<suffix>`.

`.demo-state/setup-state.json` records every exact name and ID. The file is private (`0600`) and
gitignored.

## Prerequisites

- Databricks CLI profile `df1` authenticated as `jun.choi@databricks.com`.
- Access to catalog `supervisor_agent` and warehouse `59d2ebcf58480621`.
- Access to managed MCP services `system.ai.slack`, `system.ai.web_search`, and
  `system.ai.sandbox`.
- Permission to create a Genie space and Databricks App.
- User consent for App scopes `ai-gateway`, `genie`, and `files`. If those scopes change, sign out
  and re-consent before invoking.

The checked-in `pyproject.toml` pins `databricks-agentbricks` to the exact upstream latest-main
commit used to build this demo. No credential is checked in.

## Install and test

From this directory:

```bash
UV_CACHE_DIR=/tmp/agentbricks-demo-uv-cache uv sync
.venv/bin/pytest -v
../../.venv/bin/ruff check --isolated agent runtime scripts tests
../../.venv/bin/ruff format --check --isolated agent runtime scripts tests
```

For local `ab dev`, copy `.env.example` to `.env`, replace `REPORT_SCHEMA_NAME` after setup, and run
the CLI from `integrations/agentbricks/.venv/bin/ab`.

## Provision, deploy, and invoke

Run in order:

```bash
.venv/bin/python scripts/setup_df1.py
.venv/bin/python scripts/deploy_df1.py
.venv/bin/python scripts/invoke_e2e.py
```

`scripts/setup_df1.py` creates and seeds the schema, table, Volume, and Genie space, saving state
after every mutation. A second call reuses matching state rather than creating duplicates.

`scripts/deploy_df1.py` renders a clean source copy under `.demo-state/`, inserts the recorded Genie
space and Volume into `agent.toml`, adds `REPORT_SCHEMA_NAME` to `app.yaml`, and deploys with
`--allow-user-scope-update`. It fails unless effective scopes include `ai-gateway`, `genie`, and
`files`, user-token forwarding is enabled, and the App service principal has neither `SELECT` nor
Volume read/write privileges.

`scripts/invoke_e2e.py` makes one foreground `/api/invocations` call as the current user. It then
downloads both Volume artifacts and checks the fixed Slack timestamp, Genie conversation/message
IDs, all three evidence kinds, official citation hosts, evidence references, App scopes, and
negative App-principal grants. Success is recorded at `.demo-state/result.json`.

The workflow intentionally leaves the successful resources deployed for the demo. Inspect:

```bash
python -m json.tool .demo-state/setup-state.json
python -m json.tool .demo-state/result.json
integrations/agentbricks/.venv/bin/ab --profile df1 deployments get \
  "$(python -c 'import json; print(json.load(open(".demo-state/setup-state.json"))["deployment"]["app_name"])')"
```

Token-free command evidence is under `.demo-state/evidence/`. The report and evidence Volume paths
are the `report_path` and `evidence_path` fields in setup state.

## Targeted teardown

Cleanup is explicit:

```bash
.venv/bin/python scripts/teardown_df1.py
```

`scripts/teardown_df1.py` reads exact IDs from setup state and removes only the recorded App, Genie
space, and demo schema. It never discovers by prefix, and it never deletes the catalog or warehouse.

## Security and evidence policy

All four managed tool entries have `auth = "user"`. Sandbox is downscoped to one reports Volume
with `read_write`; it receives sanitized evidence and report text, not connector credentials.

The orchestrator treats Slack as an internal field report and Genie as a governed asset catalog.
Only HTTPS citations under `docs.databricks.com` or Azure Databricks pages under
`learn.microsoft.com` are accepted as official documentation. The agent post-filters returned URLs
because `allowed_domains` is not treated as a security boundary. Genie polling must reuse the
returned `conversation_id` and `message_id`; an indeterminate submission is never blindly
resubmitted.
