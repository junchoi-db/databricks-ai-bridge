# Mason agent-tool matrix

This suite proves that CLI-generated and directly authored Databricks integration registries reach
the same runtime code. It creates two LangGraph projects (CLI/direct), runs each with `mason dev`,
deploys each to Databricks Apps, and semantically exercises four tools:

- `system.ai.google_drive` with `auth="user"`, as the request-user OBO/provider-consent fixture;
- `system.ai.sandbox` with `auth="app"`, scoped to a temporary marker table explicitly granted to
  each dedicated App service principal;
- a local Python marker tool; and
- a temporary Unity Catalog function on its existing App/default credential path.

The one framework × two authoring paths × two runtimes × four tool kinds produce 16 evidence rows.
Google Drive is not an App-auth positive fixture: the runner never grants an App service principal
the MCP Service or records such a grant in evidence metadata.

## Run

```bash
cd integrations/mason
uv build --wheel --out-dir /tmp/mason-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --freshness-marker review-head-<short-sha> \
  --wheel /tmp/mason-tooling-dist/databricks_mason-0.1.1.dev0-py3-none-any.whl \
  --output /tmp/mason-tool-matrix-df1 \
  --uc-schema aifx_benchmarks.mason_agent_tools_e2e \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.google_drive`, and permission to create a schema, table, and function. The request user
must have authorized the Google Drive provider or complete the surfaced provider-consent step. The
suite discovers and starts a SQL warehouse. Override its defaults with `--warehouse-id` or
`--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `mason init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

The runner copies the supplied wheel into every generated project under a SHA-256-addressed
`vendor/` path and adds it as that project's `databricks-mason` uv source. The Python matrix tool
checks the installed distribution's `direct_url.json` archive hash before returning, while evidence
records the expected SHA and each project source path. `--freshness-marker` is an optional live-run
value returned by that same cell. It also prints
`[freshness-check <marker>] hit matrix_marker` from the generated project so the controller can
confirm each deployed App with `databricks apps logs --search <marker>`. Pass a unique review/commit
marker for deployment proof without adding a fixed freshness string to tracked template source.

Direct authoring does not call `mason tools add`: it replaces `agent/databricks_tools.py` with
`fixtures/direct_databricks_tools.py`. CLI authoring invokes the sandbox, MCP, and UC Function
`mason tools add ...` commands with explicit auth modes. Both paths create the same ordinary,
user-owned Python tool file; local Python tools do not use the Databricks integration registry.
Every exact command and generated-file step is captured in `commands.log`.

The Google prompt calls `google_drive_list_recent(max_results=3)` and permits only a success/count
summary—never names, URLs, IDs, owners, snippets, or content. Semantic validation may inspect the
HTTP response in memory, but neither `commands.log` nor evidence `actual` stores the raw response;
successful evidence uses one fixed redacted summary. Provider URLs/query state and arbitrary HTTP
error bodies are also discarded before retry logging.

Evidence keeps authorization boundaries distinct:

- `APPS_INGRESS_AUTHORIZATION_FAILED`: Apps rejected OAuth scopes or App `CAN USE` before Mason;
- `MCP_USER_AUTHORIZATION_MISSING` / `MCP_USER_AUTHORIZATION_INVALID`: Mason did not receive a
  usable Apps-forwarded user credential;
- `MCP_PERMISSION_DENIED`: the request principal lacks downstream MCP/UC permission; and
- `MCP_AUTHORIZATION_REQUIRED`: Google Drive needs provider consent.

These rows prove the configured single-caller fixture behavior only. They do not claim two-user
isolation or audit-log attribution.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success is exactly `16 passed, 0 failed, 0 skipped`, including the expected auth/principal metadata
on every row. Temporary Apps, marker table, and UC function are deleted after a successful run.
Pass `--keep-resources` while debugging.
Dev rows correctly report the selected developer profile for both auth modes; only deploy rows for
Sandbox and UC Function claim the dedicated App service principal and its explicit grants.
