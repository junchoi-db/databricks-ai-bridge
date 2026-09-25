# Mason Genie Web Search Report Demo Design

## Summary

Build a deployable OpenAI Agents SDK demo from the latest `main` version of Mason, now shipped as
the Agent Bricks CLI (`ab`) and AgentKit. A single orchestrator reads one supplied Slack thread,
uses a request-user-authenticated native Genie Agent to find a curated catalog of Databricks web
search documentation, validates those findings with the request-user-authenticated
`system.ai.web_search` MCP service, and asks the request-user-authenticated
`system.ai.sandbox` service to write a cited Markdown report to a Unity Catalog Volume.

The demo must be repeatable in DF1: setup provisions all governed resources, deployment configures
all four tools for on-behalf-of (OBO) execution, the E2E runner invokes the deployed App as the
current user and verifies the generated artifact, and targeted teardown removes only resources
recorded by setup. The final DF1 demo resources remain deployed after successful validation unless
the operator explicitly runs teardown.

## Goals

- Demonstrate Slack, native Genie Agent, web search, and sandbox in one deployed agent invocation.
- Prove that each integration executes with the invoking user's identity rather than the App
  service principal.
- Create a useful, citation-backed report that distinguishes internal field evidence from official
  Databricks documentation.
- Make DF1 setup, deployment, invocation, verification, inspection, and cleanup reproducible.
- Preserve enough structured evidence to diagnose individual connector, authorization, polling,
  citation, or artifact failures.

## Non-goals

- Building a general-purpose deep-research framework.
- Creating a multi-agent supervisor or parallel fan-out architecture.
- Re-indexing Databricks documentation into an app-owned vector index.
- Supporting arbitrary Slack URLs, arbitrary report topics, or arbitrary output paths.
- Treating web-search domain filtering as a security boundary; returned URLs are checked again by
  the orchestrator.
- Granting the App service principal direct data access as a fallback for failed OBO.

## Fixed Inputs and DF1 Resources

The supplied Slack source is fixed:

- Channel: `C088VN8U4E5`
- Thread timestamp: `1789056476.567349`
- Highlighted reply timestamp: `1790353973.998359`
- URL: `https://databricks.slack.com/archives/C088VN8U4E5/p1790353973998359?thread_ts=1789056476.567349&cid=C088VN8U4E5`

Setup uses these base names and records any collision-avoidance suffix in `setup-state.json`:

- Profile: `df1`
- Expected setup/invoking user: `jun.choi@databricks.com`
- Catalog: `supervisor_agent`
- Schema name: `mason_genie_web_search_demo_<8-hex-suffix>`
- Schema FQN: `supervisor_agent.mason_genie_web_search_demo_<8-hex-suffix>`
- Asset table: `<schema-fqn>.web_search_assets`
- Managed Volume: `<schema-fqn>.reports`
- Report path:
  `/Volumes/supervisor_agent/mason_genie_web_search_demo_<suffix>/reports/databricks_web_search_report.md`
- Evidence path:
  `/Volumes/supervisor_agent/mason_genie_web_search_demo_<suffix>/reports/evidence.json`
- Genie space title: `Mason web search docs <suffix>`
- SQL warehouse: `59d2ebcf58480621`
- Databricks App: `mason-genie-report-<suffix>`

The asset table contains a small, curated set of official-documentation metadata with stable IDs,
topics, titles, canonical URLs, short descriptions, and keywords. Its purpose is to demonstrate
governed discovery through Genie, not to duplicate the documents themselves. Seed topics include
AI Gateway web search, managed MCP servers, OBO/user authorization, inference-table auditing, and
Databricks Apps OAuth scopes.

## Architecture

The application is an `agent-openai` scaffold using the managed `agentbricks` server and one
OpenAI Agents SDK orchestrator. Managed bindings in `agent.toml` provide:

1. `system.ai.slack` as managed MCP with `auth = "user"`.
2. A first-class native Genie Agent binding for the provisioned space with `auth = "user"`.
3. `system.ai.web_search` as managed MCP with `auth = "user"`.
4. `system.ai.sandbox` with `auth = "user"` and a `read_write` downscope limited to the reports
   Volume.

The model chooses and calls these tools under a strict developer prompt, while deterministic Python
helpers enforce fixed inputs, source normalization, URL policy, report contract, and artifact
verification. This keeps the demo visibly agentic without delegating authorization or citation
policy to model judgment alone.

The application is organized into focused units:

- `agent/agent.py`: composes the OpenAI agent, instructions, managed tools, and run loop.
- `agent/report_contract.py`: fixed source/output constants, evidence and report schemas, and URL
  policy.
- `agent/tools/report_tools.py`: local deterministic tools that normalize evidence, validate
  citations, and assemble the sandbox write request.
- `scripts/setup_df1.py`: idempotent provisioning, seed data, Genie creation, and recorded grants.
- `scripts/deploy_df1.py`: local-CLI bootstrap, managed binding configuration, deploy, and App
  permission setup.
- `scripts/invoke_e2e.py`: deployed invocation, progress polling, artifact download, semantic checks,
  and evidence capture.
- `scripts/teardown_df1.py`: targeted deletion using only IDs and names from setup state.

## Request and Data Flow

The E2E prompt asks the deployed agent to create the Databricks web search report. The orchestrator
then follows this required sequence:

1. Read the supplied Slack thread. Extract the report's field observations, preserving message
   permalinks and identifying them as internal claims rather than official product guarantees.
2. Ask the configured native Genie Agent which cataloged official assets are relevant. If the
   initial call returns an in-progress state, call the matching poll tool with the exact returned
   `conversation_id` and `message_id`; never resubmit the question blindly. Fetch query results for
   returned query attachments when present.
3. Search official Databricks documentation for each selected topic through
   `system.ai.web_search`, requesting the approved domains below.
4. Normalize Slack, Genie, and web outputs into an evidence ledger. Reject web citations whose
   canonical host is outside the approved list even if the MCP response ignored the requested
   domain filter.
5. Draft a report containing an executive summary, internal field report, cataloged assets,
   official documentation findings, discrepancies/limitations, and sources.
6. Verify every factual bullet has an evidence ID and that every external citation passes the host
   policy. Unsupported claims are removed or visibly marked as unverified.
7. Call sandbox with sanitized evidence and report content. Sandbox writes the Markdown report and
   evidence JSON only at the two fixed Volume paths.
8. Return a concise completion payload with report path, evidence path, counts by source kind,
   rejected citations, and any warnings.

## Evidence Contract

Each normalized evidence item has:

- `evidence_id`: stable run-local identifier such as `slack-001`, `genie-001`, or `web-001`.
- `source_kind`: `slack`, `genie_asset`, or `official_web`.
- `title`: short display title.
- `canonical_uri`: Slack permalink, Genie space/conversation deep link or asset URL, or official web
  URL.
- `excerpt`: bounded text or result cells actually returned by the source.
- `retrieved_at`: UTC timestamp.
- `source_metadata`: message/thread IDs, Genie conversation/message/attachment IDs, or web host.
- `authority`: `internal_field_report`, `governed_asset_catalog`, or `official_documentation`.

The report cites evidence IDs inline and resolves them in its Sources section. A Slack observation
may describe experienced behavior but cannot be presented as official documentation. A Genie row
identifies an asset but does not prove the asset's contents; official claims require a validated web
source.

## Citation and Content Policy

Approved official hosts are:

- `docs.databricks.com`
- `learn.microsoft.com` only for Azure Databricks documentation paths

The validator parses and canonicalizes every URL, requires HTTPS, rejects credentials and malformed
hosts, and checks exact host or subdomain boundaries. Redirect-looking or lookalike domains are not
accepted. The approved host check is performed after retrieval and before drafting, independently
of any `allowed_domains` argument sent to web search.

The final report must state that the Slack thread reported that domain allowlisting could still
surface out-of-scope information and that raw-result retrieval was unavailable at the time of the
field report. Official documentation findings are presented separately so readers can see whether
the current documented behavior agrees.

## Authorization and Least Privilege

All four managed entries explicitly use `auth = "user"`. Deploy requests the union of required App
user scopes: `ai-gateway`, `genie`, and `files`. User-token forwarding must be enabled and effective
before source rollout. The invoking user needs access to Slack MCP, web search MCP, the Genie space,
the SQL warehouse and asset table through the Genie space, and the reports Volume.

The App service principal receives only:

- `USE_CATALOG` on `supervisor_agent` if required for resource resolution.
- `USE_SCHEMA` on the demo schema if required by the Genie permission path.
- `CAN_READ` on the configured Genie space if the platform requires the App resource principal to
  resolve the binding.
- `CAN_USE` on the App for the invoking user.

It must not receive `SELECT` on the asset table or `READ_VOLUME`/`WRITE_VOLUME` on the reports
Volume. Setup and E2E both inspect effective grants and fail if those data privileges appear. This
makes successful Genie queries and sandbox writes evidence of request-user execution rather than an
App-identity fallback.

No connector token, forwarded access token, or credential-derived header is written to logs,
Runtime Store, evidence JSON, prompts, or the Volume.

## Setup and Lifecycle

`setup_df1.py` authenticates with profile `df1`, confirms the expected current user, creates a unique
schema and managed Volume, creates and seeds the Delta asset table, creates the Genie space over that
table, and saves every resource ID/name immediately after creation. Re-running setup with existing
state is a no-op after it verifies the recorded resources; partial state is resumed only when the
recorded resource matches the expected type and owner.

`deploy_df1.py` installs the editable CLI from the checked-out latest-main source, scaffolds the
OpenAI project when needed, writes the four explicit managed bindings, validates the manifest, and
deploys the App with `--allow-user-scope-update` when required. It records deployed commit, CLI
version, App URL, App ID, service principal ID, effective scopes, source hash, and deployment ID.

`teardown_df1.py` requires `setup-state.json`, deletes only the recorded App, Genie space, and demo
schema, and confirms each target before deletion. It has no wildcard discovery or catalog-wide
cleanup. Successful E2E leaves resources running; teardown is an explicit operator action.

## Error Handling

- Slack authorization, missing-thread, and malformed-result failures stop the run with a source-
  specific message.
- Genie `NOT_SUBMITTED` may be retried once because no question was sent.
- Genie `INDETERMINATE_SUBMISSION` is never blindly retried. The run records the state and stops
  unless usable IDs are available for polling.
- Genie in-progress results are polled with returned identifiers and a bounded overall deadline.
- Web search timeouts are retried once per topic. Empty searches become warnings; invalid-domain
  results are recorded as rejected and excluded.
- If no validated official sources remain, report generation fails rather than producing an
  uncited official-doc section.
- Sandbox writes use a unique session and fixed paths. The agent verifies both files exist and
  contain the expected run ID before returning success.
- Setup/deploy/E2E commands write timestamped, token-free evidence files for postmortem use.

## Testing Strategy

Unit tests cover URL canonicalization and host allowlisting, evidence normalization, claim/evidence
validation, required report sections, fixed output paths, and the prompt's Genie polling rule.
Mocked agent tests verify the tool order and ensure an in-progress Genie response is followed by a
poll using the same IDs rather than another ask.

Local integration tests validate the generated `agent.toml`, tool discovery, derived OAuth scope
union, sandbox Volume downscope, and OpenAI agent construction against the current checkout.

The DF1 E2E test is successful only when one deployed invocation:

1. reaches terminal success;
2. contains evidence from the exact Slack thread;
3. contains at least one Genie asset row and recorded conversation/message identifiers;
4. contains at least one allowed official-documentation URL and no accepted disallowed URL;
5. creates both Volume artifacts with the invocation run ID;
6. produces a Markdown report with all required sections and resolvable evidence IDs;
7. confirms the App service principal lacks table and Volume data privileges; and
8. records App effective scopes including `ai-gateway`, `genie`, and `files`.

The E2E runner polls deployment and invocation state at bounded intervals, emits a compact progress
line at least once per minute, captures build/runtime logs on failure, and iterates through diagnosed
code/config fixes until the full deployed invocation passes.

## Deliverables

- A current OpenAI Agents SDK app under
  `integrations/agentbricks/examples/genie-web-search-report/`.
- Complete DF1 setup, deploy, invoke/verify, inspect, and teardown scripts.
- Unit and local integration tests that do not require DF1.
- A checked-in example `.env`/configuration file without credentials.
- A README with exact commands, resource names, consent expectations, evidence locations, and
  teardown behavior.
- Captured E2E result JSON identifying the deployed App, setup resources, source commit, invocation,
  report path, checks, and final pass/fail result, with secrets removed.
