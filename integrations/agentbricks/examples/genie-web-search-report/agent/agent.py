import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from agents import Agent, Runner, RunResultStreaming, RunState
from agents.mcp import MCPServerManager
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from agent.tools.report_tools import reset_ledger
from databricks_agentkit import workspace_client, workspace_headers
from databricks_agentkit.openai import (
    configure_tracing,
    genie_tools,
    mcp_servers,
    memory_tools,
    session_store,
    start_trace,
)
from databricks_agentkit.runtime.auth import AuthError

logger = logging.getLogger(__name__)

# A Unity Catalog AI Gateway model service, served from the `system.ai` schema and queried through
# the gateway (see `use_ai_gateway=True` in configure()). Swap for any `system.ai.*` model service
# your workspace exposes — the demo chat app's picker lists what's available.
MODEL = "system.ai.claude-sonnet-4-5"

SLACK_THREAD_URL = (
    "https://databricks.slack.com/archives/C088VN8U4E5/"
    "p1790353973998359?thread_ts=1789056476.567349&cid=C088VN8U4E5"
)
MAX_TURNS = 30

INSTRUCTIONS = f"""You create one evidence-backed Databricks documentation report.

Follow this sequence exactly.

1. Slack: use the request-user Slack MCP server to read the complete fixed thread at
   {SLACK_THREAD_URL}. The thread timestamp is 1789056476.567349 and the highlighted reply is
   1790353973.998359. Record concise observations with record_slack_evidence. Treat every Slack
   statement as an internal field report, never as official product documentation.
2. Genie: call genie_assets_ask once to find the most relevant cataloged official documentation
   assets for web search, domain filtering, raw results, and auditability. If it is still running,
   call genie_assets_poll with the same conversation_id and message_id returned by ask. Continue
   bounded polling with those exact identifiers. Never resubmit the question to work around a
   timeout. For INDETERMINATE_SUBMISSION, poll only when usable identifiers were returned; otherwise
   stop with a clear error. Fetch every query attachment through genie_assets_query_result and
   record its rows with record_genie_assets.
3. Official web search: use the request-user system.ai.web_search MCP server to validate each
   relevant asset. Request allowed domains docs.databricks.com and learn.microsoft.com when the
   tool schema supports them. allowed_domains is not a security boundary: pass every candidate
   result through record_web_evidence and cite only accepted results.
4. Draft Markdown with exactly these sections: ## Executive summary, ## Internal field report,
   ## Cataloged assets, ## Official documentation findings, ## Discrepancies and limitations, and
   ## Sources. Every factual bullet must cite an evidence ID such as [slack-001], [genie-001], or
   [web-001]. State separately what the internal field report observed and what official
   documentation currently says. Explicitly cover the field reports about domain filters,
   AI Gateway inference-table auditing, and lack of raw-result retrieval.
5. Call validate_report with the complete Markdown. Fix every returned error before continuing.
   Its successful JSON supplies report_path, evidence_path, run_id, and the sanitized evidence
   ledger.
6. Use the request-user system.ai.sandbox run_code tool to write the exact Markdown as UTF-8 to
   report_path (databricks_web_search_report.md) and the complete validation JSON as UTF-8 JSON to
   evidence_path (evidence.json). Use only those fixed paths. Then use sandbox to read both files
   back and verify they contain run_id. Do not pass connector credentials or raw request headers to
   sandbox.
7. Return one compact JSON object with status, run_id, report_path, evidence_path, evidence counts,
   rejected citations, Genie conversation_id/message_id, and warnings. Do not claim success unless
   all source stages ran, validation returned no errors, and you read both files back.
"""

# Tools that require human approval before they run. Add a tool's name here and the agent pauses when
# the model calls it, emitting an `interrupt` event; the client resumes by sending `resume` with the
# same session id. The tools declare `needs_approval=True` themselves (see agent/tools/); this set is
# how the runtime knows which pending calls to surface. Empty it to disable approval gating.
REQUIRE_APPROVAL: set[str] = set()

# OpenAI Sessions persist transcript history, not a paused RunState. Keep pending approvals local.
_pending_runs: dict[str, RunState] = {}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    from agents import set_default_openai_api, set_default_openai_client

    # use_ai_gateway routes to the Unity Catalog AI Gateway (`<host>/ai-gateway/mlflow/v1`), so
    # `MODEL` is a `system.ai.*` model name rather than a serving-endpoint name.
    set_default_openai_client(
        AsyncDatabricksOpenAI(
            workspace_client=workspace_client(),
            default_headers=workspace_headers() or None,
            use_ai_gateway=True,
        )
    )
    set_default_openai_api("chat_completions")
    configure_tracing()


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable.
    """
    try:
        workspace_client()
    except Exception as e:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = (
            f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
        )
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
            "Fix one of:\n"
            "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
            "  • run `databricks auth login --profile <name>` to create one, or\n"
            "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
            f"(underlying error: {e})"
        ) from e


def create_agent(
    actor: str,
    mcp=None,
    model: str | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> Agent:
    """Build the OpenAI Agents SDK agent: tools, memory, MCP servers, and model."""
    return Agent(
        name="Databricks documentation report agent",
        instructions=INSTRUCTIONS,
        model=model or MODEL,
        tools=[
            *all_tools(),
            *memory_tools(actor),
            *genie_tools(workspace_client_for=workspace_client_for),
        ],
        mcp_servers=mcp or [],
    )


def resume_agent(session_id: str, resume: dict[str, Any]) -> RunState:
    """Apply human decisions to a paused run and return the native RunState."""
    state = _pending_runs.pop(session_id, None)
    if state is None:
        raise RuntimeError(
            "No paused run for this session. HITL pauses are in-process only, so a restart or a "
            "different replica loses them; retry the turn."
        )
    decisions = resume.get("decisions") or []
    for decision, item in zip(decisions, state.get_interruptions(), strict=False):
        if decision.get("type") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("message"))
    return state


@asynccontextmanager
async def run_agent(
    agent_input: list[Any] | RunState,
    *,
    session_id: str,
    actor: str | None = None,
    model: str | None = None,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> AsyncIterator[RunResultStreaming]:
    """Run the agent and expose its native streaming result.

    This is the framework-native entrypoint. It has no dependency on Agent Bricks request or context types,
    so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
    reset_ledger(session_id)
    auth_kwargs = {"workspace_client_for": workspace_client_for} if workspace_client_for else {}
    servers = await mcp_servers(build_mcp_servers(), **auth_kwargs)
    async with MCPServerManager(servers) as manager:
        for server, error in manager.errors.items():
            if getattr(server, "_agentbricks_request_user", False) is True or isinstance(
                error, AuthError
            ):
                raise error
        active_servers = []
        for server in manager.active_servers:
            tool_filter = server.tool_filter
            try:
                server.tool_filter = None
                server.cache_tools_list = True
                async with asyncio.timeout(manager.connect_timeout_seconds):
                    await server.list_tools()
            except Exception as error:
                if getattr(server, "_agentbricks_request_user", False) is True or isinstance(
                    error, AuthError
                ):
                    raise
                logger.warning(
                    "Failed to list tools from MCP server %r; continuing without it.",
                    server.name,
                    exc_info=True,
                )
            else:
                active_servers.append(server)
            finally:
                server.tool_filter = tool_filter

        agent = create_agent(
            actor,
            active_servers,
            model=model,
            workspace_client_for=workspace_client_for,
        )
        with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
            if isinstance(agent_input, RunState):
                result = Runner.run_streamed(agent, agent_input, max_turns=MAX_TURNS)
            else:
                result = Runner.run_streamed(
                    agent,
                    agent_input,
                    session=session_store(session_id, actor),
                    max_turns=MAX_TURNS,
                )

            try:
                yield result
            finally:
                if workspace_client_for is not None and not result.is_complete:
                    result.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        async for _ in result.stream_events():
                            pass

            if result.interruptions:
                if workspace_client_for is not None:
                    raise AuthError(
                        "MCP_USER_AUTH_HITL_UNSUPPORTED",
                        "Request-user invocations do not support paused approvals.",
                        400,
                    )
                _pending_runs[session_id] = result.to_state()
            if span is not None:
                span.set_outputs({"output": result.final_output})
