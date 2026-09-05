import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, cast

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.messages import AIMessageChunk
from langgraph.types import Command

from agent.databricks_tools import DATABRICKS_TOOLS
from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_mason import (
    configure_tracing,
    tag_session,
    workspace_client,
    workspace_headers,
)
from databricks_mason.langgraph import checkpointer, load_tools, memory_tools, thread_config
from databricks_mason.runtime import InvocationAuthPolicy, RequestAuthContext

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"
AUTH_POLICY = InvocationAuthPolicy.from_integrations(DATABRICKS_TOOLS)

# Tools that require human approval before they run. Map a tool name to True to allow every decision
# (approve / edit / reject / respond), or to a config dict to restrict them (see HumanInTheLoopMiddleware).
# When a listed tool is about to run, the agent pauses and emits an `interrupt` event; the client
# resumes by sending `resume` with the same session id. Empty this dict to disable approval gating.
REQUIRE_APPROVAL: dict[str, bool | InterruptOnConfig] = {"send_message": True}


class _RoutedChatDatabricks(ChatDatabricks):
    """Forward account-host workspace routing to the underlying OpenAI clients."""

    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
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


async def create_agent_graph(request_auth: RequestAuthContext | None = None):
    """Build the LangGraph agent: local tools + long-term-memory tools + any MCP tools."""
    if request_auth is None:
        request_auth = RequestAuthContext.from_forwarded_token(None)
    local_tools = [*all_tools(), *memory_tools()]
    tools = [
        *local_tools,
        *await load_tools(
            DATABRICKS_TOOLS,
            extra_servers=build_mcp_servers(),
            workspace_client_for=request_auth.client_for,
            existing_tools=local_tools,
        ),
    ]
    middleware = (
        [HumanInTheLoopMiddleware(interrupt_on=REQUIRE_APPROVAL)] if REQUIRE_APPROVAL else []
    )
    return create_agent(
        model=_RoutedChatDatabricks(endpoint=MODEL, workspace_client=workspace_client()),
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer(),
    )


def _session_id(request: dict) -> str:
    """Return the session id derived by the runtime from the Apps routing cookie.

    Clients do not send ``session_id`` in the body. The runtime makes the cookie value available to
    the handler after resolving the deployed Apps cookie or the local-development fallback cookie.
    """
    return str(request["session_id"])


async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
    """Run one turn to completion. Called by the runtime for POST /invocations.

    ``request`` is a dict with an ``input`` list of LangChain message dicts; the returned dict carries
    the run's new messages (LangChain-native shape) and the ``session_id`` to pass back next turn. If a
    gated tool needs approval the run pauses: ``output`` then ends with an ``interrupt`` event and
    ``status`` is ``"interrupted"`` — resume by calling again with the same session id and a ``resume``
    payload.
    """
    request = {**request, "session_id": _session_id(request)}
    async with aclosing(stream_handler(request, request_auth)) as stream:
        outputs = [event async for event in stream if event.get("type") in ("message", "interrupt")]
    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [e["message"] if e["type"] == "message" else e for e in outputs],
        "session_id": request["session_id"],
        "status": "interrupted" if interrupted else "completed",
    }


async def stream_handler(
    request: dict, request_auth: RequestAuthContext
) -> AsyncGenerator[dict, None]:
    """Stream the agent's run events as JSON dicts. Called by the runtime when stream=true."""
    routing_session = _session_id(request)
    state_session_id = request_auth.state_key(routing_session)
    tag_session(routing_session)

    agent = await create_agent_graph(request_auth)
    # A `resume` payload continues a session paused awaiting approval; otherwise start a new turn from
    # `input`. Either way the checkpointer keys off the principal-bound state session for prior
    # history / paused state. LangChain accepts message dicts natively, so `input` is passed straight
    # through (new turn only).
    resume = request.get("resume")
    agent_input = (
        Command(resume=resume) if resume is not None else {"messages": request.get("input") or []}
    )

    serialized_events = _serialize_events(
        agent.astream(
            input=agent_input,
            config=thread_config(state_session_id),
            stream_mode=["updates", "messages"],
        )
    )
    async with aclosing(serialized_events) as stream:
        async for event in stream:
            yield event


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    """Turn LangGraph's ``astream`` events into JSON dicts in LangChain's native shape (not reshaped).

    ``stream_mode=["updates", "messages"]`` yields completed node outputs (full LangChain messages,
    incl. tool calls/results) and token-level chunks. Completed messages become
    ``{"type": "message", "message": <dict>}`` and text chunks ``{"type": "delta", "content", "id"}``.
    A human-approval gate surfaces as an ``__interrupt__`` update, relayed as
    ``{"type": "interrupt", "id", "value"}``; the run is then paused on the session's thread until the
    client resumes with the same session id.
    """
    stream = cast(AsyncGenerator[Any, None], async_stream)
    async with aclosing(stream):
        async for event in stream:
            mode, payload = event[0], event[1]
            if mode == "updates":
                if interrupts := payload.get("__interrupt__"):
                    for it in interrupts:
                        yield {"type": "interrupt", "id": it.id, "value": it.value}
                    continue
                for node_data in payload.values():
                    messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                    for msg in messages:
                        yield {"type": "message", "message": msg.model_dump()}
            elif mode == "messages":
                try:
                    chunk = payload[0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield {"type": "delta", "content": content, "id": chunk.id}
                except Exception:
                    logger.exception("Error processing agent stream chunk")
