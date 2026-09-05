"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agent.agent import _apply_decisions, _normalize_item, _serialize_events, _session_id
from agent.tools import all_tools
from agents import FunctionTool, RunResultStreaming, RunState
from openai.types.responses import ResponseTextDeltaEvent

from databricks_mason.runtime import (
    InvocationAuthPolicy,
    RequestAuthContext,
    UserAuthHITLUnsupported,
)


def test_tools_autoregister():
    tools = all_tools()
    assert tools, "expected the sample tool to auto-register"
    assert all(isinstance(t, FunctionTool) for t in tools)
    assert {"get_current_time", "send_message"} <= {t.name for t in tools}


def test_gated_tool_needs_approval():
    # The gated demo tool must exist, be listed for approval, and declare needs_approval, or the HITL
    # demo does nothing.
    from agent.agent import REQUIRE_APPROVAL

    assert "send_message" in REQUIRE_APPROVAL
    send = next(t for t in all_tools() if t.name == "send_message")
    assert send.needs_approval is True


class _FakeItem:
    """Stand-in for an Agents SDK run item, matched by _normalize_item's isinstance checks."""


def test_normalize_message_item():
    from agents.items import MessageOutputItem

    item = object.__new__(MessageOutputItem)
    # ItemHelpers.text_message_output reads raw_item.content; give it a text part.
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    item.raw_item = ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="hello", annotations=[])],
    )
    assert _normalize_item(item) == {"role": "assistant", "content": "hello"}


class _FakeToolApproval:
    def __init__(self, name, args, call_id):
        self.tool_name, self.arguments, self.call_id = name, args, call_id


class _FakeStreamResult:
    """Minimal RunResultStreaming stand-in: a delta, a message, then a pending interruption."""

    def __init__(self, events, interruptions, state):
        self._events, self.interruptions, self._state = events, interruptions, state

    async def stream_events(self):
        for event in self._events:
            yield event

    def to_state(self):
        return self._state


class _TrackingAsyncIterator:
    def __init__(self, values: list[Any], *, block_when_empty: bool = False) -> None:
        self._values = iter(values)
        self._block_when_empty = block_when_empty
        self.waiting = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._values)
        except StopIteration:
            if not self._block_when_empty:
                raise StopAsyncIteration from None
            self.waiting.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_serialize_events_relays_interrupt_as_native_event():
    from agent.agent import _pending_runs

    approval = _FakeToolApproval("send_message", '{"recipient": "x", "body": "y"}', "call-1")
    sentinel_state = object()
    result = _FakeStreamResult([], [approval], sentinel_state)

    events = [e async for e in _serialize_events(cast(RunResultStreaming, result), "sess-1")]

    assert events == [
        {
            "type": "interrupt",
            "id": "call-1",
            "value": {
                "action_requests": [
                    {"name": "send_message", "args": {"recipient": "x", "body": "y"}}
                ]
            },
        }
    ]
    # The paused run is stashed in-process, keyed by session id, for a later resume.
    assert _pending_runs.pop("sess-1") is sentinel_state


@pytest.mark.asyncio
async def test_obo_interrupt_rejected_before_live_run_state_is_stored(monkeypatch):
    import agent.agent as agent_module

    approval = _FakeToolApproval("send_message", "{}", "call-obo")

    class UnsafeStateResult(_FakeStreamResult):
        def to_state(self):
            raise AssertionError("credential-bearing state must not be materialized")

    result = UnsafeStateResult([], [approval], None)
    agent_module._pending_runs.pop("principal-state-key", None)
    monkeypatch.setattr(agent_module, "AUTH_POLICY", InvocationAuthPolicy(True))

    with pytest.raises(UserAuthHITLUnsupported):
        [
            event
            async for event in _serialize_events(
                cast(RunResultStreaming, result), "principal-state-key"
            )
        ]

    assert "principal-state-key" not in agent_module._pending_runs


@pytest.mark.asyncio
async def test_stream_close_closes_result_events_and_mcp_stack(monkeypatch):
    import agent.agent as agent_module

    sdk_events = _TrackingAsyncIterator(
        [
            SimpleNamespace(
                type="raw_response_event",
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="hello",
                    item_id="item-1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=0,
                    type="response.output_text.delta",
                ),
            )
        ]
    )

    class FakeResult:
        interruptions = []

        def stream_events(self):
            return sdk_events

    class TrackingServer:
        def __init__(self) -> None:
            self.exited = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.exited = True

    server = TrackingServer()

    async def fake_bind_tools(agent, integrations, *, stack, workspace_client_for):
        del integrations, stack, workspace_client_for
        return agent

    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: [server])
    monkeypatch.setattr(agent_module, "create_agent", lambda mcp: mcp)
    monkeypatch.setattr(agent_module, "bind_tools", fake_bind_tools)
    monkeypatch.setattr(agent_module.Runner, "run_streamed", lambda *args, **kwargs: FakeResult())
    monkeypatch.setattr(agent_module, "session_store", lambda _: object())
    monkeypatch.setattr(agent_module, "tag_session", lambda _: None)
    stream = agent_module.stream_handler(
        {"input": [], "session_id": "routing-session"},
        RequestAuthContext.from_forwarded_token(None),
    )

    assert await anext(stream) == {
        "type": "delta",
        "content": "hello",
        "id": "item-1",
    }
    await stream.aclose()

    assert sdk_events.closed is True
    assert server.exited is True


def test_apply_decisions_approves_pending_run(monkeypatch):
    from agent.agent import _pending_runs

    approved = []

    class _State:
        def get_interruptions(self):
            return ["item-a"]

        def approve(self, item):
            approved.append(item)

        def reject(self, item, rejection_message=None):
            raise AssertionError("should not reject on approve")

    _pending_runs["sess-2"] = cast(RunState, _State())
    state = _apply_decisions("sess-2", {"decisions": [{"type": "approve"}]})
    assert approved == ["item-a"]
    assert "sess-2" not in _pending_runs  # popped so it can't be resumed twice


def test_apply_decisions_without_pending_run_raises():
    with pytest.raises(RuntimeError, match="No paused run"):
        _apply_decisions("never-started", {"decisions": [{"type": "approve"}]})


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_session_store_defaults_to_in_process(monkeypatch):
    import databricks_mason.openai.sessions as ss

    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    ss._local_sessions.clear()
    # In-process default: same session id returns the same cached SQLiteSession (multi-turn works).
    assert ss.session_store("abc-123") is ss.session_store("abc-123")


def test_session_store_selects_durable_store(monkeypatch):
    import databricks_mason.openai.sessions as ss

    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    store = ss.session_store("abc-123")
    assert isinstance(store, ss.DatabricksSessionStore)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


def test_session_id_from_request():
    request = {"input": [{"role": "user", "content": "hi"}], "session_id": "abc-123"}
    assert _session_id(request) == "abc-123"


def test_session_id_is_required_from_runtime():
    with pytest.raises(KeyError):
        _session_id({"input": [{"role": "user", "content": "hi"}]})


@pytest.mark.asyncio
async def test_invoke_handler_passes_same_auth_context_to_stream(monkeypatch):
    import agent.agent as agent_module

    request_auth = RequestAuthContext.from_forwarded_token(None)
    captured = {}

    async def fake_stream_handler(request, stream_request_auth):
        captured["request"] = request
        captured["request_auth"] = stream_request_auth
        yield {"type": "message", "message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(agent_module, "stream_handler", fake_stream_handler)
    response = await agent_module.invoke_handler(
        {"input": [], "session_id": "routing-session"}, request_auth
    )

    assert captured["request_auth"] is request_auth
    assert captured["request"]["session_id"] == "routing-session"
    assert response == {
        "output": [{"role": "assistant", "content": "ok"}],
        "session_id": "routing-session",
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_invoke_handler_cancellation_closes_stream_iterator(monkeypatch):
    import agent.agent as agent_module

    stream = _TrackingAsyncIterator(
        [{"type": "message", "message": {"role": "assistant", "content": "partial"}}],
        block_when_empty=True,
    )

    def fake_stream_handler(request, request_auth):
        del request, request_auth
        return stream

    monkeypatch.setattr(agent_module, "stream_handler", fake_stream_handler)
    task = asyncio.create_task(
        agent_module.invoke_handler(
            {"input": [], "session_id": "routing-session"},
            RequestAuthContext.from_forwarded_token(None),
        )
    )
    await asyncio.wait_for(stream.waiting.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True


@pytest.mark.asyncio
async def test_two_forwarded_users_partition_openai_sessions_and_inject_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.agent as agent_module

    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    contexts = [
        RequestAuthContext.from_forwarded_token("token-a"),
        RequestAuthContext.from_forwarded_token("token-b"),
    ]
    bound_contexts = []
    session_keys = []
    run_sessions = []
    run_states = []
    tagged = []

    async def fake_bind_tools(
        agent,
        integrations,
        *,
        stack,
        workspace_client_for,
    ):
        del integrations, stack
        bound_contexts.append(workspace_client_for.__self__)
        return agent

    def fake_session_store(session_id):
        session_keys.append(session_id)
        return f"store:{session_id}"

    def fake_run_streamed(agent, run_input, session=None):
        del agent, run_input
        run_sessions.append(session)
        state = object()
        run_states.append(state)
        return _FakeStreamResult(
            [],
            [_FakeToolApproval("send_message", "{}", f"call-{len(run_states)}")],
            state,
        )

    monkeypatch.setattr(agent_module, "bind_tools", fake_bind_tools)
    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: [])
    monkeypatch.setattr(agent_module, "create_agent", lambda mcp: "agent")
    monkeypatch.setattr(agent_module, "session_store", fake_session_store)
    monkeypatch.setattr(agent_module.Runner, "run_streamed", fake_run_streamed)
    monkeypatch.setattr(agent_module, "tag_session", tagged.append)

    streamed_events = []
    for request_auth in contexts:
        events = [
            event
            async for event in agent_module.stream_handler(
                {"input": [], "session_id": "routing-session"}, request_auth
            )
        ]
        streamed_events.append(events)

    expected_keys = [context.state_key("routing-session") for context in contexts]
    assert bound_contexts == contexts
    assert session_keys == expected_keys
    assert run_sessions == [f"store:{state_key}" for state_key in expected_keys]
    assert expected_keys[0] != expected_keys[1]
    assert tagged == ["routing-session", "routing-session"]
    assert [events[0]["id"] for events in streamed_events] == ["call-1", "call-2"]
    assert [agent_module._pending_runs.pop(state_key) for state_key in expected_keys] == run_states


def test_auth_policy_matches_declared_integrations() -> None:
    import agent.agent as agent_module

    assert agent_module.AUTH_POLICY == InvocationAuthPolicy.from_integrations(
        agent_module.DATABRICKS_TOOLS
    )


def _has_workspace_auth() -> bool:
    return bool(
        os.getenv("DATABRICKS_CONFIG_PROFILE")
        or (os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    )


@pytest.mark.skipif(
    not _has_workspace_auth(),
    reason="no Databricks profile configured; skipping live model call",
)
@pytest.mark.asyncio
async def test_agent_responds_end_to_end():
    from agent.agent import configure, create_agent
    from agents import Runner

    from databricks_mason.openai import session_store

    configure()
    agent = create_agent()
    result = await Runner.run(
        agent,
        [{"role": "user", "content": "Reply with the single word: pong"}],
        session=session_store("test-e2e"),
    )
    assert result.final_output
