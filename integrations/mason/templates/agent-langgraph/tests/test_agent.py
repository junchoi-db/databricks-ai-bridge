"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import asyncio
import os
from typing import Any
from uuid import UUID

import pytest
from agent.agent import _serialize_events, _session_id
from agent.tools import all_tools
from httpx import ASGITransport, AsyncClient
from langchain_core.tools import BaseTool
from runtime.runtime import build_app

from databricks_mason.langgraph.session_store import checkpointer, thread_config
from databricks_mason.runtime import InvocationAuthPolicy, RequestAuthContext


def test_tools_autoregister():
    tools = all_tools()
    assert tools, "expected the sample tool to auto-register"
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {"get_current_time", "send_message"} <= {t.name for t in tools}


def test_gated_tool_is_in_require_approval():
    # The gated demo tool must exist and be listed for approval, or the HITL demo does nothing.
    from agent.agent import REQUIRE_APPROVAL

    assert REQUIRE_APPROVAL.get("send_message")
    assert "send_message" in {t.name for t in all_tools()}


class _FakeInterrupt:
    def __init__(self, value, id):  # mirrors langgraph.types.Interrupt's `.value` / `.id`
        self.value, self.id = value, id


async def _aiter(events):
    for e in events:
        yield e


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
    hitl = {"action_requests": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}]}
    stream = _aiter([("updates", {"__interrupt__": (_FakeInterrupt(hitl, "int-1"),)})])
    events = [e async for e in _serialize_events(stream)]
    assert events == [{"type": "interrupt", "id": "int-1", "value": hitl}]


@pytest.mark.asyncio
async def test_stream_close_closes_langgraph_event_iterator(monkeypatch):
    import agent.agent as agent_module

    hitl = {"action_requests": [{"name": "send_message", "args": {}}]}
    graph_events = _TrackingAsyncIterator(
        [("updates", {"__interrupt__": (_FakeInterrupt(hitl, "int-close"),)})]
    )

    class FakeGraph:
        def astream(self, *, input, config, stream_mode):
            del input, config, stream_mode
            return graph_events

    async def fake_create_agent_graph(request_auth):
        del request_auth
        return FakeGraph()

    monkeypatch.setattr(agent_module, "create_agent_graph", fake_create_agent_graph)
    monkeypatch.setattr(agent_module, "tag_session", lambda _: None)
    stream = agent_module.stream_handler(
        {"input": [], "session_id": "routing-session"},
        RequestAuthContext.from_forwarded_token(None),
    )

    assert await anext(stream) == {"type": "interrupt", "id": "int-close", "value": hitl}
    await stream.aclose()

    assert graph_events.closed is True


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_chat_model_forwards_account_routing_header(monkeypatch):
    from agent.agent import _RoutedChatDatabricks

    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456")
    model = _RoutedChatDatabricks(endpoint="test-endpoint")

    assert model._get_client_kwargs()["default_headers"] == {"X-Databricks-Org-Id": "123456"}


def test_thread_config_from_session_id():
    # actor_id rides alongside thread_id — the durable saver maps it onto the Session's actor.
    assert thread_config("abc-123") == {
        "configurable": {"thread_id": "abc-123", "actor_id": "abc-123"}
    }


def test_thread_config_uses_configured_actor(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
    assert thread_config("abc-123") == {
        "configurable": {"thread_id": "abc-123", "actor_id": "alice"}
    }


def test_checkpointer_is_shared(monkeypatch):
    # In-memory by default (no AGENT_SESSION_STORE); built once and shared so multi-turn works.
    import databricks_mason.langgraph.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)  # reset the process-wide saver
    assert checkpointer() is checkpointer()


def test_session_store_selects_durable_saver(monkeypatch):
    # AGENT_SESSION_STORE must route to the durable Session Store saver, not stay in-memory. Stub the
    # REST client so it stays hermetic (no network); the saver builds without touching the API.
    import databricks_mason.langgraph.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)
    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    saver = checkpointer()
    assert isinstance(saver, ss.DatabricksSessionStoreSaver)


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
async def test_runtime_uses_apps_routing_cookie_for_resume_request():
    captured = {}

    async def invoke_handler(request, request_auth):
        assert isinstance(request_auth, RequestAuthContext)
        captured.update(request)
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    async def stream_handler(request, request_auth):
        del request_auth
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        response = await client.post(
            "/invocations",
            headers={"cookie": "__Host-databricks-app-router=same-session-id"},
            json={
                "session_id": "body-value-is-ignored",
                "resume": {"decisions": [{"type": "approve"}]},
            },
        )

    assert response.status_code == 200
    assert captured == {
        "resume": {"decisions": [{"type": "approve"}]},
        "session_id": "same-session-id",
    }


@pytest.mark.asyncio
async def test_runtime_sets_local_session_cookie_when_apps_router_is_absent():
    async def invoke_handler(request, request_auth):
        del request_auth
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    async def stream_handler(request, request_auth):
        del request_auth
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/invocations", json={"input": []})
        second = await client.post("/invocations", json={"input": []})

    assert first.status_code == 200
    assert first.cookies.get("mason-local-session") == first.json()["session_id"]
    assert second.json()["session_id"] == first.json()["session_id"]


@pytest.mark.asyncio
async def test_runtime_rotates_local_session_cookie():
    async def invoke_handler(request, request_auth):
        del request_auth
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    async def stream_handler(request, request_auth):
        del request_auth
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        current = (await client.post("/invocations", json={"input": []})).json()["session_id"]
        created = await client.post("/api/session/new")

        assert created.status_code == 200
        assert created.json()["previous_session_id"] == current
        assert created.json()["session_id"] != current
        UUID(created.json()["session_id"])
        assert created.cookies.get("mason-local-session") == created.json()["session_id"]
        assert (await client.post("/invocations", json={"input": []})).json()[
            "session_id"
        ] == created.json()["session_id"]


@pytest.mark.asyncio
async def test_runtime_rotates_apps_routing_cookie_and_clears_local_fallback():
    async def invoke_handler(request, request_auth):
        del request_auth
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    async def stream_handler(request, request_auth):
        del request_auth
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        created = await client.post(
            "/api/session/new",
            headers={
                "cookie": "__Host-databricks-app-router=old-session; mason-local-session=stale"
            },
        )

        assert created.status_code == 200
        assert created.json()["previous_session_id"] == "old-session"
        session_id = created.json()["session_id"]
        set_cookie_headers = created.headers.get_list("set-cookie")
        routing_cookie = next(
            header
            for header in set_cookie_headers
            if header.startswith("__Host-databricks-app-router=")
        ).lower()
        assert "httponly" in routing_cookie
        assert "path=/" in routing_cookie
        assert "samesite=lax" in routing_cookie
        assert "secure" in routing_cookie
        assert any(
            header.startswith('mason-local-session=""') and "Max-Age=0" in header
            for header in set_cookie_headers
        )
        assert (await client.post("/invocations", json={"input": []})).json()[
            "session_id"
        ] == session_id


def _has_workspace_auth() -> bool:
    return bool(
        os.getenv("DATABRICKS_CONFIG_PROFILE")
        or (os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    )


@pytest.mark.asyncio
async def test_agent_attaches_empty_databricks_registry_at_construction(monkeypatch):
    import agent.agent as agent_module

    loaded = []
    request_auth = RequestAuthContext.from_forwarded_token(None)

    async def fake_load_tools(
        integrations,
        *,
        extra_servers=(),
        workspace_client_for=None,
        existing_tools=(),
    ):
        loaded.append((integrations, extra_servers, workspace_client_for, existing_tools))
        return ["managed"]

    monkeypatch.setattr(agent_module, "load_tools", fake_load_tools)
    monkeypatch.setattr(agent_module, "all_tools", lambda: ["local"])
    monkeypatch.setattr(agent_module, "memory_tools", lambda: ["memory"])
    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: ["custom"])
    monkeypatch.setattr(agent_module, "workspace_client", lambda: object())
    monkeypatch.setattr(agent_module, "_RoutedChatDatabricks", lambda **_: "model")
    monkeypatch.setattr(agent_module, "checkpointer", lambda: "checkpointer")
    monkeypatch.setattr(agent_module, "create_agent", lambda **kwargs: kwargs)

    graph = await agent_module.create_agent_graph(request_auth)

    assert loaded[0][:2] == ((), ["custom"])
    assert loaded[0][2].__self__ is request_auth
    assert loaded[0][3] == ["local", "memory"]
    assert graph["tools"] == ["local", "memory", "managed"]


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
async def test_two_forwarded_users_partition_langgraph_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.agent as agent_module

    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.delenv("AGENT_SESSION_ACTOR_ID", raising=False)
    contexts = [
        RequestAuthContext.from_forwarded_token("token-a"),
        RequestAuthContext.from_forwarded_token("token-b"),
    ]
    factory_contexts = []
    configs = []
    tagged = []

    class FakeGraph:
        def astream(self, *, input, config, stream_mode):
            del input, stream_mode
            configs.append(config)
            return _aiter([])

    async def fake_create_agent_graph(request_auth):
        factory_contexts.append(request_auth)
        return FakeGraph()

    monkeypatch.setattr(agent_module, "create_agent_graph", fake_create_agent_graph)
    monkeypatch.setattr(agent_module, "tag_session", tagged.append)

    for request_auth in contexts:
        events = [
            event
            async for event in agent_module.stream_handler(
                {"input": [], "session_id": "routing-session"}, request_auth
            )
        ]
        assert events == []

    expected_keys = [context.state_key("routing-session") for context in contexts]
    assert factory_contexts == contexts
    assert configs == [thread_config(state_key) for state_key in expected_keys]
    assert expected_keys[0] != expected_keys[1]
    assert tagged == ["routing-session", "routing-session"]


def test_auth_policy_matches_declared_integrations() -> None:
    import agent.agent as agent_module

    assert agent_module.AUTH_POLICY == InvocationAuthPolicy.from_integrations(
        agent_module.DATABRICKS_TOOLS
    )


@pytest.mark.skipif(
    not _has_workspace_auth(),
    reason="no Databricks profile configured; skipping live model call",
)
@pytest.mark.asyncio
async def test_agent_responds_end_to_end():
    from agent.agent import configure, create_agent_graph

    configure()
    agent = await create_agent_graph()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Reply with the single word: pong"}]},
        config=thread_config("test-e2e"),
    )
    assert result["messages"][-1].content
