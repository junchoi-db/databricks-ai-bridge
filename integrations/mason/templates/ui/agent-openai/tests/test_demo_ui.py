import copy
import json
import pathlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from runtime import ui
from runtime.runtime import build_app

from databricks_mason.runtime import (
    InvocationAuthPolicy,
    MissingUserAuthorization,
    RequestAuthContext,
)


class _FakeStateClient:
    def create_memory_entry(self, request, session_id):
        return {
            "name": "memory-stores/store/entries/entry",
            "session_id": session_id,
            **request.model_dump(),
        }

    def list_memory_entries(self, path_prefix=None):
        return {"managed_memory_entries": [{"path": f"{path_prefix or ''}/profile.md"}]}

    def search_memory_entries(self, request):
        return {"managed_memory_entries": [{"path": "/profile.md", "content": request.query}]}

    def ensure_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def get_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def list_sessions(self):
        return {
            "sessions": [
                {
                    "session_id": "s1",
                    "actor_id": "alice",
                    "last_activity_time": "2026-08-28T12:00:00Z",
                },
                {
                    "session_id": "s2",
                    "actor_id": "alice",
                    "last_activity_time": "2026-08-27T12:00:00Z",
                },
                {
                    "session_id": "public-s1",
                    "actor_id": "alice",
                    "metadata": {"public_session_id": "s1"},
                    "last_activity_time": "2026-08-28T12:01:00Z",
                },
            ]
        }

    def append_session_items(self, session_id, items):
        return {"session_items": [{"item_id": "1", "data": item} for item in items]}

    def list_session_items(self, session_id):
        return {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": session_id}},
                {
                    "item_id": "2",
                    "data": {"type": "assistant", "content": "saved reply"},
                },
                {
                    "item_id": "3",
                    "data": {"event_type": "checkpoint", "checkpoint_id": "checkpoint-1"},
                },
            ]
        }


async def _session_history(routing_session, state_session_id):
    del state_session_id
    return {
        "session_id": routing_session,
        "session_items": [
            {"item_id": "1", "data": {"role": "user", "content": routing_session}},
            {"item_id": "2", "data": {"role": "assistant", "content": "in-process reply"}},
        ],
        "interrupts": [],
    }


def _client(monkeypatch, *, configured=False, history=False, session_id="routing-session"):
    if configured:
        monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
        monkeypatch.setenv("AGENT_MEMORY_ACTOR_ID", "alice")
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
        monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
        monkeypatch.setattr(ui, "_state_client", lambda: _FakeStateClient())
    else:
        monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    if history:
        monkeypatch.setattr(ui, "_local_history", _session_history)

    async def invoke_handler(request, request_auth):
        assert isinstance(request_auth, RequestAuthContext)
        return {"output": [], "session_id": request["session_id"]}

    async def stream_handler(request, request_auth):
        del request_auth
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler, InvocationAuthPolicy(False))
    ui.install_ui(app, InvocationAuthPolicy(False))
    client = TestClient(app, base_url="https://testserver")
    client.cookies.set("__Host-databricks-app-router", session_id)
    return client


def _async_app(auth_policy: InvocationAuthPolicy):
    async def invoke_handler(request, request_auth):
        return {"output": [], "session_id": request["session_id"]}

    async def stream_handler(request, request_auth):
        if False:
            yield request, request_auth

    app = build_app(invoke_handler, stream_handler, auth_policy)
    ui.install_ui(app, auth_policy)
    return app


def _user_headers(token: str) -> dict[str, str]:
    return {
        "cookie": "__Host-databricks-app-router=shared-route",
        "x-forwarded-access-token": token,
    }


@pytest.mark.asyncio
async def test_user_auth_local_history_partitions_same_routing_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import databricks_mason.openai.sessions as sessions

    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    session_ids = []

    class FakeSession:
        async def get_items(self, limit=None):
            return []

    def fake_session_store(session_id):
        session_ids.append(session_id)
        return FakeSession()

    monkeypatch.setattr(sessions, "session_store", fake_session_store)
    app = _async_app(InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        responses = [
            await client.get("/api/demo/session/items", headers=_user_headers(token))
            for token in ("token-a", "token-b")
        ]

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["session_id"] for response in responses] == [
        "shared-route",
        "shared-route",
    ]
    assert session_ids == [
        "97d58c480991f0a9dd01c4199c79de6aeefc116e12ffa848b508dba3e1917884",
        "b3787f252ff948a826a9508f3eec5128ae5257bab47f38f7c2b913117efb0397",
    ]


@pytest.mark.asyncio
async def test_user_auth_managed_session_access_uses_principal_state_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    calls = []
    raw_results = []

    def state_result(session_id, *, include_items=False):
        result = {
            "session_id": session_id,
            "actor_id": session_id,
            "metadata": {
                "default_actor_id": session_id,
                "default_session_id": session_id,
                f"session:{session_id}": f"state {session_id} ready",
                "nested": [
                    {
                        "session_id": session_id,
                        f"actor:{session_id}": f"value:{session_id}:end",
                    }
                ],
            },
            "session_items": (
                [
                    {
                        "item_id": "1",
                        "data": {"role": "assistant", "content": "saved reply"},
                        "metadata": {"session_id": session_id},
                    }
                ]
                if include_items
                else []
            ),
        }
        raw_results.append((result, copy.deepcopy(result)))
        return result

    class StateClient:
        def ensure_session(self, session_id):
            calls.append(("ensure", session_id))
            return state_result(session_id)

        def get_session(self, session_id):
            calls.append(("get", session_id))
            return state_result(session_id)

        def append_session_items(self, session_id, items):
            calls.append(("append", session_id))
            return state_result(session_id, include_items=True)

        def list_session_items(self, session_id):
            calls.append(("list-items", session_id))
            return state_result(session_id, include_items=True)

    async def immediate(operation, *args):
        return operation(*args)

    monkeypatch.setattr(ui, "_state_client", lambda: StateClient())
    monkeypatch.setattr(ui, "_managed_call", immediate)
    app = _async_app(InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        responses = []
        for token in ("token-a", "token-b"):
            headers = _user_headers(token)
            responses.extend(
                [
                    await client.post("/api/demo/sessions", headers=headers),
                    await client.get("/api/demo/session", headers=headers),
                    await client.post(
                        "/api/demo/session/items",
                        headers=headers,
                        json={"items": [{"role": "user", "content": "hello"}]},
                    ),
                    await client.get("/api/demo/session/items", headers=headers),
                ]
            )

    key_a = "97d58c480991f0a9dd01c4199c79de6aeefc116e12ffa848b508dba3e1917884"
    key_b = "b3787f252ff948a826a9508f3eec5128ae5257bab47f38f7c2b913117efb0397"
    assert [response.status_code for response in responses] == [200] * 8
    assert calls == [
        (operation, state_key)
        for state_key in (key_a, key_b)
        for operation in ("ensure", "get", "append", "list-items")
    ]
    for index, response in enumerate(responses):
        body = response.json()
        state_key = key_a if index < 4 else key_b
        assert state_key not in json.dumps(body, sort_keys=True)
        assert body["session_id"] == "shared-route"
        assert body["actor_id"] == "shared-route"
        assert body["metadata"] == {
            "default_actor_id": "shared-route",
            "default_session_id": "shared-route",
            "session:shared-route": "state shared-route ready",
            "nested": [
                {
                    "session_id": "shared-route",
                    "actor:shared-route": "value:shared-route:end",
                }
            ],
        }
        if body["session_items"]:
            assert body["session_items"][0]["metadata"]["session_id"] == "shared-route"
    assert all(result == original for result, original in raw_results)
    assert key_a in json.dumps(raw_results[0][0], sort_keys=True)
    assert key_b in json.dumps(raw_results[4][0], sort_keys=True)


@pytest.mark.asyncio
async def test_managed_state_error_never_exposes_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    state_key = "97d58c480991f0a9dd01c4199c79de6aeefc116e12ffa848b508dba3e1917884"
    sentinel = "managed-state-exception-secret"

    class StateClient:
        def get_session(self, session_id):
            sensitive_local = f"{sentinel}:{session_id}"
            raise RuntimeError(f"session {session_id} not found ({sensitive_local})")

    async def immediate_to_thread(operation, *args):
        return operation(*args)

    monkeypatch.setattr(ui, "_state_client", lambda: StateClient())
    monkeypatch.setattr(ui.asyncio, "to_thread", immediate_to_thread)
    app = _async_app(InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        response = await client.get("/api/demo/session", headers=_user_headers("token-a"))

    assert response.status_code == 502
    assert response.json() == {"detail": "Managed state request failed."}
    serialized_response = response.text
    assert state_key not in serialized_response
    assert sentinel not in serialized_response
    assert state_key not in caplog.text
    assert sentinel not in caplog.text

    with pytest.raises(HTTPException) as exc_info:
        await ui._managed_call(StateClient().get_session, state_key)
    error = exc_info.value
    assert error.status_code == 502
    assert error.detail == "Managed state request failed."
    assert error.__cause__ is None
    assert error.__context__ is None
    assert state_key not in repr(error)
    assert sentinel not in repr(error)


@pytest.mark.asyncio
async def test_user_auth_managed_session_list_and_open_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")

    class StateClient:
        def list_sessions(self):
            return {"sessions": [{"session_id": "another-principal", "actor_id": "shared"}]}

        def get_session(self, session_id):
            return {"session_id": session_id, "actor_id": "agent"}

    async def immediate(operation, *args):
        return operation(*args)

    monkeypatch.setattr(ui, "_state_client", lambda: StateClient())
    monkeypatch.setattr(ui, "_managed_call", immediate)
    app = _async_app(InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        listed = await client.get("/api/demo/sessions", headers=_user_headers("token-a"))
        opened = await client.post(
            "/api/demo/sessions/other-route/open", headers=_user_headers("token-a")
        )

    assert listed.status_code == 403
    assert opened.status_code == 403
    assert listed.json()["detail"] == (
        "Managed session listing is unavailable for user-authenticated integrations."
    )
    assert opened.json()["detail"] == (
        "Managed session switching is unavailable for user-authenticated integrations."
    )


@pytest.mark.asyncio
async def test_local_user_auth_policy_preserves_managed_session_list_and_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "app")
    monkeypatch.setenv("DATABRICKS_MASON_RUN_LOCAL", "1")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "agent")

    class StateClient:
        def list_sessions(self):
            return {"sessions": [{"session_id": "saved-session", "actor_id": "agent"}]}

        def get_session(self, session_id):
            return {"session_id": session_id, "actor_id": "agent"}

    async def immediate(operation, *args):
        return operation(*args)

    monkeypatch.setattr(ui, "_state_client", lambda: StateClient())
    monkeypatch.setattr(ui, "_managed_call", immediate)
    app = _async_app(InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        config = await client.get("/api/demo/config")
        listed = await client.get(
            "/api/demo/sessions",
            headers={"cookie": "__Host-databricks-app-router=shared-route"},
        )
        opened = await client.post(
            "/api/demo/sessions/saved-session/open",
            headers={"cookie": "__Host-databricks-app-router=shared-route"},
        )

    assert config.json()["deployed"] is False
    assert listed.status_code == 200
    assert listed.json()["sessions"] == [{"session_id": "saved-session", "actor_id": "agent"}]
    assert opened.status_code == 200
    assert opened.json()["session_id"] == "saved-session"
    assert opened.json()["previous_session_id"] == "shared-route"


@pytest.mark.parametrize("forwarded_token", [None, "   "], ids=["missing", "blank"])
@pytest.mark.parametrize("managed", [False, True], ids=["local-history", "managed-history"])
@pytest.mark.asyncio
async def test_deployed_user_auth_missing_token_rejected_before_history_access(
    monkeypatch: pytest.MonkeyPatch,
    forwarded_token: str | None,
    managed: bool,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    accesses = []
    if managed:
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")

        class StateClient:
            def list_session_items(self, session_id):
                accesses.append(("managed", session_id))
                return {"session_items": []}

        monkeypatch.setattr(ui, "_state_client", lambda: StateClient())
    else:
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

        async def local_history(routing_session, state_session_id):
            accesses.append(("local", routing_session, state_session_id))
            return {"session_id": routing_session, "session_items": [], "interrupts": []}

        monkeypatch.setattr(ui, "_local_history", local_history)

    async def immediate(operation, *args):
        return operation(*args)

    monkeypatch.setattr(ui, "_managed_call", immediate)
    app = _async_app(InvocationAuthPolicy(True))
    headers = {"cookie": "__Host-databricks-app-router=shared-route"}
    if forwarded_token is not None:
        headers["x-forwarded-access-token"] = forwarded_token
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="https://testserver",
    ) as client:
        response = await client.get("/api/demo/session/items", headers=headers)

    assert response.status_code == 401
    assert response.json() == MissingUserAuthorization().to_error_envelope()
    assert accesses == []


def test_demo_ui_routes(monkeypatch):
    client = _client(monkeypatch)

    index = client.get("/")
    assert index.status_code == 200
    assert 'id="new-session"' in index.text
    assert 'id="session-list"' in index.text
    app_script = client.get("/ui-assets/app.js")
    assert app_script.status_code == 200
    assert "refreshSessionView({ hydrateChat: true })" in app_script.text
    assert 'fetch("/api/session/new"' in app_script.text
    assert "/api/demo/sessions/${encodeURIComponent(sessionId)}/open" in app_script.text
    assert "session_id: ensureSessionId()" not in app_script.text
    styles = client.get("/ui-assets/styles.css").text
    assert "@media (min-width: 1181px)" in styles
    assert "scrollbar-gutter: stable" in styles

    config = client.get("/api/demo/config").json()
    assert config["session_id"] == "routing-session"
    assert config["deployed"] is False
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["memory"]["enabled"] is False
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True
    assert config["session"]["mode"] == "In-process session"
    assert "durability" not in config
    assert "recovery" not in config

    sessions = client.get("/api/demo/sessions").json()
    assert sessions == {
        "sessions": [
            {
                "session_id": "routing-session",
                "actor_id": "agent",
                "metadata": {"client": "mason-demo-ui-local"},
            }
        ],
        "current_session_id": "routing-session",
        "managed": False,
    }

    assert client.post("/api/demo/memory/search", json={"query": "profile"}).status_code == 503
    assert client.post("/api/demo/sessions", json={"session_id": "ignored"}).status_code == 503


def test_demo_ui_renders_structured_runtime_errors_and_provider_links() -> None:
    app_script = (pathlib.Path(__file__).parents[1] / "ui" / "app.js").read_text(encoding="utf-8")

    assert "function requestError(" in app_script
    assert "function authorizationLinks(" in app_script
    assert "throw requestError(event.error)" in app_script
    assert "throw requestError(body.detail || body.error" in app_script
    assert "throw requestError(result.error ||" in app_script
    assert 'anchor.rel = "noopener noreferrer"' in app_script
    assert "new Error(event.error)" not in app_script


def test_demo_ui_disables_unsupported_background_mode() -> None:
    app_script = (pathlib.Path(__file__).parents[1] / "ui" / "app.js").read_text(encoding="utf-8")

    assert "function selectMode(button) {" in app_script
    assert "if (!button || button.disabled) return;" in app_script
    assert 'button.addEventListener("click", () => selectMode(button));' in app_script
    assert (
        "const backgroundButton = document.querySelector('[data-mode=\"background\"]');"
        in app_script
    )
    assert "backgroundButton.disabled = !config.background.enabled;" in app_script
    assert 'if (backgroundButton.disabled && state.mode === "background") {' in app_script
    assert "selectMode(document.querySelector('[data-mode=\"streaming\"]'));" in app_script


@pytest.mark.asyncio
async def test_demo_config_disables_background_for_user_auth() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_async_app(InvocationAuthPolicy(True))),
        base_url="https://testserver",
    ) as client:
        config = (await client.get("/api/demo/config")).json()

    assert config["background"] == {"enabled": False, "durable": False}


def test_unmanaged_local_history_route(monkeypatch):
    client = _client(monkeypatch, history=True, session_id="local-session")

    config = client.get("/api/demo/config").json()
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True

    result = client.get("/api/demo/session/items")
    assert result.status_code == 200
    assert [item["data"]["content"] for item in result.json()["session_items"]] == [
        "local-session",
        "in-process reply",
    ]


def test_managed_session_list_is_actor_scoped(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", 'alice "demo"')
    state_client = object.__new__(ui._ManagedStateClient)
    calls = []
    state_client._do = lambda method, path, **kwargs: calls.append((method, path, kwargs)) or {
        "sessions": []
    }

    assert state_client.list_sessions() == {"sessions": []}
    assert calls == [
        (
            "GET",
            "/api/agents/v1/session-stores/sessions/sessions",
            {
                "query": {
                    "filter": 'actor_id = "alice \\"demo\\""',
                    "order_by": "last_activity_time desc",
                    "page_size": 50,
                }
            },
        )
    ]


def test_chat_session_items_exclude_non_message_items():
    result = ui._chat_session_items(
        {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": "hello"}},
                {"item_id": "2", "data": {"type": "ai", "content": "hi"}},
                {"item_id": "3", "data": {"event_type": "checkpoint"}},
                {"item_id": "5", "data": {"content": "missing role"}},
            ],
            "next_page_token": "next",
        }
    )

    assert result == {
        "session_items": [
            {"item_id": "1", "data": {"role": "user", "content": "hello"}},
            {"item_id": "2", "data": {"type": "ai", "content": "hi"}},
        ],
        "next_page_token": "next",
    }


@pytest.mark.asyncio
async def test_local_history_reads_messages_from_in_process_session(monkeypatch):
    import databricks_mason.openai.sessions as ss

    class _FakeSession:
        async def get_items(self, limit=None):
            return [
                {"id": "m1", "role": "user", "content": "saved message"},
                {"role": "assistant", "content": "saved reply"},
            ]

    monkeypatch.setattr(ss, "session_store", lambda session_id: _FakeSession())
    result = await ui._local_history("saved-session", "saved-session")

    assert result == {
        "session_id": "saved-session",
        "session_items": [
            {"item_id": "m1", "data": {"id": "m1", "role": "user", "content": "saved message"}},
            {"item_id": "1", "data": {"role": "assistant", "content": "saved reply"}},
        ],
        "interrupts": [],
    }


def test_managed_memory_and_session_routes(monkeypatch):
    client = _client(monkeypatch, configured=True, session_id="s1")

    config = client.get("/api/demo/config").json()
    assert config["memory"] == {
        "enabled": True,
        "store": "memory-stores/store",
        "actor": "alice",
    }
    assert config["session"]["store"] == "sessions"
    assert config["session"]["actor"] == "alice"
    assert config["session"]["history"] is True

    created = client.post(
        "/api/demo/memory/entries",
        json={"path": "/profile.md", "content": "I work at Databricks"},
    )
    assert created.status_code == 200
    assert created.json()["path"] == "/profile.md"
    assert created.json()["session_id"] == "s1"
    assert client.get("/api/demo/memory/entries", params={"path_prefix": "/"}).status_code == 200
    search = client.post("/api/demo/memory/search", json={"query": "Databricks"})
    assert search.json()["managed_memory_entries"][0]["content"] == "Databricks"

    assert (
        client.post("/api/demo/sessions", json={"session_id": "ignored"}).json()["session_id"]
        == "s1"
    )
    listed = client.get("/api/demo/sessions").json()
    assert [session["session_id"] for session in listed["sessions"]] == ["s1", "s2"]
    assert listed["current_session_id"] == "s1"
    assert listed["managed"] is True
    assert client.get("/api/demo/session").json()["session_id"] == "s1"
    appended = client.post(
        "/api/demo/session/items",
        json={"items": [{"role": "user", "content": "hello"}]},
    )
    assert appended.json()["session_items"][0]["data"]["content"] == "hello"
    assert (
        client.get("/api/demo/session/items").json()["session_items"][0]["data"]["content"] == "s1"
    )
    assert [
        item["data"]["content"]
        for item in client.get("/api/demo/session/items").json()["session_items"]
    ] == ["s1", "saved reply"]

    opened = client.post("/api/demo/sessions/s2/open")
    assert opened.json() == {
        "session_id": "s2",
        "previous_session_id": "s1",
        "managed": True,
    }
    assert client.get("/api/demo/config").json()["session_id"] == "s2"
    assert (
        client.get("/api/demo/session/items").json()["session_items"][0]["data"]["content"] == "s2"
    )


def test_open_session_rejects_another_actor(monkeypatch):
    client = _client(monkeypatch, configured=True, session_id="s1")

    class _ForeignActorClient(_FakeStateClient):
        def get_session(self, session_id):
            return {"session_id": session_id, "actor_id": "bob"}

    monkeypatch.setattr(ui, "_state_client", lambda: _ForeignActorClient())

    response = client.post("/api/demo/sessions/s2/open")
    assert response.status_code == 403
    assert response.json()["detail"] == "Session belongs to another actor."
