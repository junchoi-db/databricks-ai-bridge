import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from runtime import runtime as runtime_module
from runtime.runtime import build_app

from databricks_mason.runtime import (
    InvocationAuthPolicy,
    MCPAuthorizationRequired,
    MCPPermissionDenied,
    MissingUserAuthorization,
    RequestAuthContext,
)


async def _invoke(request: dict, request_auth: RequestAuthContext) -> dict:
    del request_auth
    return request


async def _stream(request: dict, request_auth: RequestAuthContext) -> AsyncGenerator[dict, None]:
    del request_auth
    yield request


class _RecordingSpan:
    def __init__(self, name: str, records: dict[str, dict[str, Any]]) -> None:
        self._record = records.setdefault(name, {})

    def __enter__(self) -> "_RecordingSpan":
        self._record["entered"] = True
        return self

    def __exit__(self, *args: object) -> None:
        self._record["exited"] = True
        self._record["exit_args"] = args

    def set_inputs(self, value: Any) -> None:
        self._record["inputs"] = value

    def set_outputs(self, value: Any) -> None:
        self._record["outputs"] = value


def _record_spans(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        runtime_module.mlflow,
        "start_span",
        lambda *, name: _RecordingSpan(name, records),
    )
    monkeypatch.setattr(runtime_module.mlflow, "get_current_active_span", lambda: None)
    return records


def _sse_frames(body: str) -> list[dict | str]:
    frames: list[dict | str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        frames.append(payload if payload == "[DONE]" else json.loads(payload))
    return frames


class _CloseTrackingIterator:
    def __init__(self) -> None:
        self.closed = False
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return {"type": "delta", "content": "hello", "id": "item-1"}

    async def aclose(self) -> None:
        self.closed = True


def test_invocation_routes_support_local_and_deployed_app_auth_paths() -> None:
    paths = build_app(_invoke, _stream, InvocationAuthPolicy(False)).openapi()["paths"]

    assert paths["/invocations"]["post"]
    assert paths["/api/invocations"]["post"]
    assert paths["/invocations/{invocation_id}"]["get"]
    assert paths["/api/invocations/{invocation_id}"]["get"]
    assert paths["/health"]["get"]
    assert paths["/api/health"]["get"]


@pytest.mark.asyncio
async def test_sync_passes_auth_separately_and_traces_only_sanitized_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    spans = _record_spans(monkeypatch)
    captured: dict[str, Any] = {}

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        captured["request"] = request
        captured["request_auth"] = request_auth
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        response = await client.post(
            "/invocations",
            headers={
                "cookie": "__Host-databricks-app-router=routing-session",
                "x-forwarded-access-token": "header-secret",
            },
            json={
                "input": [{"role": "user", "content": "hello"}],
                "session_id": "body-session",
                "X-Forwarded-Access-Token": "body-secret",
            },
        )

    assert response.status_code == 200
    assert captured["request"] == {
        "input": [{"role": "user", "content": "hello"}],
        "session_id": "routing-session",
    }
    assert isinstance(captured["request_auth"], RequestAuthContext)
    assert captured["request_auth"].state_key("routing-session") != "routing-session"
    assert spans["invoke_handler"]["inputs"] == captured["request"]
    assert spans["invoke_handler"]["outputs"] == response.json()
    assert "header-secret" not in repr(captured)
    assert "header-secret" not in repr(spans)
    assert "body-secret" not in repr(captured)
    assert "body-secret" not in repr(spans)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (MissingUserAuthorization("drive"), 401),
        (MCPPermissionDenied("drive"), 403),
        (
            MCPAuthorizationRequired(
                "drive",
                data={
                    "elicitations": [
                        {
                            "mode": "url",
                            "message": "Additional authorization is required.",
                            "url": "https://workspace.example/explore/data/mcp-services/drive",
                            "elicitationId": "auth-1",
                        }
                    ]
                },
            ),
            401,
        ),
    ],
)
@pytest.mark.asyncio
async def test_sync_returns_typed_runtime_error_envelope(error, status: int) -> None:
    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        del request, request_auth
        raise error

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": []})

    assert response.status_code == status
    assert response.json() == error.to_error_envelope()


@pytest.mark.asyncio
async def test_sync_provider_challenge_is_full_for_caller_but_redacted_from_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans = _record_spans(monkeypatch)
    authorization_url = (
        "https://workspace.example/explore/data/mcp-services/system/ai/google_drive"
        "?oauth_state=sensitive-query"
    )
    error = MCPAuthorizationRequired(
        "drive",
        data={
            "elicitations": [
                {
                    "mode": "url",
                    "message": "Connect Google Drive.",
                    "url": authorization_url,
                    "elicitationId": "drive-oauth",
                }
            ]
        },
    )

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        del request, request_auth
        raise error

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": []})

    assert response.json() == error.to_error_envelope()
    assert authorization_url in response.text
    assert spans["invoke_handler"]["outputs"] == error.to_observability_envelope()
    assert "sensitive-query" not in repr(spans)


@pytest.mark.asyncio
async def test_sync_unknown_error_is_sanitized_before_span_and_asgi_boundary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spans = _record_spans(monkeypatch)
    sentinel = "unknown-sync-secret"

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        del request, request_auth
        sensitive_local = sentinel
        raise RuntimeError(f"handler failed with {sensitive_local}")

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/invocations", json={"input": []})

    safe_error = {
        "error": {
            "code": "MASON_RUNTIME_ERROR",
            "message": "Agent execution failed.",
        }
    }
    assert response.status_code == 500
    assert response.json() == safe_error
    assert spans["invoke_handler"]["outputs"] == safe_error
    assert spans["invoke_handler"]["exit_args"] == (None, None, None)
    assert sentinel not in repr(spans)
    assert sentinel not in caplog.text
    assert sentinel not in response.text


@pytest.mark.asyncio
async def test_sse_keeps_auth_for_iterator_lifetime_and_emits_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    spans = _record_spans(monkeypatch)
    lifetime: dict[str, Any] = {}

    async def stream_handler(
        request: dict, request_auth: RequestAuthContext
    ) -> AsyncGenerator[dict, None]:
        lifetime["entered"] = request_auth
        try:
            yield {"type": "delta", "content": "hello", "id": "item-1"}
            raise MCPPermissionDenied("drive")
        finally:
            lifetime["closed"] = request_auth
            lifetime["state_key"] = request_auth.state_key(request["session_id"])

    app = build_app(_invoke, stream_handler, InvocationAuthPolicy(True))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        response = await client.post(
            "/invocations",
            headers={
                "cookie": "__Host-databricks-app-router=routing-session",
                "x-forwarded-access-token": "stream-secret",
            },
            json={"input": [], "stream": True},
        )

    assert response.status_code == 200
    assert _sse_frames(response.text) == [
        {"type": "delta", "content": "hello", "id": "item-1"},
        MCPPermissionDenied("drive").to_error_envelope(),
        "[DONE]",
    ]
    assert lifetime["entered"] is lifetime["closed"]
    assert lifetime["state_key"] != "routing-session"
    assert spans["stream_handler"]["exited"] is True
    assert "stream-secret" not in response.text
    assert "stream-secret" not in repr(spans)


@pytest.mark.asyncio
async def test_sse_provider_challenge_is_full_for_caller_but_redacted_from_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans = _record_spans(monkeypatch)
    authorization_url = (
        "https://workspace.example/explore/data/mcp-services/system/ai/google_drive"
        "?oauth_state=sensitive-query"
    )
    error = MCPAuthorizationRequired(
        "drive",
        data={
            "elicitations": [
                {
                    "mode": "url",
                    "message": "Connect Google Drive.",
                    "url": authorization_url,
                    "elicitationId": "drive-oauth",
                }
            ]
        },
    )

    async def stream_handler(
        request: dict, request_auth: RequestAuthContext
    ) -> AsyncGenerator[dict, None]:
        del request, request_auth
        if False:
            yield {}
        raise error

    app = build_app(_invoke, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": [], "stream": True})

    assert _sse_frames(response.text) == [error.to_error_envelope(), "[DONE]"]
    assert authorization_url in response.text
    assert spans["stream_handler"]["outputs"] == [error.to_observability_envelope()]
    assert "sensitive-query" not in repr(spans)


@pytest.mark.asyncio
async def test_sse_close_closes_handler_iterator_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_spans(monkeypatch)
    handler_iterator = _CloseTrackingIterator()

    def stream_handler(
        request: dict, request_auth: RequestAuthContext
    ) -> AsyncGenerator[dict, None]:
        del request, request_auth
        return cast(AsyncGenerator[dict, None], handler_iterator)

    request_body = json.dumps({"input": [], "stream": True}).encode()

    async def receive() -> dict:
        return {"type": "http.request", "body": request_body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/invocations", "headers": []},
        receive,
    )
    request.state.session_id = "routing-session"
    app = build_app(_invoke, stream_handler, InvocationAuthPolicy(False))
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/invocations"
    )

    response = await route.endpoint(request)
    assert isinstance(response, StreamingResponse)
    body_iterator = cast(AsyncGenerator[str, None], response.body_iterator)
    await anext(body_iterator)
    await body_iterator.aclose()

    assert handler_iterator.closed is True


@pytest.mark.asyncio
async def test_sse_never_stringifies_unsafe_exception() -> None:
    async def stream_handler(
        request: dict, request_auth: RequestAuthContext
    ) -> AsyncGenerator[dict, None]:
        del request, request_auth
        if False:
            yield {}
        raise RuntimeError("unsafe-exception-secret")

    app = build_app(_invoke, stream_handler, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": [], "stream": True})

    assert _sse_frames(response.text) == [
        {
            "error": {
                "code": "MASON_RUNTIME_ERROR",
                "message": "Agent execution failed.",
            }
        },
        "[DONE]",
    ]
    assert "unsafe-exception-secret" not in response.text


@pytest.mark.asyncio
async def test_user_auth_background_rejected_before_run_or_task_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_start(self) -> str:
        raise AssertionError("background invocation id must not be created")

    monkeypatch.setattr(runtime_module.BackgroundRuns, "start", forbidden_start)

    def forbidden_create_task(coroutine):
        coroutine.close()
        raise AssertionError("background task must not be created")

    monkeypatch.setattr(runtime_module.asyncio, "create_task", forbidden_create_task)

    async def forbidden_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        raise AssertionError("background handler must not run")

    app = build_app(forbidden_handler, _stream, InvocationAuthPolicy(True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/invocations",
            headers={"x-forwarded-access-token": "background-secret"},
            json={"input": [], "background": True},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            "message": "Background execution is not supported for user-authenticated MCP integrations.",
        }
    }
    assert "background-secret" not in response.text


@pytest.mark.asyncio
async def test_app_background_discards_forwarded_token_before_task_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    finished = asyncio.Event()
    captured: dict[str, Any] = {}

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        captured["request"] = request
        captured["state_key"] = request_auth.state_key(request["session_id"])
        captured["request_auth"] = request_auth
        finished.set()
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://testserver"
    ) as client:
        response = await client.post(
            "/invocations",
            headers={
                "cookie": "__Host-databricks-app-router=routing-session",
                "x-forwarded-access-token": "discarded-secret",
            },
            json={"input": [], "background": True},
        )
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        completed = await client.get(f"/invocations/{response.json()['id']}")

    assert response.status_code == 200
    assert completed.json() == {
        "id": response.json()["id"],
        "status": "completed",
        "output": [],
        "session_id": "routing-session",
    }
    assert captured["request"] == {"input": [], "session_id": "routing-session"}
    assert captured["state_key"] == "routing-session"
    assert "discarded-secret" not in repr(captured)


@pytest.mark.asyncio
async def test_app_background_preserves_typed_runtime_error_envelope() -> None:
    handler_started = asyncio.Event()
    error = MCPPermissionDenied("app-sandbox")

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        del request, request_auth
        handler_started.set()
        raise error

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": [], "background": True})
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        for _ in range(10):
            polled = await client.get(f"/invocations/{response.json()['id']}")
            if polled.json()["status"] == "failed":
                break
            await asyncio.sleep(0)

    assert polled.json() == {
        "id": response.json()["id"],
        "status": "failed",
        "error": error.to_error_envelope()["error"],
    }


@pytest.mark.asyncio
async def test_background_unknown_error_is_sanitized_before_span_and_poll_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spans = _record_spans(monkeypatch)
    sentinel = "unknown-background-secret"
    handler_started = asyncio.Event()

    async def invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict:
        del request, request_auth
        sensitive_local = sentinel
        handler_started.set()
        raise RuntimeError(f"background handler failed with {sensitive_local}")

    app = build_app(invoke_handler, _stream, InvocationAuthPolicy(False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/invocations", json={"input": [], "background": True})
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        for _ in range(10):
            polled = await client.get(f"/invocations/{response.json()['id']}")
            if polled.json()["status"] == "failed":
                break
            await asyncio.sleep(0)

    safe_error = {
        "error": {
            "code": "MASON_RUNTIME_ERROR",
            "message": "Agent execution failed.",
        }
    }
    assert polled.json() == {
        "id": response.json()["id"],
        "status": "failed",
        "error": safe_error["error"],
    }
    assert spans["invoke_handler"]["outputs"] == safe_error
    assert spans["invoke_handler"]["exit_args"] == (None, None, None)
    assert sentinel not in repr(spans)
    assert sentinel not in caplog.text
    assert sentinel not in polled.text
