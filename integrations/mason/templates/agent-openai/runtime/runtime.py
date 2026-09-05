"""The agent's HTTP surface: a hand-written, SDK-agnostic FastAPI app.

``build_app`` wires the endpoints to two handlers with a generic contract:

    invoke_handler(request: dict, request_auth: RequestAuthContext) -> dict
    stream_handler(request: dict, request_auth: RequestAuthContext) -> AsyncGenerator[dict]

Nothing here is SDK-specific — the agent lives entirely behind those handlers (``agent/agent.py``).
The Databricks Apps ``__Host-databricks-app-router`` cookie is both the replica-affinity key and the
application session id. Clients never send ``session_id`` in the JSON body. Local development uses
an HTTP-only fallback cookie because the Apps router is not present.

TODO: Prefer ``X-Routing-Key`` for API clients once Databricks Apps supports it; until then, use the
documented router cookie for sticky routing.

Endpoints: ``POST /invocations`` (``stream: true`` → SSE ending with ``data: [DONE]``;
``background: true`` → an ``invocation_id`` to poll), ``GET /invocations/{invocation_id}`` to poll a
background run, ``POST /api/session/new`` to rotate the routing session, and ``GET /health``. The
invocation and health routes also have ``/api`` aliases because Databricks Apps accepts programmatic
Bearer-token authentication only on paths under ``/api/``. Each request is wrapped in an MLflow span
for tracing.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing

import mlflow
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from uuid_utils import uuid7

from databricks_mason.runtime import (
    InvocationAuthPolicy,
    MasonRuntimeError,
    RequestAuthContext,
    UserAuthBackgroundUnsupported,
)
from databricks_mason.runtime.background import BackgroundRuns

REQUEST_AUTH_CONTRACT_VERSION = 1

# Request keys that control transport; stripped before the request reaches the handler.
_REQUEST_STREAM_PARAM_KEY = "stream"
_REQUEST_BACKGROUND_PARAM_KEY = "background"
_FORWARDED_ACCESS_TOKEN_HEADER = "x-forwarded-access-token"
_REQUEST_SESSION_ID_HEADER_KEY = (
    "X-Routing-Key"  # carries the session id (generic for Apps + Agents)
)
_TRACE_NAME_TAG = "mlflow.traceName"
_ROUTING_COOKIE = "__Host-databricks-app-router"
_LOCAL_SESSION_COOKIE = "mason-local-session"

# TODO: Replace the Apps routing cookie with X-Routing-Key when Databricks Apps supports it.

InvokeHandler = Callable[[dict, RequestAuthContext], Awaitable[dict]]
StreamHandler = Callable[[dict, RequestAuthContext], AsyncGenerator[dict, None]]


def _sse(data: dict | str) -> str:
    return f"data: {json.dumps(data) if isinstance(data, dict) else data}\n\n"


def _set_trace_name(name: str) -> None:
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(tags={_TRACE_NAME_TAG: name})


def _request_auth(request: Request, auth_policy: InvocationAuthPolicy) -> RequestAuthContext:
    forwarded_token = (
        request.headers.get(_FORWARDED_ACCESS_TOKEN_HEADER)
        if auth_policy.requires_user_credentials
        else None
    )
    return RequestAuthContext.from_forwarded_token(forwarded_token)


def _internal_error_envelope() -> dict:
    return MasonRuntimeError("Agent execution failed.").to_error_envelope()


def rotate_session_cookie(request: Request, response: Response, session_id: str) -> None:
    if request.cookies.get(_ROUTING_COOKIE):
        response.set_cookie(
            _ROUTING_COOKIE,
            session_id,
            secure=True,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.delete_cookie(_LOCAL_SESSION_COOKIE, path="/")
    elif request.cookies.get(_LOCAL_SESSION_COOKIE):
        response.set_cookie(
            _LOCAL_SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            path="/",
        )


def build_app(
    invoke_handler: InvokeHandler,
    stream_handler: StreamHandler,
    auth_policy: InvocationAuthPolicy,
) -> FastAPI:
    """Wire endpoints to handlers under the declared request-credential lifetime policy."""
    app = FastAPI(title="Agent Server")
    runs = BackgroundRuns()

    @app.middleware("http")
    async def bind_session(request: Request, call_next):
        routing_session = request.cookies.get(_ROUTING_COOKIE)
        local_session = request.cookies.get(_LOCAL_SESSION_COOKIE)
        request.state.session_id = routing_session or local_session or str(uuid7())
        response = await call_next(request)
        if not routing_session and not local_session:
            response.set_cookie(
                _LOCAL_SESSION_COOKIE,
                request.state.session_id,
                httponly=True,
                samesite="lax",
            )
        return response

    async def _invoke(request: dict, request_auth: RequestAuthContext) -> dict:
        failure: MasonRuntimeError | None = None
        result: dict = {}
        with mlflow.start_span(name="invoke_handler") as span:
            _set_trace_name("invoke_handler")
            span.set_inputs(request)
            try:
                result = await invoke_handler(request, request_auth)
            except MasonRuntimeError as error:
                failure = error
            except Exception:
                failure = MasonRuntimeError("Agent execution failed.")
            span.set_outputs(failure.to_observability_envelope() if failure is not None else result)
        if failure is not None:
            raise failure from None
        return result

    async def _stream(request: dict, request_auth: RequestAuthContext) -> AsyncGenerator[str, None]:
        with mlflow.start_span(name="stream_handler") as span:
            _set_trace_name("stream_handler")
            span.set_inputs(request)
            trace_chunks: list[dict] = []
            try:
                async with aclosing(stream_handler(request, request_auth)) as stream:
                    async for chunk in stream:
                        trace_chunks.append(chunk)
                        yield _sse(chunk)
                span.set_outputs(trace_chunks)
            except MasonRuntimeError as error:
                error_chunk = error.to_error_envelope()
                trace_chunks.append(error.to_observability_envelope())
                span.set_outputs(trace_chunks)
                yield _sse(error_chunk)
            except Exception:
                error_chunk = _internal_error_envelope()
                trace_chunks.append(error_chunk)
                span.set_outputs(trace_chunks)
                yield _sse(error_chunk)
            yield _sse("[DONE]")

    async def _run_background(
        invocation_id: str,
        request: dict,
        request_auth: RequestAuthContext,
    ) -> None:
        try:
            runs.complete(invocation_id, await _invoke(request, request_auth))
        except MasonRuntimeError as error:
            runs.fail(invocation_id, error.to_error_envelope()["error"])
        except Exception:
            runs.fail(invocation_id, _internal_error_envelope()["error"])

    @app.post("/api/invocations")
    @app.post("/invocations")
    async def invoke(request: Request):
        data = await request.json()
        is_stream = bool(data.pop(_REQUEST_STREAM_PARAM_KEY, False))
        is_background = bool(data.pop(_REQUEST_BACKGROUND_PARAM_KEY, False))
        data.pop("session_id", None)
        for key in tuple(data):
            if key.casefold() == _FORWARDED_ACCESS_TOKEN_HEADER:
                data.pop(key)
        data["session_id"] = request.state.session_id

        if is_background:
            if not auth_policy.allows_background:
                error = UserAuthBackgroundUnsupported()
                return JSONResponse(error.to_error_envelope(), status_code=error.status)
            request_auth = _request_auth(request, auth_policy)
            invocation_id = runs.start()
            # Fire-and-forget; the task updates `runs` when it finishes. Non-durable (in-memory).
            asyncio.create_task(_run_background(invocation_id, data, request_auth))
            return JSONResponse({"id": invocation_id, "status": "in_progress"})
        request_auth = _request_auth(request, auth_policy)
        if is_stream:
            return StreamingResponse(_stream(data, request_auth), media_type="text/event-stream")
        try:
            return JSONResponse(await _invoke(data, request_auth))
        except MasonRuntimeError as error:
            return JSONResponse(error.to_error_envelope(), status_code=error.status)

    @app.get("/api/invocations/{invocation_id}")
    @app.get("/invocations/{invocation_id}")
    async def retrieve(invocation_id: str):
        run = runs.get(invocation_id)
        if run is None:
            return JSONResponse({"error": "unknown invocation id"}, status_code=404)
        if run["status"] == "completed":
            return JSONResponse({"id": invocation_id, "status": "completed", **run["output"]})
        return JSONResponse({"id": invocation_id, "status": run["status"], "error": run["error"]})

    @app.get("/api/health")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/session/new")
    async def new_session(request: Request) -> JSONResponse:
        previous_session_id = request.state.session_id
        session_id = str(uuid7())
        request.state.session_id = session_id
        response = JSONResponse(
            {
                "session_id": session_id,
                "previous_session_id": previous_session_id,
            }
        )
        rotate_session_cookie(request, response, session_id)
        return response

    return app
