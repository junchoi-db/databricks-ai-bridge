"""Browser UI and managed-state demo controls for a Mason agent project."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from runtime.runtime import rotate_session_cookie

from databricks_mason import workspace_client
from databricks_mason.runtime import (
    InvocationAuthPolicy,
    MissingUserAuthorization,
    RequestAuthContext,
    is_deployed_app,
)

_UI_ROOT = Path(__file__).resolve().parent.parent / "ui"
_INSTANCE_ID = uuid.uuid4().hex[:12]  # identifies this process in the UI
_MEMORY_STORE_ENV = "AGENT_MEMORY_STORE"
_MEMORY_ACTOR_ENV = "AGENT_MEMORY_ACTOR_ID"
_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"
_AGENTS_API = "/api/agents/v1"
_FORWARDED_ACCESS_TOKEN_HEADER = "x-forwarded-access-token"
_MANAGED_STATE_ERROR_DETAIL = "Managed state request failed."
_MESSAGE_ROLES = {
    "ai",
    "assistant",
    "developer",
    "function",
    "human",
    "human_decision",
    "system",
    "tool",
    "user",
}


class MemoryEntryRequest(BaseModel):
    path: str = Field(min_length=1, pattern=r"^/")
    content: str = Field(min_length=1)
    description: str | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class SessionItemsRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1)


def _memory_store() -> str:
    return os.getenv(_MEMORY_STORE_ENV, "").strip().strip("/")


def _memory_actor() -> str:
    return os.getenv(_MEMORY_ACTOR_ENV, "agent")


def _session_store() -> str:
    return os.getenv(_SESSION_STORE_ENV, "").strip()


def _session_actor() -> str:
    return os.getenv(_SESSION_ACTOR_ENV) or _memory_actor()


class _ManagedStateClient:
    def __init__(self) -> None:
        self._workspace = workspace_client()

    def _do(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict:
        result = self._workspace.api_client.do(method, path, query=query, body=body)
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Expected an object response from {path}, got {type(result).__name__}"
            )
        return result

    def create_memory_entry(self, request: MemoryEntryRequest, session_id: str) -> dict:
        body = {
            "actor_id": _memory_actor(),
            "path": request.path,
            "content": request.content,
            "session_id": session_id,
        }
        if request.description:
            body["description"] = request.description
        return self._do("POST", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", body=body)

    def list_memory_entries(self, path_prefix: str | None = None) -> dict:
        query = {"actor_id": _memory_actor(), "page_size": 100}
        if path_prefix:
            query["path_prefix"] = path_prefix
        return self._do(
            "GET", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", query=query
        )

    def search_memory_entries(self, request: MemorySearchRequest) -> dict:
        return self._do(
            "POST",
            f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries:search",
            body={
                "actor_id": _memory_actor(),
                "query": request.query,
                "limit": request.limit,
            },
        )

    def ensure_session(self, session_id: str) -> dict:
        try:
            return self._do(
                "POST",
                f"{_AGENTS_API}/session-stores/{_session_store()}/sessions",
                query={"session_id": session_id},
                body={
                    "actor_id": _session_actor(),
                    "metadata": {"client": "mason-demo-ui"},
                },
            )
        except Exception as exc:
            code = str(getattr(exc, "error_code", "")).upper()
            already_exists = code in {"ALREADY_EXISTS", "RESOURCE_ALREADY_EXISTS"}
            if not already_exists and "already exists" not in str(exc).lower():
                raise
            return self._do(
                "GET",
                f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}",
            )

    def get_session(self, session_id: str) -> dict:
        return self._do(
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}",
        )

    def list_sessions(self) -> dict:
        return self._do(
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions",
            query={
                "filter": f"actor_id = {json.dumps(_session_actor())}",
                "order_by": "last_activity_time desc",
                "page_size": 50,
            },
        )

    def append_session_items(self, session_id: str, items: list[dict[str, Any]]) -> dict:
        return self._do(
            "POST",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}/items:append",
            body={"items": [{"data": item} for item in items]},
        )

    def list_session_items(self, session_id: str) -> dict:
        return self._do(
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}/items",
            query={"order_by": "create_time asc", "page_size": 100},
        )


@lru_cache(maxsize=1)
def _state_client() -> _ManagedStateClient:
    return _ManagedStateClient()


async def _managed_call(operation, *args):
    try:
        return await asyncio.to_thread(operation, *args)
    except Exception:
        pass
    raise HTTPException(status_code=502, detail=_MANAGED_STATE_ERROR_DETAIL) from None


def _require_memory() -> None:
    if not _memory_store():
        raise HTTPException(
            status_code=503,
            detail=f"Set {_MEMORY_STORE_ENV} by deploying with --memory.",
        )


def _require_session() -> None:
    if not _session_store():
        raise HTTPException(
            status_code=503,
            detail=f"Set {_SESSION_STORE_ENV} by deploying with --session.",
        )


async def _checkpoint_history(
    routing_session: str,
    state_session_id: str,
    request_auth: RequestAuthContext,
) -> dict[str, Any]:
    from agent.agent import create_agent_graph

    from databricks_mason.langgraph.session_store import thread_config

    graph = await create_agent_graph(request_auth)
    snapshot = await graph.aget_state(thread_config(state_session_id))
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    items = []
    for index, message in enumerate(values.get("messages", [])):
        data = message.model_dump() if hasattr(message, "model_dump") else message
        items.append(
            {
                "item_id": str(getattr(message, "id", None) or index),
                "data": data if isinstance(data, dict) else {"content": str(data)},
            }
        )
    interrupts = [
        {"id": interrupt.id, "value": interrupt.value}
        for task in getattr(snapshot, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
    ]
    return {"session_id": routing_session, "session_items": items, "interrupts": interrupts}


def _request_state(
    request: Request,
    auth_policy: InvocationAuthPolicy,
) -> tuple[str, str, RequestAuthContext]:
    routing_session = str(request.state.session_id)
    forwarded_token = (
        request.headers.get(_FORWARDED_ACCESS_TOKEN_HEADER)
        if auth_policy.requires_user_credentials
        else None
    )
    if _deployed_user_auth(auth_policy) and not (forwarded_token or "").strip():
        raise MissingUserAuthorization()
    request_auth = RequestAuthContext.from_forwarded_token(forwarded_token)
    return routing_session, request_auth.state_key(routing_session), request_auth


def _deployed_user_auth(auth_policy: InvocationAuthPolicy) -> bool:
    return auth_policy.requires_user_credentials and is_deployed_app()


def _externalize_state_ids(
    value: Any,
    state_session_id: str,
    routing_session: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(state_session_id, routing_session)
    if isinstance(value, dict):
        return {
            _externalize_state_ids(key, state_session_id, routing_session): _externalize_state_ids(
                item, state_session_id, routing_session
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_externalize_state_ids(item, state_session_id, routing_session) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _externalize_state_ids(item, state_session_id, routing_session) for item in value
        )
    return value


def _external_session(result: dict, state_session_id: str, routing_session: str) -> dict:
    external = _externalize_state_ids(result, state_session_id, routing_session)
    return {**external, "session_id": routing_session}


def _chat_sessions(result: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = []
    for session in result.get("sessions", []):
        if not isinstance(session, dict):
            continue
        metadata = session.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("public_session_id"):
            continue
        sessions.append(session)
    return sessions


def _chat_session_items(result: dict[str, Any]) -> dict[str, Any]:
    items = []
    for item in result.get("session_items", []):
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict) or data.get("event_type") or "content" not in data:
            continue
        role = str(data.get("role") or data.get("type") or "").lower()
        if role in _MESSAGE_ROLES:
            items.append(item)
    return {**result, "session_items": items}


def install_ui(app: FastAPI, auth_policy: InvocationAuthPolicy) -> None:
    """Mount the Mason demo UI and its runtime control endpoints."""
    app.mount("/ui-assets", StaticFiles(directory=_UI_ROOT), name="mason-demo-ui-assets")

    @app.exception_handler(MissingUserAuthorization)
    async def missing_user_authorization(
        _request: Request, error: MissingUserAuthorization
    ) -> JSONResponse:
        return JSONResponse(error.to_error_envelope(), status_code=error.status)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_UI_ROOT / "index.html")

    @app.get("/api/demo/config", include_in_schema=False)
    async def demo_config(request: Request) -> dict:
        viewer = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-forwarded-user")
            or "Local developer"
        )
        memory_store = _memory_store()
        session_store = _session_store()
        return {
            "session_id": request.state.session_id,
            "instance_id": _INSTANCE_ID,
            "viewer": viewer,
            "deployed": is_deployed_app(),
            "streaming": {"enabled": True, "transport": "Server-sent events"},
            "background": {"enabled": auth_policy.allows_background, "durable": False},
            "session": {
                "durable": bool(session_store),
                "managed": bool(session_store),
                "history": True,
                "mode": "Managed Session Store" if session_store else "In-process checkpointer",
                "store": session_store or None,
                "actor": _session_actor(),
            },
            "memory": {
                "enabled": bool(memory_store),
                "store": f"memory-stores/{memory_store}" if memory_store else None,
                "actor": _memory_actor(),
            },
        }

    @app.post("/api/demo/memory/entries", include_in_schema=False)
    async def create_memory_entry(request: Request, payload: MemoryEntryRequest) -> dict:
        _require_memory()
        return await _managed_call(
            _state_client().create_memory_entry, payload, request.state.session_id
        )

    @app.get("/api/demo/memory/entries", include_in_schema=False)
    async def list_memory_entries(
        path_prefix: str | None = Query(default=None),
    ) -> dict:
        _require_memory()
        return await _managed_call(_state_client().list_memory_entries, path_prefix)

    @app.post("/api/demo/memory/search", include_in_schema=False)
    async def search_memory_entries(request: MemorySearchRequest) -> dict:
        _require_memory()
        return await _managed_call(_state_client().search_memory_entries, request)

    @app.post("/api/demo/sessions", include_in_schema=False)
    async def ensure_session(request: Request) -> dict:
        _require_session()
        routing_session, state_session_id, _ = _request_state(request, auth_policy)
        result = await _managed_call(_state_client().ensure_session, state_session_id)
        return _external_session(result, state_session_id, routing_session)

    @app.get("/api/demo/sessions", include_in_schema=False)
    async def list_sessions(request: Request) -> dict:
        session_id = str(request.state.session_id)
        if not _session_store():
            return {
                "sessions": [
                    {
                        "session_id": session_id,
                        "actor_id": _session_actor(),
                        "metadata": {"client": "mason-demo-ui-local"},
                    }
                ],
                "current_session_id": session_id,
                "managed": False,
            }
        if _deployed_user_auth(auth_policy):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Managed session listing is unavailable for user-authenticated integrations."
                ),
            )
        result = await _managed_call(_state_client().list_sessions)
        return {
            **result,
            "sessions": _chat_sessions(result),
            "current_session_id": session_id,
            "managed": True,
        }

    @app.post("/api/demo/sessions/{session_id}/open", include_in_schema=False)
    async def open_session(request: Request, session_id: str) -> JSONResponse:
        _require_session()
        if _deployed_user_auth(auth_policy):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Managed session switching is unavailable for user-authenticated integrations."
                ),
            )
        session = await _managed_call(_state_client().get_session, session_id)
        if session.get("actor_id") != _session_actor():
            raise HTTPException(status_code=403, detail="Session belongs to another actor.")
        previous_session_id = request.state.session_id
        request.state.session_id = session_id
        response = JSONResponse(
            {
                "session_id": session_id,
                "previous_session_id": previous_session_id,
                "managed": True,
            }
        )
        rotate_session_cookie(request, response, session_id)
        return response

    @app.get("/api/demo/session", include_in_schema=False)
    async def get_session(request: Request) -> dict:
        _require_session()
        routing_session, state_session_id, _ = _request_state(request, auth_policy)
        result = await _managed_call(_state_client().get_session, state_session_id)
        return _external_session(result, state_session_id, routing_session)

    @app.post("/api/demo/session/items", include_in_schema=False)
    async def append_session_items(request: Request, payload: SessionItemsRequest) -> dict:
        _require_session()
        routing_session, state_session_id, _ = _request_state(request, auth_policy)
        result = await _managed_call(
            _state_client().append_session_items,
            state_session_id,
            payload.items,
        )
        return _externalize_state_ids(result, state_session_id, routing_session)

    @app.get("/api/demo/session/items", include_in_schema=False)
    async def list_session_items(request: Request) -> dict:
        routing_session, state_session_id, request_auth = _request_state(request, auth_policy)
        if _session_store():
            result = await _managed_call(_state_client().list_session_items, state_session_id)
            result = _chat_session_items(result)
            return _externalize_state_ids(result, state_session_id, routing_session)
        return await _checkpoint_history(routing_session, state_session_id, request_auth)
