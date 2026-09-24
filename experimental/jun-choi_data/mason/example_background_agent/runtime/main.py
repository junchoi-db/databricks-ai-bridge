# ruff: noqa: BLE001 - auditable deployed E2E fixture.

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn

import databricks_mason
from databricks_mason import AgentApp, InvocationContext
from databricks_mason.langgraph.memory import memory_tools
from databricks_mason.langgraph.mcp import _server_from_tool, mcp_client
from databricks_mason.langgraph.session_store import checkpointer, thread_config
from databricks_mason.runtime.auth import InvocationAuthPolicy
from databricks_mason.runtime.tool_manifest import (
    load_tools,
    resolve_memory_store,
    resolve_session_store,
)
from databricks_mason.runtime.workspace import workspace_client

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "fixture.json").read_text())
PROVIDER_BINDINGS = {
    "gmail": "gmail",
    "atlassian": "atlassian",
    "slack": "slack",
}


def _json(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.loads(json.dumps(value, default=str))


def _module_hashes() -> dict[str, str]:
    package = Path(databricks_mason.__file__).parent
    return {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in FIXTURE["module_hashes"]
    }


def _scope_names(client: Any) -> list[str]:
    authorization = client.config.authenticate().get("Authorization", "")
    _, separator, token = authorization.partition(" ")
    if not separator:
        return []
    segments = token.split(".")
    if len(segments) != 3:
        return []
    payload = segments[1] + "=" * (-len(segments[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return []
    scopes = claims.get("scope", claims.get("scp", []))
    return sorted(scopes.split() if isinstance(scopes, str) else scopes)


async def _identity(client: Any) -> dict[str, Any]:
    principal = await asyncio.to_thread(client.current_user.me)
    return {
        "id": principal.id,
        "user_name": principal.user_name,
        "display_name": principal.display_name,
    }


async def _declared_mcp_tools(
    binding_id: str, context: InvocationContext
) -> dict[str, Any]:
    records = {
        record.id: record for record in load_tools(expected_framework="langgraph")
    }
    record = records[binding_id]
    factory = context.request_auth.client_for if context.request_auth else None
    server = _server_from_tool(record, workspace_client_for=factory)
    if server is None:
        raise RuntimeError(f"Could not build MCP server {binding_id!r}")
    client = mcp_client([server], workspace_client_for=factory, tools=(record,))
    return {item.name: item for item in await client.get_tools(server_name=server.name)}


async def _mcp_call(tool_instance: Any, arguments: dict[str, Any]) -> Any:
    return await tool_instance.ainvoke(
        {
            "type": "tool_call",
            "id": str(uuid4()),
            "name": tool_instance.name,
            "args": arguments,
        }
    )


def _strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            found.extend(_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_strings(child))
    elif isinstance(value, str):
        found.append(value)
    return found


def _list_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, list):
                counts[str(key)] = len(child)
            counts.update(_list_counts(child))
    elif isinstance(value, list):
        for child in value:
            counts.update(_list_counts(child))
    return counts


def _tool_summary(value: Any) -> dict[str, Any]:
    normalized = _json(value)
    serialized = json.dumps(normalized, sort_keys=True, default=str)
    lowered = serialized.lower()
    parsed_values: list[Any] = [normalized]
    for text in _strings(normalized):
        try:
            parsed_values.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            continue
    counts: dict[str, int] = {}
    for parsed in parsed_values:
        counts.update(_list_counts(parsed))
    challenge = any(
        marker in lowered
        for marker in (
            "please login",
            "please log in",
            "credential_missing",
            "per-user credential",
            "/explore/data/mcp-services/",
        )
    )
    is_error = any(
        marker in lowered
        for marker in (
            '"iserror": true',
            '"is_error": true',
            '"status": "error"',
            "configured mcp tool failed",
        )
    )
    return {
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "bytes": len(serialized.encode()),
        "challenge": challenge,
        "is_error": is_error,
        "list_counts": counts,
        "type": type(value).__name__,
    }


async def _provider(
    provider: str, mode: str, context: InvocationContext
) -> dict[str, Any]:
    binding_id = f"{PROVIDER_BINDINGS[provider]}_{mode}"
    tools = await _declared_mcp_tools(binding_id, context)
    names = sorted(tools)
    call = None
    if provider == "gmail" and "gmail_search" in tools:
        call = _tool_summary(
            await _mcp_call(
                tools["gmail_search"],
                {
                    "query": f"rfc822msgid:<{FIXTURE['provider_nonce']}@example.invalid>",
                    "max_results": 1,
                },
            )
        )
    elif provider == "atlassian" and "getAccessibleAtlassianResources" in tools:
        call = _tool_summary(
            await _mcp_call(tools["getAccessibleAtlassianResources"], {})
        )
    elif provider == "slack" and "slack_read_user_profile" in tools:
        call = _tool_summary(
            await _mcp_call(
                tools["slack_read_user_profile"],
                {"response_format": "concise"},
            )
        )
    return {
        "binding_id": binding_id,
        "discovered": names,
        "tool_count": len(names),
        "call": call,
        "semantic_call_performed": call is not None,
        "gmail_send_present": any("send" in name.lower() for name in names),
        "gmail_draft_present": any("draft" in name.lower() for name in names),
    }


async def _memory_write(probe: dict[str, str]) -> dict[str, Any]:
    tools = {tool.name: tool for tool in memory_tools(probe["actor"])}
    if set(tools) != {"recall", "remember"}:
        raise RuntimeError(
            f"agent.toml memory binding did not expose tools: {sorted(tools)}"
        )
    stored = await tools["remember"].ainvoke(
        {"fact": probe["memory_fact"], "topic": probe["memory_topic"]}
    )
    return {
        "configured_store": resolve_memory_store(),
        "tool_names": sorted(tools),
        "remember_result": _json(stored),
    }


async def _memory_read(probe: dict[str, str]) -> dict[str, Any]:
    tools = {tool.name: tool for tool in memory_tools(probe["actor"])}
    attempts = []
    recalled = ""
    for attempt in range(1, 13):
        recalled = str(await tools["recall"].ainvoke({"query": probe["memory_query"]}))
        matched = probe["marker"] in recalled
        attempts.append(
            {"attempt": attempt, "matched": matched, "response": recalled[:4000]}
        )
        if matched:
            break
        await asyncio.sleep(5)
    return {
        "configured_store": resolve_memory_store(),
        "tool_names": sorted(tools),
        "matched": probe["marker"] in recalled,
        "recall": recalled[:4000],
        "attempts": attempts,
    }


async def _session_write(probe: dict[str, str]) -> dict[str, Any]:
    saver = checkpointer()
    config = thread_config(probe["thread_id"], probe["actor"])
    version = saver.get_next_version(None, "bugbash_probe")
    checkpoint = {
        "v": 1,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "id": probe["checkpoint_id"],
        "channel_values": {"bugbash_probe": probe["marker"]},
        "channel_versions": {"bugbash_probe": version},
        "versions_seen": {},
        "updated_channels": ["bugbash_probe"],
    }
    saved_config = await saver.aput(
        config,
        checkpoint,
        {"source": "input", "step": 0, "parents": {}, "writes": {}},
        {"bugbash_probe": version},
    )
    loaded = await saver.aget_tuple(config)
    retrieved = (
        loaded.checkpoint.get("channel_values", {}).get("bugbash_probe")
        if loaded
        else None
    )
    return {
        "configured_store": resolve_session_store(),
        "saver_type": type(saver).__name__,
        "saved_config": _json(saved_config),
        "checkpoint_id": loaded.checkpoint.get("id") if loaded else None,
        "retrieved_marker": retrieved,
        "matched": retrieved == probe["marker"],
    }


async def _session_read(probe: dict[str, str]) -> dict[str, Any]:
    saver = checkpointer()
    loaded = await saver.aget_tuple(thread_config(probe["thread_id"], probe["actor"]))
    retrieved = (
        loaded.checkpoint.get("channel_values", {}).get("bugbash_probe")
        if loaded
        else None
    )
    return {
        "configured_store": resolve_session_store(),
        "saver_type": type(saver).__name__,
        "checkpoint_id": loaded.checkpoint.get("id") if loaded else None,
        "retrieved_marker": retrieved,
        "matched": retrieved == probe["marker"],
    }


async def _agent_stores(action: str, probe: dict[str, str]) -> dict[str, Any]:
    app_client = workspace_client()
    if action == "write":
        memory = await _memory_write(probe)
        session = await _session_write(probe)
    elif action == "read":
        memory = await _memory_read(probe)
        session = await _session_read(probe)
    else:
        raise ValueError("store action must be write or read")
    return {
        "contract": "agent.toml -> app.yaml env -> Mason runtime helpers -> App SP",
        "action": action,
        "storage_identity": await _identity(app_client),
        "storage_scope_names": _scope_names(app_client),
        "memory": memory,
        "session": session,
        "passed": bool(memory.get("matched", True) and session.get("matched")),
    }


app = AgentApp(auth_policy=InvocationAuthPolicy.from_manifest())


@app.invoke
async def invoke(payload: Any, context: InvocationContext) -> dict[str, Any]:
    if not isinstance(payload, dict) or not {"operation", "auth"} <= set(payload):
        raise ValueError("input must contain operation and auth")
    if set(payload) - {"operation", "auth", "probe"}:
        raise ValueError("input contains unsupported fields")
    operation = payload["operation"]
    mode = payload["auth"]
    if mode not in {"user", "app"}:
        raise ValueError("auth must be user or app")
    if context.request_auth is None:
        raise RuntimeError("request auth context is required")
    client = context.request_auth.client_for(mode)
    identity = await _identity(client)
    started = asyncio.get_running_loop().time()
    output = None
    error = None
    try:
        if operation.startswith("provider_"):
            provider = operation.removeprefix("provider_")
            if provider not in PROVIDER_BINDINGS:
                raise ValueError("unknown provider")
            output = await _provider(provider, mode, context)
        elif operation in {"agent_stores_write", "agent_stores_read"}:
            probe = payload.get("probe")
            if not isinstance(probe, dict):
                raise ValueError("store operations require a probe object")
            output = await _agent_stores(operation.rsplit("_", 1)[-1], probe)
        elif operation == "identity":
            output = {"identity_only": True}
        else:
            raise ValueError("unknown operation")
    except Exception as caught:
        error = {
            "type": type(caught).__name__,
            "message": str(caught)[:2000],
            "status_code": getattr(caught, "status_code", None),
            "error_code": getattr(caught, "error_code", None),
        }
    result = {
        "proof": "mason-agent-toml-anshul-bugbash-rerun",
        "commit": FIXTURE["commit"],
        "wheel_sha256": FIXTURE["wheel_sha256"],
        "operation": operation,
        "auth": mode,
        "identity": identity,
        "scope_names": _scope_names(client),
        "output": output,
        "error": error,
        "invocation_id": context.invocation_id,
        "attempt": context.attempt,
        "seconds": round(asyncio.get_running_loop().time() - started, 3),
        "installed_module_hashes": _module_hashes(),
    }
    print(
        "[mason-agent-toml-bugbash-freshness] " + json.dumps(result, sort_keys=True),
        flush=True,
    )
    await context.emit({"type": "bugbash.result", "result": result})
    return result


def main() -> None:
    print(
        json.dumps(
            {
                "marker": "mason-agent-toml-bugbash-startup",
                "commit": FIXTURE["commit"],
                "wheel_sha256": FIXTURE["wheel_sha256"],
                "module_hashes": _module_hashes(),
                "app_name_present": bool(os.environ.get("DATABRICKS_APP_NAME")),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
