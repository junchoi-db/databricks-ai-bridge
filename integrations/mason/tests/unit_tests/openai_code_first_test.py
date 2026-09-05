"""Tests for attaching explicit integrations to an OpenAI Agents SDK agent."""

from __future__ import annotations

import asyncio
import importlib
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agents import Agent, UserError
from agents.mcp import MCPServerStreamableHttp, MCPUtil
from agents.tool_context import ToolContext
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import Unauthenticated
from databricks_openai.agents import McpServer
from mcp import McpError
from mcp.types import CallToolResult, ErrorData, TextContent, Tool

import databricks_mason.openai as openai_adapter
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction
from databricks_mason.openai import bind_tools
from databricks_mason.openai import mcp as openai_mcp

_AUTHORIZATION_URL = (
    "https://workspace.example.com/explore/data/mcp-services/system/ai/google_drive"
    "?oauth_state=sensitive-query"
)


def test_package_example_threads_request_auth_client_resolver() -> None:
    assert openai_adapter.__doc__ is not None
    assert "workspace_client_for=request_auth.client_for" in openai_adapter.__doc__


def _typed_authorization_error() -> McpError:
    return McpError(
        ErrorData(
            code=-32042,
            message=f"Unsafe upstream message containing {_AUTHORIZATION_URL}",
            data={
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Connect Google Drive",
                        "url": _AUTHORIZATION_URL,
                        "elicitationId": "google-drive-oauth",
                        "secret": "must-not-be-retained",
                    }
                ]
            },
        )
    )


def _flattened_authorization_result(url: str) -> CallToolResult:
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=f"Google Drive authorization required. Open {url}")],
    )


def test_bind_tools_owns_new_server_lifecycle_and_returns_an_isolated_clone(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        return []

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)

    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com/")),
    )
    existing_tool = cast(Any, object())
    existing_server = McpServer(
        url="https://custom.example.com/mcp",
        name="existing",
        workspace_client=workspace_client,
    )
    existing_handoff = cast(Any, object())
    existing_input_guardrail = cast(Any, object())
    existing_output_guardrail = cast(Any, object())
    original = Agent(
        name="claims",
        tools=[existing_tool],
        mcp_servers=[existing_server],
        handoffs=cast(Any, [existing_handoff]),
        input_guardrails=[existing_input_guardrail],
        output_guardrails=[existing_output_guardrail],
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                original,
                [MCPService(id="claims_mcp", service="main.claims.service")],
                stack=stack,
                workspace_client=workspace_client,
            )

            assert lifecycle == [
                ("connect", "claims_mcp"),
                ("list", "claims_mcp"),
            ]
            assert bound is not original
            assert bound.tools == [existing_tool]
            assert bound.mcp_servers[0] is existing_server
            assert len(bound.mcp_servers) == 2
            generated = bound.mcp_servers[1]
            assert isinstance(generated, McpServer)
            assert generated.params["url"] == (
                "https://workspace.example.com/ai-gateway/mcp-services/main.claims.service"
            )
            assert generated.workspace_client is workspace_client

            for field in (
                "tools",
                "mcp_servers",
                "handoffs",
                "input_guardrails",
                "output_guardrails",
            ):
                assert getattr(bound, field) is not getattr(original, field)

            assert original.tools == [existing_tool]
            assert original.mcp_servers == [existing_server]
            assert original.handoffs == [existing_handoff]
            assert original.input_guardrails == [existing_input_guardrail]
            assert original.output_guardrails == [existing_output_guardrail]
            return bound

        raise AssertionError("unreachable")

    bound = asyncio.run(run())

    assert lifecycle == [
        ("connect", "claims_mcp"),
        ("list", "claims_mcp"),
        ("cleanup", "claims_mcp"),
    ]
    assert original.mcp_servers == [existing_server]
    assert len(bound.mcp_servers) == 2


def test_bind_tools_materializes_uc_function_and_managed_sandbox(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [
                    UCFunction(id="lookup", function="main.claims.lookup"),
                    Sandbox(
                        id="python",
                        scopes=(Scope.volume("main.claims.files"),),
                    ),
                ],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())
    uc_function, sandbox = (cast(McpServer, server) for server in bound.mcp_servers)

    assert uc_function.name == "lookup"
    assert uc_function.params["url"] == (
        "https://workspace.example.com/api/2.0/mcp/functions/main/claims/lookup"
    )
    assert uc_function.workspace_client is workspace_client
    assert sandbox.name == "python"
    assert sandbox.params["url"] == (
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.sandbox"
    )
    assert sandbox.workspace_client is workspace_client
    assert sandbox.tool_filter == {"allowed_tool_names": ["sandbox", "run_code"]}


def test_bind_tools_selects_a_client_for_each_integration_auth_mode(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    user_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://user.example.com")),
    )
    app_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://app.example.com")),
    )
    resolved_modes: list[str] = []

    def resolve(mode):
        resolved_modes.append(mode)
        return user_client if mode == "user" else app_client

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [
                    MCPService(id="user", service="main.tools.user", auth="user"),
                    Sandbox(
                        id="shared",
                        scopes=(Scope.volume("main.claims.files"),),
                        auth="app",
                    ),
                    UCFunction(id="lookup", function="main.claims.lookup"),
                ],
                stack=stack,
                workspace_client_for=cast(Any, resolve),
            )

    bound = asyncio.run(run())

    user, shared, lookup = (cast(McpServer, server) for server in bound.mcp_servers)
    assert user.workspace_client is user_client
    assert shared.workspace_client is app_client
    assert lookup.workspace_client is app_client
    assert resolved_modes == ["user", "app", "app"]
    assert lifecycle == [
        ("connect", "user"),
        ("connect", "shared"),
        ("connect", "lookup"),
        ("cleanup", "lookup"),
        ("cleanup", "shared"),
        ("cleanup", "user"),
    ]


def test_sandbox_overrides_incoming_downscope_without_mutating_caller_data(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def call_tool(
        _server: McpServer,
        tool_name: str,
        arguments: dict[str, object] | None,
        **kwargs: object,
    ) -> tuple[str, dict[str, object] | None, dict[str, object]]:
        return tool_name, arguments, kwargs

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", call_tool)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    arguments: dict[str, object] = {"code": 'print("hello")'}
    incoming_meta = {
        "downscope": {"volumes": []},
        "trace_id": "request-123",
    }

    async def run() -> tuple[str, dict[str, object] | None, dict[str, object]]:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                Agent(name="claims"),
                [
                    Sandbox(
                        id="python",
                        scopes=(Scope.volume("main.claims.files"),),
                    )
                ],
                stack=stack,
                workspace_client=workspace_client,
            )
            return cast(
                Any,
                await bound.mcp_servers[0].call_tool(
                    "sandbox",
                    arguments,
                    meta=incoming_meta,
                ),
            )

    tool_name, forwarded_arguments, forwarded_kwargs = asyncio.run(run())

    assert tool_name == "sandbox"
    assert forwarded_arguments is arguments
    assert forwarded_kwargs["meta"] == {
        "downscope": {
            "volumes": [
                {"name": "main.claims.files", "permission": "read_only"},
            ]
        },
        "trace_id": "request-123",
    }
    assert forwarded_kwargs["meta"] is not incoming_meta
    assert incoming_meta == {
        "downscope": {"volumes": []},
        "trace_id": "request-123",
    }


def test_empty_selection_is_a_credential_free_no_op(monkeypatch):
    def unexpected_workspace_client() -> None:
        raise AssertionError("empty integrations must not resolve workspace credentials")

    monkeypatch.setattr(openai_mcp, "WorkspaceClient", unexpected_workspace_client)
    original = Agent(name="claims", tools=[cast(Any, object())])

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                original,
                (),
                stack=stack,
                workspace_client_for=lambda mode: unexpected_workspace_client(),
            )

    bound = asyncio.run(run())

    assert bound is not original
    assert bound.tools == original.tools
    assert bound.tools is not original.tools
    assert bound.mcp_servers == []


def test_bind_tools_rejects_both_workspace_client_seams(monkeypatch):
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="workspace_client.*workspace_client_for"):
                await bind_tools(
                    Agent(name="claims"),
                    (),
                    stack=stack,
                    workspace_client=workspace_client,
                    workspace_client_for=lambda mode: workspace_client,
                )

    asyncio.run(run())


def test_deployed_user_integration_without_resolver_fails_before_default_client(
    monkeypatch,
):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setattr(
        openai_mcp,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before default auth")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="web.*user authorization"):
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="web", service="system.ai.web_search", auth="user")],
                    stack=stack,
                )

    asyncio.run(run())


def test_deployed_app_and_uc_integrations_may_use_the_default_client(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    app_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://app.example.com")),
    )
    monkeypatch.setattr(openai_mcp, "_default_workspace_client", lambda: app_client)

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [
                    MCPService(id="shared", service="main.tools.shared", auth="app"),
                    UCFunction(id="lookup", function="main.claims.lookup"),
                ],
                stack=stack,
            )

    bound = asyncio.run(run())

    assert all(
        cast(McpServer, server).workspace_client is app_client for server in bound.mcp_servers
    )


def test_resolver_failure_identifies_integration_without_retrying_as_app(monkeypatch):
    resolved_modes: list[str] = []

    def fail(mode):
        resolved_modes.append(mode)
        raise RuntimeError("unsafe resolver detail: secret-forwarded-token")

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="web") as exc_info:
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="web", service="system.ai.web_search")],
                    stack=stack,
                    workspace_client_for=cast(Any, fail),
                )
            assert "secret-forwarded-token" not in str(exc_info.value)
            assert exc_info.value.__context__ is None

    asyncio.run(run())
    assert resolved_modes == ["user"]


def test_discovery_cancellation_preserves_selected_client_and_stack_cleanup(monkeypatch):
    lifecycle: list[tuple[str, str]] = []
    selected_clients: list[WorkspaceClient] = []
    user_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://user.example.com")),
    )

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def cancel(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        selected_clients.append(server.workspace_client)
        raise asyncio.CancelledError

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", cancel)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(asyncio.CancelledError):
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="web", service="system.ai.web_search")],
                    stack=stack,
                    workspace_client_for=lambda mode: user_client,
                )

    asyncio.run(run())

    assert selected_clients == [user_client]
    assert lifecycle == [("connect", "web"), ("cleanup", "web")]


def test_default_client_uses_runtime_workspace_routing_helper(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    def unexpected_workspace_client() -> None:
        raise AssertionError("bind_tools must use the workspace routing helper")

    routed_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://accounts.example.com")),
    )
    helper_calls = 0

    def routed_workspace_client():
        nonlocal helper_calls
        helper_calls += 1
        return routed_client

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    monkeypatch.setattr(openai_mcp, "WorkspaceClient", unexpected_workspace_client)
    monkeypatch.setattr(
        openai_mcp,
        "_default_workspace_client",
        routed_workspace_client,
        raising=False,
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [MCPService(id="claims_mcp", service="main.claims.service")],
                stack=stack,
            )

    bound = asyncio.run(run())

    assert helper_calls == 1
    assert bound.mcp_servers[0].workspace_client is routed_client


@pytest.mark.parametrize(
    "integration",
    [
        MCPService(id="search", service="system.ai.web_search"),
        Sandbox(id="python", scopes=(Scope.volume("main.claims.files"),)),
        UCFunction(id="lookup", function="main.claims.lookup"),
    ],
)
def test_bind_tools_adds_workspace_routing_header_to_every_generated_transport(
    monkeypatch,
    integration,
):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456789")
    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://accounts.example.com")),
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [integration],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())

    server = cast(McpServer, bound.mcp_servers[0])
    assert server.params["headers"] == {"X-Databricks-Org-Id": "123456789"}


def test_bind_tools_rejects_duplicate_integration_ids_before_resolving_credentials(
    monkeypatch,
):
    def unexpected_workspace_client() -> None:
        raise AssertionError("duplicate names must fail before credentials are resolved")

    monkeypatch.setattr(openai_mcp, "_default_workspace_client", unexpected_workspace_client)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="server name.*duplicate"):
                await bind_tools(
                    Agent(name="claims"),
                    [
                        MCPService(id="duplicate", service="main.claims.first"),
                        MCPService(id="duplicate", service="main.claims.second"),
                    ],
                    stack=stack,
                )

    asyncio.run(run())


def test_bind_tools_rejects_generated_name_matching_existing_server_before_credentials(
    monkeypatch,
):
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    existing = McpServer(
        url="https://custom.example.com/mcp",
        name="duplicate",
        workspace_client=workspace_client,
    )

    def unexpected_workspace_client() -> None:
        raise AssertionError("duplicate names must fail before credentials are resolved")

    monkeypatch.setattr(openai_mcp, "_default_workspace_client", unexpected_workspace_client)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="server name.*duplicate"):
                await bind_tools(
                    Agent(name="claims", mcp_servers=[existing]),
                    [MCPService(id="duplicate", service="main.claims.generated")],
                    stack=stack,
                )

    asyncio.run(run())


def test_bind_tools_eagerly_discovers_and_rejects_duplicate_generated_tools(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        return [Tool(name="lookup_claim", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(
                ValueError,
                match="lookup_claim.*first.*second",
            ):
                await bind_tools(
                    Agent(name="claims"),
                    [
                        MCPService(id="first", service="main.claims.first"),
                        MCPService(id="second", service="main.claims.second"),
                    ],
                    stack=stack,
                    workspace_client=workspace_client,
                )
            assert lifecycle == [
                ("connect", "first"),
                ("list", "first"),
                ("connect", "second"),
                ("list", "second"),
            ]

        assert lifecycle == [
            ("connect", "first"),
            ("list", "first"),
            ("connect", "second"),
            ("list", "second"),
            ("cleanup", "second"),
            ("cleanup", "first"),
        ]

    asyncio.run(run())


def test_bind_tools_defers_existing_dynamic_tool_filters_to_request_context(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        if server is existing:
            raise AssertionError("existing dynamic filters require a request context")
        lifecycle.append(("list", server.name))
        return [Tool(name="generated_tool", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    existing = McpServer(
        url="https://custom.example.com/mcp",
        name="custom",
        workspace_client=workspace_client,
        tool_filter=lambda _context, _tool: True,
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims", mcp_servers=[existing]),
                [MCPService(id="generated", service="main.claims.generated")],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())

    assert lifecycle == [
        ("connect", "generated"),
        ("list", "generated"),
        ("cleanup", "generated"),
    ]
    assert bound.mcp_servers[0] is existing


def test_tool_discovery_failure_surfaces_at_bind_and_remains_stack_managed(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        raise RuntimeError("tool catalog denied")

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="tool catalog denied"):
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="claims", service="main.claims.service")],
                    stack=stack,
                    workspace_client=workspace_client,
                )
            assert lifecycle == [("connect", "claims"), ("list", "claims")]

        assert lifecycle == [
            ("connect", "claims"),
            ("list", "claims"),
            ("cleanup", "claims"),
        ]

    asyncio.run(run())


def test_managed_discovery_normalizes_provider_authorization_and_cleans_up(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        raise _typed_authorization_error()

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    caught: list[Exception] = []

    async def run() -> None:
        try:
            async with AsyncExitStack() as stack:
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="drive", service="system.ai.google_drive")],
                    stack=stack,
                    workspace_client=workspace_client,
                )
        except Exception as error:
            caught.append(error)

    asyncio.run(run())

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert len(caught) == 1
    error = caught[0]
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.integration_id == "drive"
    assert error.data is not None
    assert error.data["elicitations"][0]["url"] == _AUTHORIZATION_URL
    assert "sensitive-query" not in str(error)
    assert "sensitive-query" not in repr(error)
    assert "must-not-be-retained" not in repr(error.data)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert lifecycle == [("connect", "drive"), ("list", "drive"), ("cleanup", "drive")]


@pytest.mark.parametrize(
    ("auth_mode", "error_name", "code", "status"),
    [
        ("user", "InvalidUserAuthorization", "MCP_USER_AUTHORIZATION_INVALID", 401),
        ("app", "InvalidAppAuthorization", "MCP_APP_AUTHORIZATION_INVALID", 500),
    ],
)
def test_managed_discovery_normalizes_invalid_authorization_by_mode(
    monkeypatch, auth_mode, error_name, code, status
):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        raise Unauthenticated("unsafe authorization response body")

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> Exception:
        async with AsyncExitStack() as stack:
            with pytest.raises(Exception) as exc_info:
                await bind_tools(
                    Agent(name="claims"),
                    [
                        MCPService(
                            id="drive",
                            service="system.ai.google_drive",
                            auth=auth_mode,
                        )
                    ],
                    stack=stack,
                    workspace_client=workspace_client,
                )
            return exc_info.value

    error = asyncio.run(run())

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(error, getattr(errors, error_name))
    assert error.integration_id == "drive"
    assert error.code == code
    assert error.status == status
    assert "unsafe authorization response body" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_managed_call_normalizes_flattened_google_drive_authorization(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    result = _flattened_authorization_result(_AUTHORIZATION_URL)

    async def call_tool(
        _server: McpServer,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        return result

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", call_tool)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> Exception:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                Agent(name="claims"),
                [MCPService(id="drive", service="system.ai.google_drive")],
                stack=stack,
                workspace_client=workspace_client,
            )
            with pytest.raises(Exception) as exc_info:
                await bound.mcp_servers[0].call_tool("search", {})
            return exc_info.value

    error = asyncio.run(run())

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.integration_id == "drive"
    assert error.data is not None
    assert error.data["elicitations"][0]["url"] == _AUTHORIZATION_URL
    assert "sensitive-query" not in repr(error)


def test_managed_call_sanitizes_untrusted_flattened_authorization_url(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    result = _flattened_authorization_result(
        "https://attacker.example/explore/data/mcp-services/system/ai/google_drive?secret=x"
    )

    async def call_tool(
        _server: McpServer,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        return result

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", call_tool)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> Exception:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                Agent(name="claims"),
                [MCPService(id="drive", service="system.ai.google_drive")],
                stack=stack,
                workspace_client=workspace_client,
            )
            server = cast(McpServer, bound.mcp_servers[0])
            assert server.max_retry_attempts == 0
            with pytest.raises(Exception) as exc_info:
                await server.call_tool("search", {})
            return exc_info.value

    error = asyncio.run(run())

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert "attacker.example" not in str(error)
    assert "attacker.example" not in repr(error)


@pytest.mark.parametrize(
    ("url", "has_clickable_data"),
    [
        (_AUTHORIZATION_URL, True),
        (
            "https://attacker.example/explore/data/mcp-services/system/ai/google_drive?secret=x",
            False,
        ),
        (
            "ftp://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?secret=x",
            False,
        ),
        (
            "workspace.example.com/explore/data/mcp-services/system/ai/google_drive?secret=x",
            False,
        ),
    ],
)
def test_managed_function_tool_propagates_structured_authorization_error(
    monkeypatch, url, has_clickable_data
):
    result = _flattened_authorization_result(url)

    async def transport_call(
        _server: MCPServerStreamableHttp,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        return result

    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", transport_call)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    server = openai_mcp._server_from_integration(
        MCPService(id="drive", service="system.ai.google_drive"),
        workspace_client,
    )
    function_tool = MCPUtil.to_function_tool(
        Tool(name="search", inputSchema={}),
        server,
        False,
        Agent(name="claims"),
    )
    context = ToolContext(
        context=None,
        tool_name="search",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    async def run() -> Exception:
        with pytest.raises(Exception) as exc_info:
            await function_tool.on_invoke_tool(context, "{}")
        return exc_info.value

    error = asyncio.run(run())

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert isinstance(error, UserError)
    assert error.code == "MCP_AUTHORIZATION_REQUIRED"
    envelope = error.to_error_envelope()["error"]
    if has_clickable_data:
        assert envelope["data"]["elicitations"][0]["url"] == url
    else:
        assert "data" not in envelope
    assert "secret=x" not in str(error)
    assert "secret=x" not in repr(error)


def test_managed_function_tool_keeps_non_catalog_auth_error_model_visible(monkeypatch):
    message = (
        "Authorization header rejected; see https://docs.example.com/troubleshooting/authentication"
    )
    result = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=message)],
    )

    async def transport_call(
        _server: MCPServerStreamableHttp,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        return result

    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", transport_call)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    server = openai_mcp._server_from_integration(
        MCPService(id="drive", service="system.ai.google_drive"),
        workspace_client,
    )
    function_tool = MCPUtil.to_function_tool(
        Tool(name="search", inputSchema={}),
        server,
        False,
        Agent(name="claims"),
    )
    context = ToolContext(
        context=None,
        tool_name="search",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    async def run() -> Any:
        return await function_tool.on_invoke_tool(context, "{}")

    output = asyncio.run(run())

    assert output == {"type": "text", "text": message}


@pytest.mark.parametrize("source_kind", ["result", "error"])
def test_managed_call_classifies_before_databricks_traced_method(monkeypatch, source_kind):
    raw_result = _flattened_authorization_result(_AUTHORIZATION_URL)
    raw_error = _typed_authorization_error()
    traced_inputs: list[object] = []

    async def transport_call(
        _server: MCPServerStreamableHttp,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        if source_kind == "error":
            raise raw_error
        return raw_result

    async def databricks_traced_call(
        server: McpServer,
        tool_name: str,
        arguments: dict[str, Any] | None,
        **kwargs: Any,
    ) -> CallToolResult:
        try:
            value = await MCPServerStreamableHttp.call_tool(
                server,
                tool_name,
                arguments,
                **kwargs,
            )
        except Exception as error:
            traced_inputs.append(error)
            raise
        traced_inputs.append(value)
        return value

    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", transport_call)
    monkeypatch.setattr(McpServer, "call_tool", databricks_traced_call)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    server = openai_mcp._server_from_integration(
        MCPService(id="drive", service="system.ai.google_drive"),
        workspace_client,
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(server.call_tool("search", {}))

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.MCPAuthorizationRequired)
    assert traced_inputs == []
    assert "sensitive-query" not in str(exc_info.value)
    assert "sensitive-query" not in repr(exc_info.value)


def test_uc_and_customer_servers_do_not_gain_managed_provider_normalization(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    async def call_tool(
        _server: McpServer,
        _tool_name: str,
        _arguments: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> CallToolResult:
        raise _typed_authorization_error()

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    monkeypatch.setattr(McpServer, "call_tool", call_tool)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    custom = McpServer(
        url="https://custom.example.com/mcp",
        name="custom",
        workspace_client=workspace_client,
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                Agent(name="claims", mcp_servers=[custom]),
                [
                    MCPService(id="drive", service="system.ai.google_drive"),
                    UCFunction(id="lookup", function="main.tools.lookup"),
                ],
                stack=stack,
                workspace_client=workspace_client,
            )
            for server in (bound.mcp_servers[0], bound.mcp_servers[2]):
                with pytest.raises(McpError):
                    await server.call_tool("search", {})

    asyncio.run(run())


def test_bind_tools_rejects_generated_name_that_collides_with_local_tool(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return [Tool(name="lookup_claim", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    agent = Agent(
        name="claims",
        tools=[cast(Any, SimpleNamespace(name="lookup_claim"))],
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="lookup_claim.*local agent.*claims"):
                await bind_tools(
                    agent,
                    [MCPService(id="claims", service="main.claims.service")],
                    stack=stack,
                    workspace_client=workspace_client,
                )

    asyncio.run(run())
