from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from databricks.sdk.errors import PermissionDenied, Unauthenticated
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp import McpError
from mcp.types import CallToolResult, ErrorData, TextContent, Tool

import databricks_mason.langgraph as langgraph_adapter
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction
from databricks_mason.langgraph import load_tools
from databricks_mason.langgraph import mcp as mcp_module

_AUTHORIZATION_URL = (
    "https://workspace.example.com/explore/data/mcp-services/system/ai/google_drive"
    "?oauth_state=sensitive-query"
)


def test_package_example_threads_request_auth_client_resolver() -> None:
    assert langgraph_adapter.__doc__ is not None
    assert "workspace_client_for=request_auth.client_for" in langgraph_adapter.__doc__


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


class _Server:
    def __init__(self, name: str, url: str, **kwargs) -> None:
        self.name = name
        self.url = url
        self.kwargs = kwargs

    @classmethod
    def from_uc_function(
        cls,
        catalog: str,
        schema: str,
        function_name: str,
        name: str,
        workspace_client,
        **kwargs,
    ):
        return cls(
            name,
            f"{workspace_client.config.host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}",
            workspace_client=workspace_client,
            **kwargs,
        )

    def to_connection_dict(self):
        return {
            "transport": "streamable_http",
            "url": self.url,
            "workspace_client": self.kwargs.get("workspace_client"),
        }


class _Client:
    last: _Client | None = None

    def __init__(self, servers, **kwargs) -> None:
        self.servers = list(servers)
        self.kwargs = kwargs
        _Client.last = self

    async def get_tools(self, server_name=None):
        servers = (
            self.servers
            if server_name is None
            else [server for server in self.servers if server.name == server_name]
        )
        return [server.name for server in servers]


@pytest.fixture
def runtime(monkeypatch):
    client = SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com"))
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.setattr(mcp_module, "DatabricksMCPServer", _Server)
    monkeypatch.setattr(mcp_module, "DatabricksMultiServerMCPClient", _Client)
    monkeypatch.setattr(mcp_module, "_default_workspace_client", lambda: client)
    monkeypatch.setattr(mcp_module, "workspace_headers", lambda: {})
    return client


def test_load_tools_materializes_only_explicit_integrations_and_extra_servers(runtime) -> None:
    custom = _Server("custom", "https://custom.example.com/mcp")

    tools = asyncio.run(
        load_tools(
            [
                Sandbox(id="sandbox", scopes=(Scope.table("samples.nyctaxi.trips"),)),
                MCPService(id="web", service="system.ai.web_search"),
                UCFunction(id="lookup", function="main.tools.lookup"),
            ],
            extra_servers=cast(Any, [custom]),
        )
    )

    assert tools == ["sandbox", "web", "lookup", "custom"]
    client = cast(Any, _Client.last)
    assert [server.url for server in client.servers] == [
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.sandbox",
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.web_search",
        "https://workspace.example.com/api/2.0/mcp/functions/main/tools/lookup",
        "https://custom.example.com/mcp",
    ]


def test_load_tools_selects_a_client_for_each_integration_auth_mode(runtime) -> None:
    user_client = SimpleNamespace(config=SimpleNamespace(host="https://user.example.com"))
    app_client = SimpleNamespace(config=SimpleNamespace(host="https://app.example.com"))
    resolved_modes: list[str] = []

    def resolve(mode):
        resolved_modes.append(mode)
        return user_client if mode == "user" else app_client

    tools = asyncio.run(
        load_tools(
            [
                MCPService(id="user", service="main.tools.user", auth="user"),
                Sandbox(
                    id="shared",
                    scopes=(Scope.volume("main.data.files"),),
                    auth="app",
                ),
                UCFunction(id="lookup", function="main.tools.lookup"),
            ],
            workspace_client_for=cast(Any, resolve),
        )
    )

    assert tools == ["user", "shared", "lookup"]
    client = cast(Any, _Client.last)
    user, shared, lookup = client.servers
    assert user.kwargs["workspace_client"] is user_client
    assert shared.kwargs["workspace_client"] is app_client
    assert lookup.kwargs["workspace_client"] is app_client
    assert resolved_modes == ["user", "app", "app"]


def test_sandbox_interceptor_closes_over_selection_and_protects_downscope(
    runtime, monkeypatch
) -> None:
    call = {}

    class _Session:
        async def initialize(self):
            call["initialized"] = True

        async def call_tool(self, name, arguments, **kwargs):
            call.update(name=name, arguments=arguments, kwargs=kwargs)
            return "ok"

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(mcp_module, "create_session", lambda connection: _SessionContext())
    asyncio.run(
        mcp_module.load_tools([Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))])
    )
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][-1]
    request = SimpleNamespace(
        server_name="sandbox",
        name="sandbox",
        args={"code": "print('ok')", "downscope": "model-controlled"},
    )

    result = asyncio.run(interceptor(request, lambda request: None))

    assert result == "ok"
    assert call["arguments"] is request.args
    assert call["kwargs"] == {
        "meta": {"downscope": {"volumes": [{"name": "main.data.files", "permission": "read_only"}]}}
    }


def test_request_workspace_client_is_reused_by_sandbox_interceptor(runtime, monkeypatch) -> None:
    request_client = SimpleNamespace(
        config=SimpleNamespace(host="https://request-workspace.example.com")
    )
    monkeypatch.setattr(
        mcp_module,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must use the request client")),
    )

    class _Session:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return "ok"

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(mcp_module, "create_session", lambda connection: _SessionContext())

    asyncio.run(
        mcp_module.load_tools(
            [Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))],
            workspace_client=cast(Any, request_client),
        )
    )
    client = cast(Any, _Client.last)
    server = client.servers[0]
    interceptor = client.kwargs["tool_interceptors"][-1]

    assert server.kwargs["workspace_client"] is request_client
    assert (
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name="sandbox", name="sandbox", args={"code": "1"}),
                lambda request: None,
            )
        )
        == "ok"
    )


def test_sandbox_reconnect_uses_selected_client_and_cleans_up_on_cancellation(
    runtime, monkeypatch
) -> None:
    request_client = SimpleNamespace(
        config=SimpleNamespace(host="https://request-workspace.example.com")
    )
    lifecycle: list[str] = []
    connections: list[dict[str, Any]] = []

    class _Session:
        async def initialize(self):
            lifecycle.append("initialize")

        async def call_tool(self, name, arguments, **kwargs):
            lifecycle.append("call")
            raise asyncio.CancelledError

    class _SessionContext:
        async def __aenter__(self):
            lifecycle.append("enter")
            return _Session()

        async def __aexit__(self, exc_type, *_args):
            lifecycle.append(f"exit:{exc_type.__name__}")
            return False

    def create_session(connection):
        connections.append(connection)
        return _SessionContext()

    monkeypatch.setattr(mcp_module, "create_session", create_session)
    asyncio.run(
        mcp_module.load_tools(
            [Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))],
            workspace_client_for=cast(Any, lambda mode: request_client),
        )
    )
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][-1]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name="sandbox", name="sandbox", args={"code": "1"}),
                lambda request: None,
            )
        )

    assert connections[0]["workspace_client"] is request_client
    assert lifecycle == ["enter", "initialize", "call", "exit:CancelledError"]


def test_empty_selection_returns_without_constructing_a_client(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve auth")),
    )

    assert asyncio.run(mcp_module.load_tools([])) == []


def test_extra_servers_do_not_resolve_databricks_credentials(runtime) -> None:
    custom = _Server("custom", "https://custom.example.com/mcp")

    def unexpected_resolver(mode):
        raise AssertionError(f"extra servers must not resolve {mode} credentials")

    assert asyncio.run(
        mcp_module.load_tools(
            [],
            extra_servers=cast(Any, [custom]),
            workspace_client_for=cast(Any, unexpected_resolver),
        )
    ) == ["custom"]


def test_deployed_user_integration_without_resolver_fails_before_default_client(
    runtime, monkeypatch
) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setattr(
        mcp_module,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must fail before default auth")),
    )

    with pytest.raises(RuntimeError, match="web.*user authorization"):
        asyncio.run(
            mcp_module.load_tools(
                [MCPService(id="web", service="system.ai.web_search", auth="user")]
            )
        )


def test_deployed_app_and_uc_integrations_may_use_the_default_client(runtime, monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")

    assert asyncio.run(
        mcp_module.load_tools(
            [
                MCPService(id="shared", service="main.tools.shared", auth="app"),
                UCFunction(id="lookup", function="main.tools.lookup"),
            ]
        )
    ) == ["shared", "lookup"]
    client = cast(Any, _Client.last)
    assert all(server.kwargs["workspace_client"] is runtime for server in client.servers)


def test_resolver_failure_identifies_integration_without_retrying_as_app(runtime) -> None:
    resolved_modes: list[str] = []

    def fail(mode):
        resolved_modes.append(mode)
        raise RuntimeError("unsafe resolver detail: secret-forwarded-token")

    with pytest.raises(RuntimeError, match="web") as exc_info:
        asyncio.run(
            mcp_module.load_tools(
                [MCPService(id="web", service="system.ai.web_search")],
                workspace_client_for=cast(Any, fail),
            )
        )

    assert resolved_modes == ["user"]
    assert "secret-forwarded-token" not in str(exc_info.value)
    assert exc_info.value.__context__ is None


def test_load_tools_rejects_both_workspace_client_seams(runtime) -> None:
    with pytest.raises(ValueError, match="workspace_client.*workspace_client_for"):
        asyncio.run(
            mcp_module.load_tools(
                [],
                workspace_client=cast(Any, runtime),
                workspace_client_for=cast(Any, lambda mode: runtime),
            )
        )


def test_configured_integration_discovery_failure_is_not_silently_dropped(
    runtime, monkeypatch
) -> None:
    async def fail(self, server_name=None):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(_Client, "get_tools", fail)

    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(mcp_module.load_tools([MCPService(id="web", service="system.ai.web_search")]))


def test_managed_discovery_normalizes_typed_provider_authorization(runtime, monkeypatch) -> None:
    async def fail(self, server_name=None):
        raise _typed_authorization_error()

    monkeypatch.setattr(_Client, "get_tools", fail)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            mcp_module.load_tools([MCPService(id="drive", service="system.ai.google_drive")])
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    error = exc_info.value
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.integration_id == "drive"
    assert error.data is not None
    assert error.data["elicitations"][0]["url"] == _AUTHORIZATION_URL
    assert "sensitive-query" not in str(error)
    assert "sensitive-query" not in repr(error)
    assert "must-not-be-retained" not in repr(error.data)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_managed_discovery_normalizes_typed_permission_denial(runtime, monkeypatch) -> None:
    async def fail(self, server_name=None):
        raise PermissionDenied("unsafe permission response body")

    monkeypatch.setattr(_Client, "get_tools", fail)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            mcp_module.load_tools([MCPService(id="drive", service="system.ai.google_drive")])
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    error = exc_info.value
    assert isinstance(error, errors.MCPPermissionDenied)
    assert error.integration_id == "drive"
    assert "unsafe permission response body" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_managed_call_normalizes_flattened_google_drive_authorization(runtime) -> None:
    asyncio.run(mcp_module.load_tools([MCPService(id="drive", service="system.ai.google_drive")]))
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][0]
    result = _flattened_authorization_result(_AUTHORIZATION_URL)

    async def call(_request):
        return result

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name="drive", name="search", args={}),
                call,
            )
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    error = exc_info.value
    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.integration_id == "drive"
    assert error.data is not None
    assert error.data["elicitations"][0]["url"] == _AUTHORIZATION_URL
    assert "sensitive-query" not in repr(error)


def test_managed_call_sanitizes_untrusted_flattened_authorization_url(runtime) -> None:
    asyncio.run(mcp_module.load_tools([MCPService(id="drive", service="system.ai.google_drive")]))
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][0]
    result = _flattened_authorization_result(
        "https://attacker.example/explore/data/mcp-services/system/ai/google_drive?secret=x"
    )

    async def call(_request):
        return result

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name="drive", name="search", args={}),
                call,
            )
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.MCPAuthorizationRequired)
    assert exc_info.value.data is None
    assert "attacker.example" not in str(exc_info.value)
    assert "attacker.example" not in repr(exc_info.value)


@pytest.mark.parametrize(
    "candidate",
    [
        "https://attacker.example/explore/data/mcp-services/system/ai/google_drive?secret=x",
        "ftp://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?secret=x",
        "workspace.example.com/explore/data/mcp-services/system/ai/google_drive?secret=x",
    ],
)
def test_langgraph_tool_boundary_sanitizes_rejected_authorization_url(candidate) -> None:
    result = _flattened_authorization_result(candidate)

    class _Session:
        async def call_tool(self, name, arguments, **kwargs):
            return result

    interceptor = mcp_module._managed_server_interceptor(
        {"drive": ("user", "https://workspace.example.com")}
    )
    tool = convert_mcp_tool_to_langchain_tool(
        cast(Any, _Session()),
        Tool(name="search", inputSchema={}),
        server_name="drive",
        tool_interceptors=[interceptor],
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(cast(Any, tool).ainvoke({}))

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.MCPAuthorizationRequired)
    assert exc_info.value.data is None
    assert candidate not in str(exc_info.value)
    assert candidate not in repr(exc_info.value)


def test_langgraph_tool_boundary_keeps_non_catalog_auth_error_model_visible() -> None:
    message = (
        "Authorization header rejected; see https://docs.example.com/troubleshooting/authentication"
    )
    result = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=message)],
    )

    class _Session:
        async def call_tool(self, name, arguments, **kwargs):
            return result

    interceptor = mcp_module._managed_server_interceptor(
        {"drive": ("user", "https://workspace.example.com")}
    )
    tool = convert_mcp_tool_to_langchain_tool(
        cast(Any, _Session()),
        Tool(name="search", inputSchema={}),
        server_name="drive",
        tool_interceptors=[interceptor],
    )

    output = asyncio.run(cast(Any, tool).ainvoke({}))

    assert isinstance(output, list)
    assert output[0]["text"] == message


def test_managed_discovery_uses_app_auth_failure_code(runtime, monkeypatch) -> None:
    async def fail(self, server_name=None):
        raise Unauthenticated("unsafe App credential response")

    monkeypatch.setattr(_Client, "get_tools", fail)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            mcp_module.load_tools(
                [
                    MCPService(
                        id="shared",
                        service="system.ai.web_search",
                        auth="app",
                    )
                ]
            )
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.InvalidAppAuthorization)
    assert exc_info.value.code == "MCP_APP_AUTHORIZATION_INVALID"
    assert exc_info.value.status == 500
    assert "unsafe App credential response" not in str(exc_info.value)


def test_uc_resolver_failure_uses_app_auth_failure_code(runtime) -> None:
    resolved_modes: list[str] = []

    def fail(mode):
        resolved_modes.append(mode)
        raise RuntimeError("unsafe App resolver response")

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            mcp_module.load_tools(
                [UCFunction(id="lookup", function="main.tools.lookup")],
                workspace_client_for=cast(Any, fail),
            )
        )

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.InvalidAppAuthorization)
    assert exc_info.value.code == "MCP_APP_AUTHORIZATION_INVALID"
    assert exc_info.value.status == 500
    assert "unsafe App resolver response" not in str(exc_info.value)
    assert resolved_modes == ["app"]


def test_sandbox_call_keeps_downscope_and_normalizes_provider_authorization(
    runtime, monkeypatch
) -> None:
    result = _flattened_authorization_result(_AUTHORIZATION_URL)

    class _Session:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            assert kwargs == {
                "meta": {
                    "downscope": {
                        "volumes": [{"name": "main.data.files", "permission": "read_only"}]
                    }
                }
            }
            return result

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(mcp_module, "create_session", lambda connection: _SessionContext())
    asyncio.run(
        mcp_module.load_tools([Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))])
    )
    client = cast(Any, _Client.last)
    managed, sandbox = client.kwargs["tool_interceptors"]
    request = SimpleNamespace(server_name="sandbox", name="sandbox", args={"code": "1"})

    async def sandbox_call(inner_request):
        return await sandbox(inner_request, lambda _request: None)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(managed(request, sandbox_call))

    errors = importlib.import_module("databricks_mason.runtime.errors")
    assert isinstance(exc_info.value, errors.MCPAuthorizationRequired)


@pytest.mark.parametrize("unmanaged_name", ["lookup", "custom"])
def test_managed_interceptor_does_not_trust_uc_or_customer_servers(runtime, unmanaged_name) -> None:
    custom = _Server("custom", "https://custom.example.com/mcp")
    asyncio.run(
        mcp_module.load_tools(
            [
                MCPService(id="drive", service="system.ai.google_drive"),
                UCFunction(id="lookup", function="main.tools.lookup"),
            ],
            extra_servers=cast(Any, [custom]),
        )
    )
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][0]

    async def fail(_request):
        raise _typed_authorization_error()

    with pytest.raises(McpError):
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name=unmanaged_name, name="search", args={}),
                fail,
            )
        )


def test_discovery_failure_cancels_and_awaits_blocked_siblings(runtime, monkeypatch) -> None:
    lifecycle: list[str] = []
    sibling_started = asyncio.Event()

    class _GatheringClient(_Client):
        async def get_tools(self, server_name=None):
            if server_name is None:
                tasks = [
                    asyncio.create_task(self.get_tools(server.name)) for server in self.servers
                ]
                groups = await asyncio.gather(*tasks)
                return [tool for group in groups for tool in group]
            if server_name == "failing":
                await sibling_started.wait()
                lifecycle.append("failing:error")
                raise RuntimeError("discovery failed")

            lifecycle.append("blocked:start")
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                lifecycle.append("blocked:cancelled")
                raise
            finally:
                lifecycle.append("blocked:cleanup")

    monkeypatch.setattr(mcp_module, "DatabricksMultiServerMCPClient", _GatheringClient)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="discovery failed"):
            await mcp_module.load_tools(
                [
                    MCPService(id="failing", service="main.tools.failing"),
                    MCPService(id="blocked", service="main.tools.blocked"),
                ],
                workspace_client=cast(Any, runtime),
            )
        assert lifecycle == [
            "blocked:start",
            "failing:error",
            "blocked:cancelled",
            "blocked:cleanup",
        ]

    asyncio.run(run())


def test_load_tools_rejects_server_name_collisions_before_attaching_sandbox_policy(runtime) -> None:
    custom = _Server("shared", "https://custom.example.com/mcp")
    _Client.last = None

    with pytest.raises(ValueError, match="unique.*shared"):
        asyncio.run(
            mcp_module.load_tools(
                [
                    Sandbox(
                        id="shared",
                        scopes=(Scope.volume("main.data.files"),),
                    )
                ],
                extra_servers=cast(Any, [custom]),
            )
        )

    assert _Client.last is None


def test_direct_mcp_client_rejects_duplicate_server_names(runtime) -> None:
    with pytest.raises(ValueError, match="server names.*shared"):
        mcp_module.mcp_client(
            cast(
                Any,
                [
                    _Server("shared", "https://one.example.com/mcp"),
                    _Server("shared", "https://two.example.com/mcp"),
                ],
            )
        )


def test_load_tools_rejects_duplicate_advertised_tool_names(runtime, monkeypatch) -> None:
    async def duplicate_tools(self, server_name=None):
        return [
            SimpleNamespace(name="lookup"),
            SimpleNamespace(name="lookup"),
        ]

    monkeypatch.setattr(_Client, "get_tools", duplicate_tools)

    with pytest.raises(ValueError, match="tool names.*lookup"):
        asyncio.run(
            mcp_module.load_tools(
                [
                    MCPService(id="first", service="main.tools.first"),
                    MCPService(id="second", service="main.tools.second"),
                ]
            )
        )


def test_load_tools_rejects_remote_name_that_collides_with_existing_agent_tool(
    runtime, monkeypatch
) -> None:
    async def remote_tools(self, server_name=None):
        return [SimpleNamespace(name="lookup")]

    monkeypatch.setattr(_Client, "get_tools", remote_tools)

    with pytest.raises(ValueError, match="tool names.*lookup"):
        asyncio.run(
            mcp_module.load_tools(
                [MCPService(id="remote", service="main.tools.remote")],
                existing_tools=[SimpleNamespace(name="lookup")],
            )
        )


@pytest.mark.parametrize(
    "extra_servers",
    [None, [_Server("custom", "https://custom.example.com/mcp")]],
)
def test_legacy_mcp_tools_call_fails_with_migration_guidance(runtime, extra_servers) -> None:
    with pytest.raises(RuntimeError, match=r"mcp_tools\(\).*agent\.toml.*load_tools") as exc:
        asyncio.run(mcp_module.mcp_tools(extra_servers))

    assert "workspace_client_for=request_auth.client_for" in str(exc.value)
