"""Materialize explicit Databricks integration specs as native LangChain tools."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks_langchain import MCPServer

from databricks_mason.integrations import (
    AuthMode,
    Integration,
    MCPService,
    Sandbox,
    downscope_wire,
)
from databricks_mason.runtime.auth import integration_client_resolver
from databricks_mason.runtime.errors import (
    classify_managed_mcp_exception,
    classify_managed_mcp_result,
)
from databricks_mason.runtime.workspace import workspace_client as _default_workspace_client
from databricks_mason.runtime.workspace import workspace_headers


def _server_from_integration(
    integration: Integration,
    client: WorkspaceClient,
) -> DatabricksMCPServer:
    host = client.config.host.rstrip("/")
    if isinstance(integration, (Sandbox, MCPService)):
        service = "system.ai.sandbox" if isinstance(integration, Sandbox) else integration.service
        return DatabricksMCPServer(
            name=integration.id,
            url=f"{host}/ai-gateway/mcp-services/{service}",
            headers=workspace_headers() or None,
            workspace_client=client,
            timeout=120.0,
        )
    catalog, schema, function_name = integration.function.split(".")
    return DatabricksMCPServer.from_uc_function(
        catalog=catalog,
        schema=schema,
        function_name=function_name,
        name=integration.id,
        headers=workspace_headers() or None,
        workspace_client=client,
        timeout=120.0,
    )


def _sandbox_interceptor(
    sandboxes: dict[str, tuple[Sandbox, DatabricksMCPServer]],
):
    async def interceptor(request: Any, handler: Any) -> Any:
        binding = sandboxes.get(request.server_name)
        if binding is None:
            return await handler(request)

        sandbox, server = binding
        async with create_session(server.to_connection_dict()) as session:
            await session.initialize()
            return await session.call_tool(
                request.name,
                request.args,
                meta={"downscope": downscope_wire(sandbox)},
            )

    return interceptor


def _managed_server_interceptor(managed_servers: dict[str, tuple[AuthMode, str]]):
    async def interceptor(request: Any, handler: Any) -> Any:
        binding = managed_servers.get(request.server_name)
        if binding is None:
            return await handler(request)
        auth_mode, workspace_url = binding

        normalized = None
        try:
            result = await handler(request)
        except Exception as error:
            normalized = classify_managed_mcp_exception(
                error,
                integration_id=request.server_name,
                workspace_url=workspace_url,
                auth_mode=auth_mode,
            )
            if normalized is None:
                raise
        if normalized is not None:
            raise normalized from None

        normalized = classify_managed_mcp_result(
            result,
            integration_id=request.server_name,
            workspace_url=workspace_url,
            auth_mode=auth_mode,
        )
        if normalized is not None:
            raise normalized from None
        return result

    return interceptor


def mcp_client(
    servers: Sequence[DatabricksMCPServer],
    *,
    sandboxes: dict[str, tuple[Sandbox, DatabricksMCPServer]] | None = None,
    managed_servers: dict[str, tuple[AuthMode, str]] | None = None,
) -> DatabricksMultiServerMCPClient:
    """Build a native client whose Sandbox policy closes over the explicit selection."""

    server_list = list(servers)
    names = [server.name for server in server_list]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"MCP server names must be unique; duplicates: {rendered}.")
    for name, (sandbox, server) in (sandboxes or {}).items():
        if sandbox.id != name or server.name != name or server not in server_list:
            raise ValueError(f"Sandbox binding {name!r} does not match its MCP server.")
    server_names = set(names)
    if unknown := sorted(set(managed_servers or {}) - server_names):
        rendered = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Managed MCP bindings do not match a server: {rendered}.")
    interceptors = []
    if managed_servers:
        interceptors.append(_managed_server_interceptor(managed_servers))
    if sandboxes:
        interceptors.append(_sandbox_interceptor(sandboxes))
    # DatabricksMCPServer is a subclass of MCPServer, so coerce the type for the API
    servers_as_mcp = cast("list[MCPServer]", server_list)
    return DatabricksMultiServerMCPClient(servers_as_mcp, tool_interceptors=interceptors)


async def _discover_tools(
    client: DatabricksMultiServerMCPClient,
    servers: Sequence[DatabricksMCPServer],
    managed_servers: dict[str, tuple[AuthMode, str]],
) -> list[Any]:
    async def discover(server: DatabricksMCPServer) -> list[Any]:
        binding = managed_servers.get(server.name)
        if binding is None:
            return await client.get_tools(server_name=server.name)
        auth_mode, workspace_url = binding

        normalized = None
        try:
            return await client.get_tools(server_name=server.name)
        except Exception as error:
            normalized = classify_managed_mcp_exception(
                error,
                integration_id=server.name,
                workspace_url=workspace_url,
                auth_mode=auth_mode,
            )
            if normalized is None:
                raise
        raise normalized from None

    tasks = [asyncio.create_task(discover(server)) for server in servers]
    try:
        groups = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [tool for group in groups for tool in group]


async def load_tools(
    integrations: Sequence[Integration],
    *,
    extra_servers: Sequence[DatabricksMCPServer] = (),
    workspace_client: WorkspaceClient | None = None,
    workspace_client_for: Callable[[AuthMode], WorkspaceClient] | None = None,
    existing_tools: Sequence[Any] = (),
) -> list:
    """Resolve only ``integrations`` and return their native LangChain tools."""

    selected = tuple(integrations)
    supplied_servers = tuple(extra_servers)
    names = [item.id for item in selected]
    names.extend(server.name for server in supplied_servers)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"Integration and MCP server names must be unique; duplicates: {rendered}."
        )

    client_for_integration = integration_client_resolver(
        selected,
        workspace_client=workspace_client,
        workspace_client_for=workspace_client_for,
        default_workspace_client=_default_workspace_client,
    )
    declared_servers: list[DatabricksMCPServer] = []
    sandbox_bindings: dict[str, tuple[Sandbox, DatabricksMCPServer]] = {}
    managed_servers: dict[str, tuple[AuthMode, str]] = {}
    for item in selected:
        integration_client = client_for_integration(item)
        server = _server_from_integration(item, integration_client)
        declared_servers.append(server)
        if isinstance(item, (MCPService, Sandbox)):
            managed_servers[item.id] = (item.auth, integration_client.config.host)
        if isinstance(item, Sandbox):
            sandbox_bindings[item.id] = (item, server)
    servers = [*declared_servers, *supplied_servers]
    if not servers:
        return []
    tools = await _discover_tools(
        mcp_client(
            servers,
            sandboxes=sandbox_bindings,
            managed_servers=managed_servers,
        ),
        servers,
        managed_servers,
    )
    tool_names = [
        name
        for tool in (*existing_tools, *tools)
        if isinstance(name := getattr(tool, "name", None), str)
    ]
    duplicates = sorted(name for name, count in Counter(tool_names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"LangGraph MCP tool names must be unique; duplicates: {rendered}.")
    return tools


async def mcp_tools(extra_servers: list[DatabricksMCPServer] | None = None) -> list:
    """Fail loudly for the retired manifest-backed API instead of dropping integrations."""

    del extra_servers
    raise RuntimeError(
        "mcp_tools() no longer discovers tools from agent.toml; migrate the selected "
        "integrations to DATABRICKS_TOOLS and call "
        "load_tools(DATABRICKS_TOOLS, extra_servers=build_mcp_servers(), "
        "workspace_client_for=request_auth.client_for)."
    )
