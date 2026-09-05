"""Attach explicit Databricks integrations to an OpenAI Agents SDK agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from agents import Agent, UserError
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from databricks.sdk import WorkspaceClient
from databricks_openai.agents import McpServer

from databricks_mason.integrations import (
    AuthMode,
    Integration,
    MCPService,
    Sandbox,
    UCFunction,
    downscope_wire,
)
from databricks_mason.runtime.auth import integration_client_resolver
from databricks_mason.runtime.errors import (
    InvalidAppAuthorization,
    InvalidUserAuthorization,
    MasonRuntimeError,
    MCPAuthorizationRequired,
    MCPPermissionDenied,
    classify_managed_mcp_exception,
    classify_managed_mcp_result,
)
from databricks_mason.runtime.workspace import (
    workspace_client as _default_workspace_client,
)
from databricks_mason.runtime.workspace import (
    workspace_headers,
)

TContext = TypeVar("TContext")


class _MCPAuthorizationRequiredUserError(MCPAuthorizationRequired, UserError):
    run_data: None = None


class _InvalidUserAuthorizationUserError(InvalidUserAuthorization, UserError):
    run_data: None = None


class _InvalidAppAuthorizationUserError(InvalidAppAuthorization, UserError):
    run_data: None = None


class _MCPPermissionDeniedUserError(MCPPermissionDenied, UserError):
    run_data: None = None


def _as_agents_user_error(error: MasonRuntimeError) -> MasonRuntimeError:
    integration_id = error.integration_id
    if integration_id is None:
        raise TypeError("Managed MCP errors must identify their integration.")
    if isinstance(error, MCPAuthorizationRequired):
        return _MCPAuthorizationRequiredUserError(integration_id, data=error.data)
    if isinstance(error, InvalidUserAuthorization):
        return _InvalidUserAuthorizationUserError(integration_id)
    if isinstance(error, InvalidAppAuthorization):
        return _InvalidAppAuthorizationUserError(integration_id)
    if isinstance(error, MCPPermissionDenied):
        return _MCPPermissionDeniedUserError(integration_id)
    raise TypeError(f"Unsupported managed MCP error: {type(error).__name__}.")


class _ManagedMcpServer(McpServer):
    """MCP server that classifies only Mason-declared AI Gateway failures."""

    def __init__(
        self,
        *,
        integration_id: str,
        workspace_url: str,
        auth_mode: AuthMode,
        **kwargs: Any,
    ) -> None:
        self._integration_id = integration_id
        self._workspace_url = workspace_url
        self._auth_mode = auth_mode
        kwargs["max_retry_attempts"] = 0
        kwargs["failure_error_function"] = None
        super().__init__(**kwargs)

    async def list_tools(self, *args: Any, **kwargs: Any) -> list[Any]:
        normalized = None
        try:
            return await super().list_tools(*args, **kwargs)
        except Exception as error:
            normalized = classify_managed_mcp_exception(
                error,
                integration_id=self._integration_id,
                workspace_url=self._workspace_url,
                auth_mode=self._auth_mode,
            )
            if normalized is None:
                raise
        raise _as_agents_user_error(normalized) from None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Any:
        normalized = None
        try:
            meta = kwargs.pop("meta", None)
            if kwargs:
                unexpected = next(iter(kwargs))
                raise TypeError(f"Unexpected MCP call option {unexpected!r}.")
            result = await MCPServerStreamableHttp.call_tool(
                self,
                tool_name,
                arguments,
                meta=meta,
            )
        except Exception as error:
            normalized = classify_managed_mcp_exception(
                error,
                integration_id=self._integration_id,
                workspace_url=self._workspace_url,
                auth_mode=self._auth_mode,
            )
            if normalized is None:
                raise
        if normalized is not None:
            raise _as_agents_user_error(normalized) from None

        normalized = classify_managed_mcp_result(
            result,
            integration_id=self._integration_id,
            workspace_url=self._workspace_url,
            auth_mode=self._auth_mode,
        )
        if normalized is not None:
            raise _as_agents_user_error(normalized) from None
        return result


class _SandboxMcpServer(_ManagedMcpServer):
    """MCP server that enforces the selected Sandbox scope on every call."""

    def __init__(self, *, sandbox: Sandbox, **kwargs: Any) -> None:
        self._sandbox = sandbox
        super().__init__(**kwargs)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Any:
        incoming_meta = kwargs.pop("meta", None)
        meta = dict(incoming_meta) if isinstance(incoming_meta, dict) else {}
        meta["downscope"] = downscope_wire(self._sandbox)
        return await super().call_tool(tool_name, arguments, meta=meta, **kwargs)


def _transport_params() -> MCPServerStreamableHttpParams | None:
    headers = workspace_headers()
    if not headers:
        return None
    return MCPServerStreamableHttpParams(url="", headers=headers)


def _server_from_integration(
    integration: Integration,
    workspace_client: WorkspaceClient,
) -> McpServer:
    host = workspace_client.config.host.rstrip("/")
    if isinstance(integration, MCPService):
        return _ManagedMcpServer(
            integration_id=integration.id,
            workspace_url=workspace_client.config.host,
            auth_mode=integration.auth,
            url=f"{host}/ai-gateway/mcp-services/{integration.service}",
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
        )
    if isinstance(integration, Sandbox):
        return _SandboxMcpServer(
            sandbox=integration,
            integration_id=integration.id,
            workspace_url=workspace_client.config.host,
            auth_mode=integration.auth,
            url=f"{host}/ai-gateway/mcp-services/system.ai.sandbox",
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
            tool_filter={"allowed_tool_names": ["sandbox", "run_code"]},
        )
    if isinstance(integration, UCFunction):
        catalog, schema, function_name = integration.function.split(".")
        return McpServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
        )
    raise TypeError(f"Unsupported integration: {type(integration).__name__}")


def _validate_server_names(agent: Agent[Any], integrations: Sequence[Integration]) -> None:
    owners: dict[str, str] = {}
    candidates = [
        *((server.name, "existing agent") for server in agent.mcp_servers),
        *((integration.id, "Databricks integration") for integration in integrations),
    ]
    for name, owner in candidates:
        if previous_owner := owners.get(name):
            raise ValueError(
                f"MCP server name {name!r} is used by both {previous_owner} and {owner}."
            )
        owners[name] = owner


def _claim_tool(tool_owners: dict[str, str], name: str, owner: str) -> None:
    if previous_owner := tool_owners.get(name):
        raise ValueError(
            f"MCP tool {name!r} is advertised by both {previous_owner!r} and {owner!r}."
        )
    tool_owners[name] = owner


async def bind_tools(
    agent: Agent[TContext],
    integrations: Sequence[Integration],
    *,
    stack: AsyncExitStack,
    workspace_client: WorkspaceClient | None = None,
    workspace_client_for: Callable[[AuthMode], WorkspaceClient] | None = None,
) -> Agent[TContext]:
    """Connect selected integrations and return an isolated clone of ``agent``.

    The caller owns ``stack`` and must keep it open for as long as the returned agent can run.
    Closing the stack disconnects every server materialized by this call. Existing servers on the
    input agent are preserved, and their lifecycle remains owned by the caller that supplied them.
    Existing servers are not eagerly inspected because dynamic tool filters require request
    context; the Agents SDK discovers and validates the full MCP tool set during each run. Server
    names and newly materialized tool names are validated before cloning.
    """

    selected = tuple(integrations)
    servers: list[McpServer] = []
    if selected:
        _validate_server_names(agent, selected)
        tool_owners = {
            name: "local agent tool"
            for tool in agent.tools
            if isinstance(name := getattr(tool, "name", None), str)
        }
    client_for_integration = integration_client_resolver(
        selected,
        workspace_client=workspace_client,
        workspace_client_for=workspace_client_for,
        default_workspace_client=_default_workspace_client,
    )
    if selected:
        for integration in selected:
            server = await stack.enter_async_context(
                _server_from_integration(integration, client_for_integration(integration))
            )
            for tool in await server.list_tools():
                _claim_tool(tool_owners, tool.name, integration.id)
            servers.append(server)
    return agent.clone(
        tools=[*agent.tools],
        mcp_servers=[*agent.mcp_servers, *servers],
        handoffs=[*agent.handoffs],
        input_guardrails=[*agent.input_guardrails],
        output_guardrails=[*agent.output_guardrails],
    )
