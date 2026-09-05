from __future__ import annotations

import pytest

from databricks_mason.errors import AgentCliError
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction


def test_sandbox_requires_a_valid_fixed_downscope() -> None:
    sandbox = Sandbox(
        id="sandbox",
        scopes=(Scope.table("samples.nyctaxi.trips"),),
    )

    assert sandbox.kind == "sandbox"
    assert sandbox.scopes[0].resource == "table:samples.nyctaxi.trips"

    with pytest.raises(AgentCliError, match="at least one scope"):
        Sandbox(id="sandbox", scopes=())


@pytest.mark.parametrize(
    ("default_integration", "app_integration"),
    [
        (
            lambda: MCPService(id="web", service="system.ai.web_search"),
            lambda: MCPService(id="web", service="system.ai.web_search", auth="app"),
        ),
        (
            lambda: Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),)),
            lambda: Sandbox(
                id="sandbox",
                scopes=(Scope.volume("main.data.files"),),
                auth="app",
            ),
        ),
    ],
)
def test_ai_gateway_integrations_default_to_user_and_accept_app_auth(
    default_integration,
    app_integration,
) -> None:
    assert default_integration().auth == "user"
    assert app_integration().auth == "app"


@pytest.mark.parametrize(
    "integration",
    [
        lambda: MCPService(
            id="web",
            service="system.ai.web_search",
            auth="creator",  # type: ignore[arg-type]
        ),
        lambda: Sandbox(
            id="sandbox",
            scopes=(Scope.volume("main.data.files"),),
            auth="creator",  # type: ignore[arg-type]
        ),
    ],
)
def test_ai_gateway_integrations_reject_unsupported_auth(integration) -> None:
    with pytest.raises(AgentCliError, match="auth.*creator"):
        integration()


@pytest.mark.parametrize(
    ("integration", "expected"),
    [
        (
            lambda: MCPService(id="web", service="system.ai.web_search"),
            frozenset({"ai-gateway"}),
        ),
        (
            lambda: Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),)),
            frozenset({"ai-gateway"}),
        ),
        (
            lambda: MCPService(id="web", service="system.ai.web_search", auth="app"),
            frozenset(),
        ),
        (
            lambda: Sandbox(
                id="sandbox",
                scopes=(Scope.volume("main.data.files"),),
                auth="app",
            ),
            frozenset(),
        ),
        (lambda: UCFunction(id="lookup", function="main.tools.lookup"), frozenset()),
    ],
)
def test_integrations_report_required_apps_user_scopes(
    integration,
    expected: frozenset[str],
) -> None:
    assert integration().required_user_scopes == expected


def test_uc_function_does_not_expose_request_auth_configuration() -> None:
    with pytest.raises(TypeError, match="auth"):
        UCFunction(
            id="lookup",
            function="main.tools.lookup",
            auth="user",  # type: ignore[unknown-argument]
        )


@pytest.mark.parametrize(
    "integration",
    [
        MCPService(id="web", service="system.ai.web_search"),
        UCFunction(id="lookup", function="main.tools.lookup"),
    ],
)
def test_remote_integrations_validate_three_part_names(
    integration: MCPService | UCFunction,
) -> None:
    assert integration.kind in {"mcp", "uc_function"}

    with pytest.raises(AgentCliError, match="three-part"):
        MCPService(id="web", service="web_search")


def test_scope_validates_resource_kind_and_permission() -> None:
    assert Scope.workspace("/Workspace/Users/alice").resource == (
        "workspace:/Workspace/Users/alice"
    )

    with pytest.raises(AgentCliError, match="permission"):
        Scope.table("samples.nyctaxi.trips", permission="owner")  # type: ignore[arg-type]


def test_public_specs_reject_invalid_runtime_values_with_domain_errors() -> None:
    with pytest.raises(AgentCliError, match="workspace scope"):
        Scope.workspace(None)  # type: ignore[arg-type]

    with pytest.raises(AgentCliError, match="Scope"):
        Sandbox(id="sandbox", scopes=("main.data.files",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "integration",
    [
        lambda: MCPService(id="web", service="system.ai.web?redirect=/other"),
        lambda: UCFunction(id="lookup", function="main.tools.lookup/extra"),
    ],
)
def test_remote_targets_reject_url_control_characters(integration) -> None:
    with pytest.raises(AgentCliError, match="Invalid three-part"):
        integration()


def test_scope_parse_rejects_mistyped_explicit_kind() -> None:
    with pytest.raises(AgentCliError, match="scope kind.*tables"):
        Scope.parse("tables:samples.nyctaxi.trips")
