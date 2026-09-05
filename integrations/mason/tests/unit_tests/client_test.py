"""Unit tests for MasonClient paths, verbs, bodies, and error mapping."""

from __future__ import annotations

from unittest import mock

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.apps import App

from databricks_mason.client import (
    MasonClient,
    _workspace_client,
    memory_entry_path,
    memory_store_path,
    session_store_path,
)
from databricks_mason.errors import AgentCliError


def _client(workspace_client):
    inst = workspace_client.return_value
    inst.config.host = "https://ws.example.com"
    inst.api_client.do.return_value = {}
    return MasonClient(profile="p"), inst.api_client.do


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_memory_store(workspace_client):
    c, do = _client(workspace_client)
    c.create_memory_store("acme", "desc")
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/memory-stores",
        query=None,
        body={"display_name": "acme", "description": "desc"},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_list_memory_stores_query(workspace_client):
    c, do = _client(workspace_client)
    c.list_memory_stores(page_size=10)
    do.assert_called_once_with(
        "GET", "/api/agents/v1/memory-stores", query={"page_size": 10}, body=None
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_list_mcp_services_query(workspace_client):
    c, do = _client(workspace_client)

    c.list_mcp_services("system.ai", page_token="next")

    do.assert_called_once_with(
        "GET",
        "/api/2.1/unity-catalog/mcp-services",
        query={"parent": "schemas/system.ai", "page_token": "next"},
        body=None,
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_memory_store_normalizes_id(workspace_client):
    c, do = _client(workspace_client)
    c.get_memory_store("abc123")
    do.assert_called_once_with("GET", "/api/agents/v1/memory-stores/abc123", query=None, body=None)


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_search_memory_entries(workspace_client):
    c, do = _client(workspace_client)
    c.search_memory_entries("s1", "alice", "style", limit=5)
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/memory-stores/s1/entries:search",
        query=None,
        body={"actor_id": "alice", "query": "style", "limit": 5},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_session_puts_session_id_in_query(workspace_client):
    c, do = _client(workspace_client)
    c.create_session("store1", "alice", session_id="sid")
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/session-stores/store1/sessions",
        query={"session_id": "sid"},
        body={"actor_id": "alice"},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_session_scoped_vs_unscoped(workspace_client):
    c, do = _client(workspace_client)
    c.get_session("sid", store="store1")
    c.get_session("sid")
    assert do.call_args_list[0].args[1] == "/api/agents/v1/session-stores/store1/sessions/sid"
    assert do.call_args_list[1].args[1] == "/api/agents/v1/sessions/sid"


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_append_wraps_items_in_data(workspace_client):
    c, do = _client(workspace_client)
    c.append_session_items("store1", "sid", [{"role": "user", "content": "hi"}])
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/session-stores/store1/sessions/sid/items:append",
        query=None,
        body={"items": [{"data": {"role": "user", "content": "hi"}}]},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_delete_session_force_flag(workspace_client):
    c, do = _client(workspace_client)
    c.delete_session("store1", "sid", force=True)
    do.assert_called_once_with(
        "DELETE",
        "/api/agents/v1/session-stores/store1/sessions/sid",
        query={"force": True},
        body=None,
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_preview_error_is_mapped_with_hint(workspace_client):
    c, do = _client(workspace_client)

    class PreviewApiError(RuntimeError):
        error_code = "NOT_IMPLEMENTED"

    err = PreviewApiError("not implemented")
    do.side_effect = err
    try:
        c.list_memory_stores()
        raise AssertionError("expected AgentCliError")
    except AgentCliError as mapped:
        assert mapped.error_code == "NOT_IMPLEMENTED"
        assert mapped.hint is not None


def test_account_routed_profile_uses_configured_host_and_workspace_header():
    resolved = mock.Mock()
    resolved.config.host = "https://workspace.example.com"
    resolved.config.workspace_id = "123"
    routed = mock.Mock()

    with (
        mock.patch("databricks_mason.client.WorkspaceClient", side_effect=[resolved, routed]) as wc,
        mock.patch(
            "databricks_mason.client._profile_host",
            return_value="https://account.example.com",
        ),
    ):
        client = _workspace_client("p")

    assert client is routed
    assert wc.call_args_list == [
        mock.call(profile="p"),
        mock.call(
            profile="p",
            host="https://account.example.com",
            custom_headers={"X-Databricks-Org-Id": "123"},
        ),
    ]


def test_path_helpers():
    assert memory_store_path("abc") == "memory-stores/abc"
    assert memory_store_path("memory-stores/abc") == "memory-stores/abc"
    assert memory_entry_path("s", "e") == "memory-stores/s/entries/e"
    assert memory_entry_path("s", "memory-stores/s/entries/e") == "memory-stores/s/entries/e"


def test_memory_store_path_rejects_empty_and_wrong_type():
    # Empty / whitespace-only ids must not build a "memory-stores/" URL (ML-69222).
    for bad in ["", "   ", "/", "memory-stores/"]:
        with pytest.raises(AgentCliError):
            memory_store_path(bad)
    # A wrong-typed resource name is rejected rather than nested into the URL.
    with pytest.raises(AgentCliError):
        memory_store_path("session-stores/abc")


def test_session_store_path_rejects_empty():
    for bad in ["", "   ", "/", "session-stores/"]:
        with pytest.raises(AgentCliError):
            session_store_path(bad)
    assert session_store_path("s1") == "session-stores/s1"
    assert session_store_path("session-stores/s1") == "session-stores/s1"


def test_memory_entry_path_rejects_empty_entry():
    with pytest.raises(AgentCliError):
        memory_entry_path("s", "")


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_update_memory_store_no_fields_raises_without_calling_api(workspace_client):
    c, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        c.update_memory_store("abc")  # no display_name / description
    do.assert_not_called()


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_update_session_store_no_fields_raises_without_calling_api(workspace_client):
    c, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        c.update_session_store("s1")  # no description / metadata
    do.assert_not_called()


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_update_memory_entry_no_fields_raises_without_calling_api(workspace_client):
    c, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        c.update_memory_entry("s", "memory-stores/s/entries/e")
    do.assert_not_called()


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_delete_session_store_normalizes_path(workspace_client):
    c, do = _client(workspace_client)
    c.delete_session_store("session-stores/s1")
    do.assert_called_once_with("DELETE", "/api/agents/v1/session-stores/s1", query=None, body=None)


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_app_uses_typed_sdk_api(workspace_client):
    c, _ = _client(workspace_client)
    expected = App(
        name="agent",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway", "openid"],
    )
    workspace_client.return_value.apps.get.return_value = expected

    assert c.get_app("agent") is expected
    workspace_client.return_value.apps.get.assert_called_once_with("agent")


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_app_normalizes_sdk_not_found_without_error_code(workspace_client):
    c, _ = _client(workspace_client)
    workspace_client.return_value.apps.get.side_effect = NotFound("missing")

    with pytest.raises(AgentCliError) as exc_info:
        c.get_app("agent")

    assert exc_info.value.error_code == "NOT_FOUND"


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_app_sends_initial_typed_user_api_scopes(workspace_client):
    c, _ = _client(workspace_client)
    expected = App(name="agent", user_api_scopes=["ai-gateway", "sql"])
    waiter = mock.Mock(response=expected)
    waiter.result.return_value = expected
    workspace_client.return_value.apps.create.return_value = waiter

    assert c.create_app("agent", ["ai-gateway", "sql"]) is expected

    request = workspace_client.return_value.apps.create.call_args.args[0]
    assert isinstance(request, App)
    assert request.as_dict() == {
        "name": "agent",
        "user_api_scopes": ["ai-gateway", "sql"],
    }


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_update_app_masks_the_update_to_user_api_scopes(workspace_client):
    c, _ = _client(workspace_client)
    expected = App(name="agent", user_api_scopes=["sql"])
    workspace_client.return_value.apps.get.return_value = expected

    assert c.update_app("agent", ["sql"]) is expected

    call = workspace_client.return_value.apps.create_update_and_wait.call_args
    assert call.args == ("agent", "user_api_scopes")
    request = call.kwargs["app"]
    assert isinstance(request, App)
    assert request.as_dict() == {"name": "agent", "user_api_scopes": ["sql"]}
    workspace_client.return_value.apps.update.assert_not_called()


@pytest.mark.parametrize("operation", ["create", "update"])
@mock.patch("databricks_mason.client.WorkspaceClient")
def test_app_scope_writes_preserve_an_explicit_empty_list_on_the_wire(workspace_client, operation):
    c, _ = _client(workspace_client)
    response = App(name="agent", user_api_scopes=[])
    workspace_client.return_value.apps.create.return_value = mock.Mock(response=response)
    workspace_client.return_value.apps.get.return_value = response

    if operation == "create":
        c.create_app("agent", [])
        request = workspace_client.return_value.apps.create.call_args.args[0]
    else:
        c.update_app("agent", [])
        request = workspace_client.return_value.apps.create_update_and_wait.call_args.kwargs["app"]

    assert isinstance(request, App)
    assert request.as_dict() == {"name": "agent", "user_api_scopes": []}


@pytest.mark.parametrize("operation", ["get", "create", "update"])
@mock.patch("databricks_mason.client.WorkspaceClient")
def test_app_sdk_errors_are_wrapped_as_agent_cli_errors(workspace_client, operation):
    c, _ = _client(workspace_client)

    class AppsApiError(RuntimeError):
        error_code = "PERMISSION_DENIED"

    sdk_operation = "create_update_and_wait" if operation == "update" else operation
    getattr(workspace_client.return_value.apps, sdk_operation).side_effect = AppsApiError("denied")

    with pytest.raises(AgentCliError) as exc_info:
        if operation == "get":
            c.get_app("agent")
        elif operation == "create":
            c.create_app("agent", ["ai-gateway"])
        else:
            c.update_app("agent", ["ai-gateway"])

    assert exc_info.value.error_code == "PERMISSION_DENIED"
