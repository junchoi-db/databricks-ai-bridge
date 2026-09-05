"""Unit tests for the deploy wrapper: app.yaml env injection, store reuse, deploy argv."""

from __future__ import annotations

import inspect
import json
import pathlib
import types
from typing import Any, cast
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner
from databricks.sdk.errors import NotFound
from databricks.sdk.service.apps import App

from databricks_mason import deploy as deploy_mod
from databricks_mason.client import MasonClient
from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import render_registry
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction


def _separate_stderr_runner() -> CliRunner:
    if "mix_stderr" in inspect.signature(CliRunner).parameters:
        return cast(Any, CliRunner)(mix_stderr=False)
    return CliRunner()


def test_upsert_manifest_env_scaffolds_when_missing(tmp_path: pathlib.Path):
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "memory-stores/x"}
    )
    assert scaffolded is True
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert {"name": "AGENT_MEMORY_STORE", "value": "memory-stores/x"} in doc["env"]
    assert "command" in doc  # placeholder written


def test_upsert_manifest_env_updates_existing(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["uvicorn", "app:app"],
                "env": [{"name": "AGENT_MEMORY_STORE", "value": "old"}],
            }
        )
    )
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}
    )
    assert scaffolded is False
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["command"] == ["uvicorn", "app:app"]  # preserved
    by_name = {e["name"]: e["value"] for e in doc["env"]}
    assert by_name == {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}


def test_ensure_session_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_session_store.return_value = {"session_store_name": "s"}
    assert deploy_mod._ensure_session_store(client, "s") == {"session_store_name": "s"}


class _FakeClient:
    host = "https://ws"
    current_user = "me@example.com"

    def get_memory_store(self, name):
        return {"name": f"memory-stores/{name}"}

    def list_memory_stores(self, page_size=None, page_token=None):
        # One page; the store's resource name is an id distinct from its display name (as the real
        # API returns), so resolution must match on display_name, not id.
        return {
            "managed_memory_stores": [{"name": "memory-stores/mem-id-123", "display_name": "mem"}],
            "next_page_token": "",
        }

    def create_memory_store(self, display_name):
        # Create-if-not-exists is the deploy default; return the same id resolution would find.
        return {"name": "memory-stores/mem-id-123", "display_name": display_name}

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name):
        return {"session_store_name": name}


class _FakeCtx:
    profile = "prof"
    output = "text"

    def client(self):
        return _FakeClient()


def test_deploy_drives_sync_and_apps_deploy(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--memory", "mem"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    ws = "/Workspace/Users/me@example.com/mason_deployments/myapp"
    # uv.lock is excluded so the build resolves fresh against its own index (not the dev machine's).
    assert ["sync", str(src), ws, "--exclude", "uv.lock"] in calls
    assert ["apps", "deploy", "myapp", "--source-code-path", ws] in calls
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    # Display name "mem" resolves to store id memory-stores/mem-id-123; the runtime re-adds the
    # `memory-stores/` prefix, so the env var must carry the bare id.
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"


def test_deploy_sync_keeps_code_selected_tool_registry(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    registry = src / "agent" / "databricks_tools.py"
    registry.parent.mkdir()
    registry.write_text(render_registry([]))
    original_registry = registry.read_text()
    client = _ScopeClient([_app([])])
    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src)],
        obj=_ScopeCtx(client),
    )

    assert result.exit_code == 0, result.output
    sync = next(args for args in calls if args[0] == "sync")
    assert sync[:3] == [
        "sync",
        str(src),
        "/Workspace/Users/me@example.com/mason_deployments/myapp",
    ]
    excluded = {sync[index + 1] for index, value in enumerate(sync[:-1]) if value == "--exclude"}
    assert "agent/databricks_tools.py" not in excluded
    assert registry.read_text() == original_registry


def test_first_deploy_waits_for_running_before_deploying(tmp_path: pathlib.Path, monkeypatch):
    # A brand-new app isn't RUNNING right after `apps create`; deploy must wait, or it races and
    # fails ("not in RUNNING state"). Verify create -> wait -> sync/deploy ordering.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod, "_deployment_exists", lambda a, p: False
    )  # app doesn't exist yet
    waited = {"called": False}
    monkeypatch.setattr(
        deploy_mod, "_wait_for_running", lambda name, profile: waited.__setitem__("called", True)
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "myapp"] in calls
    assert waited["called"], "must wait for the new app to be running before deploying"
    # the wait happens after create and before sync/deploy
    create_i = calls.index(["apps", "create", "myapp"])
    sync_i = next(i for i, a in enumerate(calls) if a[:1] == ["sync"])
    assert create_i < sync_i


def test_wait_for_running_returns_when_compute_active(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, p: "ACTIVE")
    deploy_mod._wait_for_running("app", "prof", timeout_s=1)  # returns without raising


def test_wait_for_running_times_out(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, p: "STARTING")
    monkeypatch.setattr(deploy_mod.time, "sleep", lambda s: None)  # don't actually wait
    try:
        deploy_mod._wait_for_running("app", "prof", timeout_s=0)
        raise AssertionError("expected AgentCliError on timeout")
    except AgentCliError:
        pass


def test_deploy_injects_shared_actor_for_managed_stores(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(src),
            "--memory",
            "mem",
            "--session",
            "sessions",
            "--actor-id",
            "alice",
        ],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env["AGENT_MEMORY_ACTOR_ID"] == "alice"
    assert env["AGENT_SESSION_ACTOR_ID"] == "alice"


def test_deploy_with_traces_injects_tracing_env(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(src),
            "--with-traces",
            "cat.schema",
            "--traces-experiment",
            "/Shared/x",
        ],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    doc = yaml.safe_load((src / "app.yaml").read_text())
    env = {e["name"]: e["value"] for e in doc["env"]}
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Shared/x"


def test_resolve_memory_store_pages_at_100_and_matches_display_name():
    # The list API caps page_size at 100, so resolution must page (not request 1000) and match the
    # display name across pages.
    class _PagingClient:
        def __init__(self):
            self.calls = []

        def list_memory_stores(self, page_size=None, page_token=None):
            self.calls.append((page_size, page_token))
            if page_token is None:
                return {
                    "managed_memory_stores": [{"name": "memory-stores/a", "display_name": "other"}],
                    "next_page_token": "p2",
                }
            return {
                "managed_memory_stores": [{"name": "memory-stores/b", "display_name": "wanted"}],
                "next_page_token": "",
            }

    client = _PagingClient()
    store = deploy_mod._resolve_memory_store(client, "wanted")
    assert store is not None
    assert store["name"] == "memory-stores/b"  # found on page 2
    assert all(ps == 100 for ps, _ in client.calls)  # never exceeds the API cap
    assert [pt for _, pt in client.calls] == [None, "p2"]  # followed the page token


def test_resolve_memory_store_returns_none_when_absent():
    class _EmptyClient:
        def list_memory_stores(self, page_size=None, page_token=None):
            return {"managed_memory_stores": [], "next_page_token": ""}

    assert deploy_mod._resolve_memory_store(_EmptyClient(), "nope") is None


def test_memory_store_database_resolves_by_display_name():
    # The grant step derives the Lakebase db from the store; it must resolve by display name
    # (list+match), not get_memory_store (by id), or it 404s on the deploy flag's value.
    class _Client:
        def list_memory_stores(self, page_size=None, page_token=None):
            return {
                "managed_memory_stores": [
                    {
                        "name": "memory-stores/uuid-x",
                        "display_name": "mem",
                        "storage_backend": {
                            "backend_id": "projects/p/branches/production/databases/memory-uuidx"
                        },
                    }
                ],
                "next_page_token": "",
            }

    assert deploy_mod._memory_store_database(_Client(), "mem") == "memory-uuidx"


def test_deploy_no_create_stores_resolves_memory_by_display_name(
    tmp_path: pathlib.Path, monkeypatch
):
    # --no-create-stores path must resolve by display name (list+match), not get_memory_store (by id).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--memory", "mem", "--no-create-stores"],
        obj=_FakeCtx(),
    )
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"  # resolved by display name, bare id


def test_deploy_creates_missing_stores_by_default(tmp_path: pathlib.Path, monkeypatch):
    # Create-if-not-exists is now the default (no flag): a referenced store is created via the API.
    created: list[str] = []
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    class _CreatingClient(_FakeClient):
        def create_memory_store(self, display_name):
            created.append(display_name)
            return {"name": "memory-stores/mem-id-123", "display_name": display_name}

    class _Ctx(_FakeCtx):
        def client(self):
            return _CreatingClient()

    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(src), "--memory", "new-mem"], obj=_Ctx()
    )
    assert result.exit_code == 0, result.output
    assert created == ["new-mem"]  # created without any --create-stores flag


def test_deploy_accepts_short_store_flags(tmp_path: pathlib.Path, monkeypatch):
    # -m / -s are the short forms of --memory / --session.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "-m", "mem", "-s", "s"],
        obj=_FakeCtx(),
    )
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"
    assert env["AGENT_SESSION_STORE"] == "s"


def test_with_traces_defaults_the_experiment_per_app():
    # --with-traces alone must still set the experiment, or the agent ships tracing half-configured
    # (destination set, experiment missing) and silently disables it. The default is per-app, so
    # each agent's traces are isolated instead of piling into one shared experiment.
    env = deploy_mod.resolve_store_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment=None,
        create_stores=False,
    )
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Users/me@example.com/mason-traces/my-agent"


def test_with_traces_explicit_experiment_wins_over_per_app():
    env = deploy_mod.resolve_store_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment="/Shared/custom",
        create_stores=False,
    )
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Shared/custom"


def _run_deploy(src, monkeypatch, extra_args):
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    return CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(src), *extra_args], obj=_FakeCtx()
    )


def test_deploy_injects_public_pypi_index_by_default(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    result = _run_deploy(src, monkeypatch, [])
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    for name in ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"):
        assert env[name] == "https://pypi.org/simple/"


def test_deploy_empty_pip_index_disables_override(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    result = _run_deploy(src, monkeypatch, ["--pip-index-url", ""])
    assert result.exit_code == 0, result.output
    doc = yaml.safe_load((src / "app.yaml").read_text())
    env = {e["name"]: e["value"] for e in (doc.get("env") or [])}
    assert "PIP_INDEX_URL" not in env  # empty -> no override, use the build's default index


class _JsonCtx:
    profile = "prof"
    output = "json"


def test_lifecycle_commands_honor_json_output(monkeypatch):
    # start/stop/delete must emit JSON (not the Rich success panel) under --output json.
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    for command, key, args in (
        (deploy_mod.deployments_start, "started", ["myapp"]),
        (deploy_mod.deployments_stop, "stopped", ["myapp", "--yes"]),  # destructive: needs --yes
        (deploy_mod.deployments_delete, "deleted", ["myapp", "--yes"]),
    ):
        result = CliRunner().invoke(command, args, obj=_JsonCtx())
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {key: "myapp"}


def _app(requested: list[str], effective: list[str] | None = None, *, name: str = "myapp") -> App:
    return App(
        name=name,
        user_api_scopes=requested,
        effective_user_api_scopes=requested if effective is None else effective,
    )


class _ScopeClient(_FakeClient):
    def __init__(self, reads: list[App | AgentCliError]):
        self._reads = list(reads)
        self.get_app_calls: list[str] = []
        self.create_app_calls: list[tuple[str, list[str]]] = []
        self.update_app_calls: list[tuple[str, list[str]]] = []
        self.created_stores: list[str] = []

    def _next_read(self) -> App:
        value = self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]
        if isinstance(value, AgentCliError):
            raise value
        return value

    def get_app(self, name: str) -> App:
        self.get_app_calls.append(name)
        return self._next_read()

    def create_app(self, name: str, user_api_scopes: list[str]) -> App:
        self.create_app_calls.append((name, list(user_api_scopes)))
        return _app(list(user_api_scopes), [], name=name)

    def update_app(self, name: str, user_api_scopes: list[str]) -> App:
        self.update_app_calls.append((name, list(user_api_scopes)))
        return _app(list(user_api_scopes), [], name=name)

    def create_memory_store(self, display_name):
        self.created_stores.append(display_name)
        return super().create_memory_store(display_name)


class _ScopeCtx:
    profile = "prof"

    def __init__(self, client: _ScopeClient, output: str = "text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def _write_mason_source(
    root: pathlib.Path,
    integrations,
    *,
    contract_version: int | None = 1,
    extra_scopes: tuple[str, ...] = (),
) -> pathlib.Path:
    source = root / "mason-app"
    source.mkdir()
    (source / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    metadata = source / ".mason" / "project.toml"
    metadata.parent.mkdir()
    contract = (
        f"request_auth_contract_version = {contract_version}\n"
        if contract_version is not None
        else ""
    )
    metadata.write_text(
        'schema_version = 1\nframework = "langgraph"\ntemplate = "agent-langgraph"\n'
        f"{contract}extra_user_api_scopes = {json.dumps(list(extra_scopes))}\n"
    )
    registry = source / "agent" / "databricks_tools.py"
    registry.parent.mkdir()
    registry.write_text(render_registry(integrations))
    return source


def test_new_mason_app_includes_desired_scopes_in_initial_create() -> None:
    client = _ScopeClient(
        [
            AgentCliError("missing", error_code="RESOURCE_DOES_NOT_EXIST"),
            _app(["ai-gateway", "sql"], ["ai-gateway", "openid", "sql"]),
        ]
    )

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway", "sql"),
        confirm_removal=False,
        poll_timeout_s=0,
        poll_interval_s=0,
    )

    assert client.create_app_calls == [("myapp", ["ai-gateway", "sql"])]
    assert client.update_app_calls == []
    assert state.created is True
    assert state.changed is True
    assert state.requested_scopes == ("ai-gateway", "sql")
    assert state.effective_scopes == ("ai-gateway", "openid", "sql")


def test_new_mason_app_waits_for_requested_scopes_to_become_visible() -> None:
    client = _ScopeClient(
        [
            AgentCliError("missing", error_code="RESOURCE_DOES_NOT_EXIST"),
            _app([], []),
            _app(["ai-gateway"], ["ai-gateway", "openid"]),
        ]
    )

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway",),
        confirm_removal=False,
        poll_timeout_s=1,
        poll_interval_s=0,
    )

    assert client.create_app_calls == [("myapp", ["ai-gateway"])]
    assert state.requested_scopes == ("ai-gateway",)
    assert state.effective_scopes == ("ai-gateway", "openid")


def test_existing_mason_app_with_exact_scopes_is_idempotent() -> None:
    client = _ScopeClient([_app(["ai-gateway"], ["ai-gateway", "openid"])])

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway",),
        confirm_removal=False,
        poll_timeout_s=0,
        poll_interval_s=0,
    )

    assert client.create_app_calls == []
    assert client.update_app_calls == []
    assert state.changed is False
    assert state.requested_scopes == ("ai-gateway",)


def test_existing_mason_app_adds_desired_scopes_exactly() -> None:
    client = _ScopeClient(
        [
            _app(["sql"]),
            _app(["sql"]),
            _app(["ai-gateway", "sql"], ["ai-gateway", "openid", "sql"]),
        ]
    )

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway", "sql"),
        confirm_removal=False,
        poll_timeout_s=0,
        poll_interval_s=0,
    )

    assert client.update_app_calls == [("myapp", ["ai-gateway", "sql"])]
    assert state.changed is True


def test_existing_mason_app_waits_for_updated_scopes_to_become_visible() -> None:
    client = _ScopeClient(
        [
            _app([]),
            _app([]),
            _app([], []),
            _app(["ai-gateway"], ["ai-gateway", "openid"]),
        ]
    )

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway",),
        confirm_removal=False,
        poll_timeout_s=1,
        poll_interval_s=0,
    )

    assert client.update_app_calls == [("myapp", ["ai-gateway"])]
    assert state.requested_scopes == ("ai-gateway",)
    assert state.effective_scopes == ("ai-gateway", "openid")


def test_existing_unknown_scope_requires_explicit_adoption_or_removal() -> None:
    client = _ScopeClient([_app(["ai-gateway"])])

    with pytest.raises(AgentCliError) as exc_info:
        deploy_mod._reconcile_user_api_scopes(
            client,
            "myapp",
            (),
            confirm_removal=False,
            poll_timeout_s=0,
            poll_interval_s=0,
        )

    rendered = f"{exc_info.value} {exc_info.value.hint}"
    assert "ai-gateway" in rendered
    assert "extra_user_api_scopes" in rendered
    assert "--confirm-user-scope-removal" in rendered
    assert client.update_app_calls == []


def test_confirm_user_scope_removal_authorizes_exact_stale_scope_removal() -> None:
    client = _ScopeClient([_app(["ai-gateway"]), _app(["ai-gateway"]), _app([])])

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        (),
        confirm_removal=True,
        poll_timeout_s=0,
        poll_interval_s=0,
    )

    assert client.update_app_calls == [("myapp", [])]
    assert state.requested_scopes == ()
    assert state.changed is True


def test_scope_reconciliation_detects_drift_immediately_before_update() -> None:
    client = _ScopeClient([_app([]), _app(["sql"])])

    with pytest.raises(AgentCliError, match="concurrent"):
        deploy_mod._reconcile_user_api_scopes(
            client,
            "myapp",
            ("ai-gateway",),
            confirm_removal=False,
            poll_timeout_s=0,
            poll_interval_s=0,
        )

    assert client.update_app_calls == []


def test_scope_reconciliation_detects_requested_drift_after_update() -> None:
    client = _ScopeClient([_app([]), _app([]), _app(["sql"], ["sql"])])

    with pytest.raises(AgentCliError, match="concurrent"):
        deploy_mod._reconcile_user_api_scopes(
            client,
            "myapp",
            ("ai-gateway",),
            confirm_removal=False,
            poll_timeout_s=0,
            poll_interval_s=0,
        )

    assert client.update_app_calls == [("myapp", ["ai-gateway"])]


def test_requested_scope_propagation_timeout_is_not_reported_as_concurrent() -> None:
    client = _ScopeClient(
        [
            AgentCliError("missing", error_code="RESOURCE_DOES_NOT_EXIST"),
            _app([], []),
        ]
    )

    with pytest.raises(AgentCliError) as exc_info:
        deploy_mod._reconcile_user_api_scopes(
            client,
            "myapp",
            ("ai-gateway",),
            confirm_removal=False,
            poll_timeout_s=0,
            poll_interval_s=0,
        )

    assert "requested user API scopes" in str(exc_info.value)
    assert "visible" in str(exc_info.value)
    assert "concurrent" not in str(exc_info.value)


def test_scope_reconciliation_times_out_before_required_scopes_are_effective() -> None:
    client = _ScopeClient([_app(["ai-gateway"], [])])

    with pytest.raises(AgentCliError, match="effective"):
        deploy_mod._reconcile_user_api_scopes(
            client,
            "myapp",
            ("ai-gateway",),
            confirm_removal=False,
            poll_timeout_s=0,
            poll_interval_s=0,
        )


class _PreflightCtx:
    profile = "prof"
    output = "text"

    def __init__(self):
        self.client_calls = 0

    def client(self):
        self.client_calls += 1
        raise AssertionError("preflight must fail before constructing a client")


def test_deploy_rejects_legacy_missing_auth_with_exact_registry_edits(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="web", service="system.ai.web_search", auth="user")],
    )
    registry = source / "agent" / "databricks_tools.py"
    registry.write_text(registry.read_text().replace('        auth="user",\n', ""))
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: cli_calls.append(args),
    )
    ctx = _PreflightCtx()

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(source)], obj=ctx)

    output = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "web" in output
    assert 'auth="user"' in output
    assert 'auth="app"' in output
    assert "databricks_tools.py" in "".join(result.output.split())
    assert ctx.client_calls == 0
    assert cli_calls == []


def test_deploy_rejects_user_auth_with_legacy_request_handler_before_mutation(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="web", service="system.ai.web_search", auth="user")],
        contract_version=None,
    )
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: cli_calls.append(args),
    )
    ctx = _PreflightCtx()

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source), "--memory", "must-not-create"],
        obj=ctx,
    )

    output = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "request-auth" in output
    assert "version 1" in output
    assert ctx.client_calls == 0
    assert cli_calls == []


def test_deploy_rejects_declared_extra_scopes_without_request_handler_contract(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="app-mcp", service="system.ai.genie", auth="app")],
        contract_version=None,
        extra_scopes=("sql",),
    )
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: cli_calls.append(args),
    )
    ctx = _PreflightCtx()

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(source)], obj=ctx)

    output = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "request-auth" in output
    assert "version 1" in output
    assert "sql" in output
    assert ctx.client_calls == 0
    assert cli_calls == []


def test_deploy_rejects_extra_only_scopes_without_a_user_auth_integration(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="app-mcp", service="system.ai.genie", auth="app")],
        extra_scopes=("sql",),
    )
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: cli_calls.append(args),
    )
    ctx = _PreflightCtx()

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(source)], obj=ctx)

    output = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "extra_user_api_scopes" in output
    assert 'auth="user"' in output
    assert "contract version 1" in output
    assert ctx.client_calls == 0
    assert cli_calls == []


def test_generic_byo_deploy_never_reads_or_mutates_user_scopes(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = tmp_path / "byo"
    source.mkdir()
    (source / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    class _ByoClient(_FakeClient):
        def get_app(self, name):
            raise AssertionError("generic/BYO deploy must not read Apps scopes")

        def create_app(self, name, user_api_scopes):
            raise AssertionError("generic/BYO deploy must not create through scope reconciliation")

        def update_app(self, name, user_api_scopes):
            raise AssertionError("generic/BYO deploy must not mutate Apps scopes")

    class _ByoCtx(_FakeCtx):
        def client(self):
            return _ByoClient()

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda name, profile: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(source)], obj=_ByoCtx()
    )

    assert result.exit_code == 0, result.output
    assert any(call[0] == "sync" for call in calls)
    assert any(call[:3] == ["apps", "deploy", "myapp"] for call in calls)


def test_mason_deploy_derives_sorted_scope_union_and_safe_json_auth_output(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [
            MCPService(id="user-mcp", service="system.ai.web_search", auth="user"),
            MCPService(id="app-mcp", service="system.ai.genie", auth="app"),
            Sandbox(
                id="app-sandbox",
                scopes=(Scope.table("main.tools.items"),),
                auth="app",
            ),
            UCFunction(id="function", function="main.tools.lookup"),
        ],
        extra_scopes=("sql",),
    )
    client = _ScopeClient(
        [
            _app(["sql"]),
            _app(["sql"]),
            _app(["ai-gateway", "sql"], ["openid", "sql", "ai-gateway"]),
        ]
    )
    ctx = _ScopeCtx(client, output="json")
    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda name, profile: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = _separate_stderr_runner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(source)], obj=ctx
    )

    assert result.exit_code == 0, result.output
    assert client.update_app_calls == [("myapp", ["ai-gateway", "sql"])]
    payload = json.loads(result.stdout)
    assert payload["user_api_scopes"] == {
        "desired": ["ai-gateway", "sql"],
        "requested": ["ai-gateway", "sql"],
        "effective": ["ai-gateway", "openid", "sql"],
        "changed": True,
    }
    assert payload["app_auth_integration_ids"] == ["app-mcp", "app-sandbox"]
    warnings = {warning["code"]: warning for warning in payload["warnings"]}
    consent = warnings["USER_API_SCOPES_CHANGED"]
    assert consent["reconsent_required"] is True
    assert consent["token_remint_required"] is True
    assert "re-consent" in consent["message"]
    assert "remint" in consent["message"]
    shared = warnings["APP_AUTH_SHARED_AUTHORITY"]
    assert shared["integration_ids"] == ["app-mcp", "app-sandbox"]
    assert 'auth="app"' in shared["message"]
    assert "CAN USE" in shared["message"]
    assert any(call[0] == "sync" for call in calls)


def test_new_mason_deploy_uses_typed_create_then_existing_compute_wait(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="user-mcp", service="system.ai.web_search", auth="user")],
    )
    client = _ScopeClient(
        [
            AgentCliError("missing", error_code="RESOURCE_DOES_NOT_EXIST"),
            _app(["ai-gateway"]),
        ]
    )
    events: list[str] = []
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_wait_for_running",
        lambda name, profile: events.append("wait"),
    )

    def fake_databricks(args, profile, **kwargs):
        events.append(args[0])
        cli_calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        fake_databricks,
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source)],
        obj=_ScopeCtx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.create_app_calls == [("myapp", ["ai-gateway"])]
    assert not any(value[:2] == ["apps", "create"] for value in cli_calls)
    wait_index = events.index("wait")
    sync_index = events.index("sync")
    assert wait_index < sync_index


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_sdk_not_found_without_error_code_reaches_new_app_create(workspace_client) -> None:
    instance = workspace_client.return_value
    instance.config.host = "https://ws.example.com"
    instance.config.workspace_id = None
    instance.apps.get.side_effect = [NotFound("missing"), _app(["ai-gateway"])]
    instance.apps.create.return_value = mock.Mock(response=_app(["ai-gateway"], [], name="myapp"))
    client = MasonClient("prof")

    state = deploy_mod._reconcile_user_api_scopes(
        client,
        "myapp",
        ("ai-gateway",),
        confirm_removal=False,
        poll_timeout_s=0,
        poll_interval_s=0,
    )

    assert state.created is True
    request = instance.apps.create.call_args.args[0]
    assert request.as_dict() == {"name": "myapp", "user_api_scopes": ["ai-gateway"]}


def test_deploy_flag_authorizes_stale_ai_gateway_removal(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="app-mcp", service="system.ai.genie", auth="app")],
    )
    client = _ScopeClient([_app(["ai-gateway"]), _app(["ai-gateway"]), _app([])])
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(source),
            "--confirm-user-scope-removal",
        ],
        obj=_ScopeCtx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.update_app_calls == [("myapp", [])]


def test_app_only_legacy_contract_stays_app_auth_and_emits_shared_authority_warning(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [
            MCPService(id="app-mcp", service="system.ai.genie", auth="app"),
            Sandbox(
                id="app-sandbox",
                scopes=(Scope.volume("main.tools.files"),),
                auth="app",
            ),
        ],
        contract_version=None,
    )
    client = _ScopeClient([_app([])])
    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda name, profile: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source)],
        obj=_ScopeCtx(client),
    )

    output = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert client.update_app_calls == []
    assert "CAN USE" in output
    assert "app-mcp" in output
    assert "app-sandbox" in output


def test_scope_change_output_requires_reconsent_and_token_remint(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [
            MCPService(id="user-mcp", service="system.ai.web_search", auth="user"),
            MCPService(id="app-mcp", service="system.ai.genie", auth="app"),
        ],
    )
    client = _ScopeClient([_app([]), _app([]), _app(["ai-gateway"])])
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda name, profile: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source)],
        obj=_ScopeCtx(client),
    )

    output = " ".join(result.output.split()).lower()
    assert result.exit_code == 0, result.output
    assert "re-consent" in output
    assert "remint" in output
    assert "can use" in output
    assert "app-mcp" in output


def test_json_scope_warnings_survive_failure_after_reconciliation(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [
            MCPService(id="user-mcp", service="system.ai.web_search", auth="user"),
            MCPService(id="app-mcp", service="system.ai.genie", auth="app"),
        ],
    )
    client = _ScopeClient([_app([]), _app([]), _app(["ai-gateway"])])

    def fail_sync(args, profile, **kwargs):
        if args[0] == "sync":
            raise AgentCliError("sync failed after scope reconciliation")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", fail_sync)

    result = _separate_stderr_runner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source)],
        obj=_ScopeCtx(client, output="json"),
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    warning_records = []
    for line in result.stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "warning" in record:
            warning_records.append(record["warning"])
    assert {warning["code"] for warning in warning_records} == {
        "USER_API_SCOPES_CHANGED",
        "APP_AUTH_SHARED_AUTHORITY",
    }
    assert "re-consent" in result.stderr
    assert "remint" in result.stderr


def test_scope_reconciliation_failure_happens_before_stores_sync_or_deploy(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    source = _write_mason_source(
        tmp_path,
        [MCPService(id="user-mcp", service="system.ai.web_search", auth="user")],
    )
    client = _ScopeClient([_app(["ai-gateway"], [])])
    calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod,
        "_reconcile_user_api_scopes",
        mock.Mock(side_effect=AgentCliError("required user scopes are not effective")),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append(args),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(source), "--memory", "must-not-create"],
        obj=_ScopeCtx(client),
    )

    assert result.exit_code != 0
    assert client.created_stores == []
    assert calls == []
