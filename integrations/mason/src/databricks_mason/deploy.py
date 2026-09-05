"""`mason deploy` and the `mason deployments` group — manage agent deployments.

`mason deploy` is the integrated entry point: it can provision a memory store and a
session store for the agent, inject their identifiers into the deployment's `app.yaml`
env, then roll out the deployment. `mason deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Deployments run on the Databricks Apps runtime, which this module drives via the
`databricks apps` CLI — an implementation detail that is not part of Mason's surface.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import click
import yaml

from databricks_mason import memory_store_access, render, session_store_access, timefmt
from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import IntegrationRegistry, registry_relative_path
from databricks_mason.integrations import MCPService, Sandbox
from databricks_mason.project_config import (
    REQUEST_AUTH_CONTRACT_VERSION,
    load_project_metadata,
)
from databricks_mason.render import field
from databricks_mason.store_access import _databricks, apply_postgres_resources, grant_tables
from databricks_mason.tracing import TRACES_DEST_ENV, TRACES_EXPERIMENT_ENV, default_experiment

_MEMORY_ENV = "AGENT_MEMORY_STORE"
_MEMORY_ACTOR_ENV = "AGENT_MEMORY_ACTOR_ID"
_SESSION_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")

_PROJECT_CONFIG_PATH = pathlib.Path(".mason/project.toml")
_DEFAULT_REGISTRY_PATH = pathlib.Path("agent/databricks_tools.py")
_APP_NOT_FOUND_CODES = frozenset({"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"})
_USER_SCOPE_POLL_TIMEOUT_S = 60.0
_USER_SCOPE_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class _DeploymentAuthPlan:
    desired_scopes: tuple[str, ...]
    app_auth_integration_ids: tuple[str, ...]


@dataclass(frozen=True)
class _AppScopeState:
    requested_scopes: tuple[str, ...]
    effective_scopes: tuple[str, ...]
    changed: bool
    created: bool


def _deployment_auth_warnings(
    auth_plan: _DeploymentAuthPlan, scope_state: _AppScopeState
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if scope_state.changed:
        warnings.append(
            {
                "code": "USER_API_SCOPES_CHANGED",
                "message": "Users must re-consent to the App's requested user API scopes and "
                "remint their forwarded access token before invoking user-auth integrations.",
                "reconsent_required": True,
                "token_remint_required": True,
            }
        )
    if auth_plan.app_auth_integration_ids:
        integration_ids = list(auth_plan.app_auth_integration_ids)
        warnings.append(
            {
                "code": "APP_AUTH_SHARED_AUTHORITY",
                "message": "Shared-authority warning: every user who CAN USE this App can invoke "
                'these auth="app" integrations with the App service principal\'s authority: '
                f"{', '.join(integration_ids)}.",
                "integration_ids": integration_ids,
            }
        )
    return warnings


def _emit_deployment_auth_warnings(warnings: list[dict[str, Any]], output: str) -> None:
    for warning in warnings:
        if output == "json":
            click.echo(json.dumps({"warning": warning}, sort_keys=True), err=True)
        else:
            click.echo(f"Warning: {warning['message']}", err=True)


# --- databricks CLI plumbing (the deployment runtime) -----------------------


def _deployment_exists(name: str, profile: Optional[str]) -> bool:
    return _databricks(["apps", "get", name], profile, capture=True, check=False).returncode == 0


def _app_service_principal(name: str, profile: Optional[str]) -> Optional[str]:
    """The app's service principal client id (its Postgres role identity), or None if unavailable."""
    result = _databricks(["apps", "get", name, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("service_principal_client_id")
    except json.JSONDecodeError:
        return None


def _app_compute_state(name: str, profile: Optional[str]) -> Optional[str]:
    """The app's compute state (e.g. RUNNING), or None if it can't be read."""
    result = _databricks(["apps", "get", name, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("compute_status", {}).get("state")
    except json.JSONDecodeError:
        return None


def _validate_deployment_name(name: str) -> str:
    """Reject an empty or unsafe deployment name before it reaches a URL / workspace path."""
    if (
        not (name or "").strip()
        or name != name.strip()
        or any(token in name for token in ("/", "\\", ".."))
        or any(character.isspace() for character in name)
    ):
        raise AgentCliError(
            f"Invalid deployment name {name!r}.",
            hint="Use a non-empty name of letters, digits, and hyphens "
            "(no slashes, spaces, or '..').",
        )
    return name


def _confirm_destroy(target: str, *, assume_yes: bool) -> None:
    """Prompt before a destructive deployment op; --yes/-y skips it (for scripts)."""
    if assume_yes:
        return
    if not click.confirm(f"{target}? This cannot be undone.", default=False):
        raise click.Abort()


def _wait_for_running(name: str, profile: Optional[str], timeout_s: int = 300) -> None:
    """Block until a just-created app's compute is ACTIVE (or raise on timeout).

    `apps create` returns before compute is provisioned, but `apps deploy` requires the app to be
    ACTIVE — so a first deploy races without this wait.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _app_compute_state(name, profile) == "ACTIVE":
            return
        time.sleep(5)
    raise AgentCliError(
        f"App '{name}' did not reach a running state within {timeout_s}s.",
        hint=f"Check `mason deployments get {name}`, then re-run deploy once it's running.",
    )


def _load_deployment_auth_plan(source: pathlib.Path) -> _DeploymentAuthPlan | None:
    """Statically derive Apps user scopes and shared-App-auth integrations for a Mason source."""
    has_metadata = (source / _PROJECT_CONFIG_PATH).is_file()
    has_default_registry = (source / _DEFAULT_REGISTRY_PATH).is_file()
    if not has_metadata and not has_default_registry:
        return None

    metadata = load_project_metadata(source) if has_metadata else None
    relative_registry = (
        registry_relative_path(metadata.framework)
        if metadata is not None
        else _DEFAULT_REGISTRY_PATH
    )
    registry = IntegrationRegistry.load(source, relative_path=relative_registry)
    if registry.legacy_auth_ids:
        edits = "; ".join(
            f"{registry.path}:{registry.definition_line(integration_id)} ({integration_id!r}): "
            'add auth="user" or auth="app"'
            for integration_id in sorted(registry.legacy_auth_ids)
        )
        raise AgentCliError(
            "The canonical Databricks integration registry contains entries with unspecified "
            f"legacy auth: {', '.join(sorted(registry.legacy_auth_ids))}.",
            hint=f"Make these exact edits before deploying: {edits}.",
        )

    user_auth_ids = tuple(
        sorted(
            integration.id
            for integration in registry.integrations
            if integration.required_user_scopes
        )
    )
    declared_extra_scopes = metadata.extra_user_api_scopes if metadata is not None else ()
    desired_scopes = set(declared_extra_scopes)
    for integration in registry.integrations:
        desired_scopes.update(integration.required_user_scopes)
    contract_version = metadata.request_auth_contract_version if metadata is not None else None
    if desired_scopes and contract_version != REQUEST_AUTH_CONTRACT_VERSION:
        reasons = []
        if user_auth_ids:
            reasons.append(f"user-auth integrations: {', '.join(user_auth_ids)}")
        if declared_extra_scopes:
            reasons.append(f"declared extra scopes: {', '.join(declared_extra_scopes)}")
        raise AgentCliError(
            f"Mason request-auth contract version {REQUEST_AUTH_CONTRACT_VERSION} is required "
            f"before deploying requested user API scopes ({'; '.join(reasons)}).",
            hint="Regenerate the project with the current `mason init` template, then migrate "
            "the request handler before setting request_auth_contract_version = 1.",
        )
    if declared_extra_scopes and not user_auth_ids:
        raise AgentCliError(
            "Mason request-auth contract version 1 cannot activate extra_user_api_scopes "
            "without a declarative user-auth integration.",
            hint='Add at least one MCPService or Sandbox with auth="user", or remove the extra '
            "scopes. Custom-only request-user auth requires a future runtime-policy contract.",
        )

    app_auth_ids = tuple(
        sorted(
            integration.id
            for integration in registry.integrations
            if isinstance(integration, (MCPService, Sandbox)) and integration.auth == "app"
        )
    )
    return _DeploymentAuthPlan(tuple(sorted(desired_scopes)), app_auth_ids)


def _app_scopes(app: Any, attribute: str) -> tuple[str, ...]:
    raw = getattr(app, attribute, None)
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(scope, str) for scope in raw):
        raise AgentCliError(f"Databricks Apps returned invalid {attribute} for {app.name!r}.")
    return tuple(sorted(raw))


def _poll_effective_user_api_scopes(
    client,
    name: str,
    desired_scopes: tuple[str, ...],
    *,
    poll_timeout_s: float,
    poll_interval_s: float,
    propagating_from_scopes: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    deadline = time.monotonic() + poll_timeout_s
    while True:
        app = client.get_app(name)
        requested = _app_scopes(app, "user_api_scopes")
        if requested != desired_scopes:
            if requested == propagating_from_scopes:
                if time.monotonic() < deadline:
                    if poll_interval_s > 0:
                        time.sleep(poll_interval_s)
                    continue
                raise AgentCliError(
                    f"App '{name}' did not make requested user API scopes visible within "
                    f"{poll_timeout_s:g}s (expected {list(desired_scopes)!r}, still found "
                    f"{list(requested)!r}).",
                    hint="The Apps control plane may still be propagating the write; re-run deploy "
                    "after checking the App's requested scopes.",
                )
            raise AgentCliError(
                f"App '{name}' user API scopes changed concurrently while Mason was reconciling "
                f"them (expected {list(desired_scopes)!r}, found {list(requested)!r}).",
                hint="Review the App's requested scopes and re-run deploy; Mason did not overwrite "
                "the concurrent change.",
            )
        effective = _app_scopes(app, "effective_user_api_scopes")
        missing = sorted(set(desired_scopes) - set(effective))
        if not missing:
            return requested, effective
        if time.monotonic() >= deadline:
            raise AgentCliError(
                f"App '{name}' did not make required user API scopes effective within "
                f"{poll_timeout_s:g}s: {', '.join(missing)}.",
                hint="Ask a workspace administrator to allow the scopes, then re-run deploy. "
                "Source sync and App deployment were not started.",
            )
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)


def _reconcile_user_api_scopes(
    client,
    name: str,
    desired_scopes: tuple[str, ...],
    *,
    confirm_removal: bool,
    poll_timeout_s: float = _USER_SCOPE_POLL_TIMEOUT_S,
    poll_interval_s: float = _USER_SCOPE_POLL_INTERVAL_S,
) -> _AppScopeState:
    """Create or exactly reconcile one Mason App's requested/effective user API scopes."""
    desired = tuple(sorted(desired_scopes))
    try:
        app = client.get_app(name)
    except AgentCliError as exc:
        if exc.error_code not in _APP_NOT_FOUND_CODES:
            raise
        client.create_app(name, list(desired))
        requested, effective = _poll_effective_user_api_scopes(
            client,
            name,
            desired,
            poll_timeout_s=poll_timeout_s,
            poll_interval_s=poll_interval_s,
            propagating_from_scopes=(),
        )
        return _AppScopeState(requested, effective, changed=bool(desired), created=True)

    initial_requested = _app_scopes(app, "user_api_scopes")
    stale_scopes = tuple(sorted(set(initial_requested) - set(desired)))
    if stale_scopes and not confirm_removal:
        raise AgentCliError(
            f"App '{name}' currently requests user API scopes Mason would remove: "
            f"{', '.join(stale_scopes)}.",
            hint=f"Adopt {json.dumps(list(stale_scopes))} by adding each scope to "
            "extra_user_api_scopes in .mason/project.toml, "
            "or explicitly authorize their exact removal by re-running with "
            "--confirm-user-scope-removal.",
        )

    changed = initial_requested != desired
    if changed:
        latest = client.get_app(name)
        latest_requested = _app_scopes(latest, "user_api_scopes")
        if latest_requested != initial_requested:
            raise AgentCliError(
                f"App '{name}' user API scopes changed concurrently before Mason's update "
                f"(initially {list(initial_requested)!r}, now {list(latest_requested)!r}).",
                hint="Review the App's requested scopes and re-run deploy; Mason did not overwrite "
                "the concurrent change.",
            )
        client.update_app(name, list(desired))

    requested, effective = _poll_effective_user_api_scopes(
        client,
        name,
        desired,
        poll_timeout_s=poll_timeout_s,
        poll_interval_s=poll_interval_s,
        propagating_from_scopes=initial_requested if changed else None,
    )
    return _AppScopeState(requested, effective, changed=changed, created=False)


# --- app.yaml manifest handling ---------------------------------------------


def _upsert_manifest_env(source: pathlib.Path, updates: dict[str, str]) -> bool:
    """Inject/overwrite env entries in <source>/app.yaml. Returns True if it scaffolded a new file."""
    app_yaml = source / "app.yaml"
    if app_yaml.exists():
        loaded = yaml.safe_load(app_yaml.read_text())
        doc: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        scaffolded = False
    else:
        doc = {"command": ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"], "env": []}
        scaffolded = True

    raw_env = doc.get("env")
    env: list[dict[str, Any]] = (
        [entry for entry in raw_env if isinstance(entry, dict)] if isinstance(raw_env, list) else []
    )
    by_name = {e.get("name"): e for e in env if isinstance(e, dict)}
    for name, value in updates.items():
        if name in by_name:
            by_name[name]["value"] = value
            by_name[name].pop("valueFrom", None)
        else:
            env.append({"name": name, "value": value})
    doc["env"] = env
    app_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    return scaffolded


# --- store provisioning -----------------------------------------------------


_MEMORY_STORE_PAGE_SIZE = 100  # the memory-stores list API caps page_size at 100


def _resolve_memory_store(client, display_name: str) -> Optional[dict]:
    """Find a memory store by display name, paging through the list, or None if none matches.

    `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
    pass, so resolving a name means listing and matching on `display_name`. The list API caps
    `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
    """
    page_token: Optional[str] = None
    while True:
        listing = client.list_memory_stores(
            page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
        )
        for store in field(listing, "managed_memory_stores") or []:
            if field(store, "display_name") == display_name:
                return store
        page_token = field(listing, "next_page_token")
        if not page_token:
            return None


def _ensure_memory_store(client, display_name: str) -> dict:
    try:
        return client.create_memory_store(display_name)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    store = _resolve_memory_store(client, display_name)
    if store is None:
        raise AgentCliError(f"Memory store '{display_name}' exists but could not be resolved.")
    return store


def _ensure_session_store(client, name: str) -> dict:
    try:
        return client.create_session_store(name)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    return client.get_session_store(name)


def _memory_store_database(client, memory_store: str) -> Optional[str]:
    """Resolve the memory store's per-store Lakebase database name from its storage backend.

    Resolves by display name (what the deploy flag carries), not get_memory_store (which is by id).
    """
    store = _resolve_memory_store(client, memory_store)
    if store is None:
        return None
    backend_id = field(field(store, "storage_backend") or {}, "backend_id")
    return memory_store_access.database_from_backend_id(backend_id) if backend_id else None


def resolve_store_env(
    client,
    *,
    app: Optional[str],
    memory_store: Optional[str],
    session_store: Optional[str],
    traces_destination: Optional[str],
    traces_experiment: Optional[str],
    create_stores: bool,
) -> dict[str, str]:
    """Resolve store/trace references to the AGENT_*/MLFLOW_* env vars that wire them in.

    Shared by `mason deploy` and `mason dev` so both wire an agent's stores into app.yaml the same
    way. With `create_stores` (the default), missing stores are created (idempotent); when it is off
    they must already exist. The memory store resolves to its bare id (the runtime re-adds the
    `memory-stores/` prefix when building the entries URL); the session store and trace destination
    are used verbatim.
    """
    env: dict[str, str] = {}
    if memory_store:
        # Resolve by display name (what users pass) in both cases: get_memory_store looks up by
        # resource id, so it can't resolve a display name on the non-create path.
        if create_stores:
            store = _ensure_memory_store(client, memory_store)
        else:
            store = _resolve_memory_store(client, memory_store)
            if store is None:
                raise AgentCliError(
                    f"Memory store '{memory_store}' does not exist "
                    "(drop --no-create-stores to create it)."
                )
        store_name = field(store, "name") or memory_store
        env[_MEMORY_ENV] = store_name.split("/", 1)[-1]
    if session_store:
        if create_stores:
            _ensure_session_store(client, session_store)
        else:
            # Validate existence up front so a typo fails at deploy time, not at runtime.
            try:
                client.get_session_store(session_store)
            except AgentCliError as exc:
                raise AgentCliError(
                    f"Session store '{session_store}' does not exist "
                    "(drop --no-create-stores to create it).",
                    error_code=exc.error_code,
                ) from exc
        env[_SESSION_ENV] = session_store
    if traces_destination:
        env[TRACES_DEST_ENV] = traces_destination
        # The agent enables tracing only when BOTH a destination and an experiment are set, so
        # default the experiment to this agent's per-app path (matching `mason tracing setup --app`),
        # otherwise --with-traces alone would ship a half-config that silently disables tracing.
        env[TRACES_EXPERIMENT_ENV] = traces_experiment or default_experiment(
            client.current_user, app
        )
    elif traces_experiment:
        env[TRACES_EXPERIMENT_ENV] = traces_experiment
    return env


def _grant_store_access(
    app: str,
    sp: str,
    owner: str,
    session_store: Optional[str],
    memory_database: Optional[str],
    profile: Optional[str],
) -> Optional[str]:
    """Give the app's SP access to the deployed stores (best-effort, two steps).

    Binds every store's database as a `postgres` app resource in one update (the update replaces the
    whole resource array, so they must be applied together), then GRANTs the SP read/write on each
    store's tables. Returns None on success or a human-readable reason on the first failure.
    """
    backends = []
    if session_store:
        backends.append(session_store_access.backend(session_store))
    if memory_database:
        backends.append(memory_store_access.backend(memory_database))
    if not backends:
        return None

    error = apply_postgres_resources(app, backends, profile)
    if error:
        return error
    for backend in backends:
        error = grant_tables(backend, sp, owner, profile)
        if error:
            return error
    return None


# --- mason deploy -----------------------------------------------------------


@click.command()
@click.argument("name")
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory for the deployment (containing app.yaml). Defaults to the "
    "current directory.",
)
@click.option(
    "--memory",
    "-m",
    "memory_store",
    default=None,
    help="Memory store display name to wire in via AGENT_MEMORY_STORE.",
)
@click.option(
    "--session",
    "-s",
    "session_store",
    default=None,
    help="Session store name to wire in via AGENT_SESSION_STORE.",
)
@click.option(
    "--actor-id",
    default="agent",
    show_default=True,
    help="Actor id used for managed memory entries and sessions.",
)
@click.option(
    "--with-traces",
    "traces_destination",
    default=None,
    help="UC trace destination 'catalog.schema' to wire in via MLFLOW_TRACING_DESTINATION "
    "(link it first with `mason tracing setup`).",
)
@click.option(
    "--traces-experiment",
    default=None,
    help="MLflow experiment path to wire in via MLFLOW_EXPERIMENT_NAME.",
)
@click.option(
    "--no-create-stores",
    is_flag=True,
    help="Require referenced stores to already exist. By default missing stores are created "
    "(idempotent).",
)
@click.option(
    "--pip-index-url",
    default=_DEFAULT_PIP_INDEX_URL,
    show_default=True,
    help="Package index the Apps build installs from. Defaults to public PyPI as a temporary "
    "workaround: the Apps build environment currently can't reach the internal proxy. Pass an "
    "empty string to use the build's default index.",
)
@click.option(
    "--workspace-path",
    default=None,
    help="Workspace destination for the synced source (defaults to a per-user path).",
)
@click.option(
    "--confirm-user-scope-removal",
    is_flag=True,
    help="Authorize removal of existing App user API scopes not declared by this Mason project.",
)
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    memory_store,
    session_store,
    actor_id,
    traces_destination,
    traces_experiment,
    no_create_stores,
    pip_index_url,
    workspace_path,
    confirm_user_scope_removal,
) -> None:
    """Deploy an agent: provision its stores, wire them in, and roll out the deployment."""
    _validate_deployment_name(name)
    source_dir = pathlib.Path(source)
    auth_plan = _load_deployment_auth_plan(source_dir)
    client = obj.client()

    # A canonical Mason registry owns the App's exact requested user scopes. Reconcile before
    # provisioning stores or syncing source so an unsafe removal, concurrent edit, or unavailable
    # effective scope fails without any downstream deployment mutation.
    scope_state: _AppScopeState | None = None
    auth_warnings: list[dict[str, Any]] = []
    if auth_plan is not None:
        scope_state = _reconcile_user_api_scopes(
            client,
            name,
            auth_plan.desired_scopes,
            confirm_removal=confirm_user_scope_removal,
        )
        auth_warnings = _deployment_auth_warnings(auth_plan, scope_state)
        _emit_deployment_auth_warnings(auth_warnings, obj.output)
        if scope_state.created:
            _wait_for_running(name, obj.profile)

    # 1. Provision / resolve stores and build the env to inject.
    env_updates = resolve_store_env(
        client,
        app=name,
        memory_store=memory_store,
        session_store=session_store,
        traces_destination=traces_destination,
        traces_experiment=traces_experiment,
        create_stores=not no_create_stores,
    )
    provisioned: dict[str, Any] = {}
    if _MEMORY_ENV in env_updates:
        provisioned["Memory store"] = env_updates[_MEMORY_ENV]
        env_updates[_MEMORY_ACTOR_ENV] = actor_id
    if _SESSION_ENV in env_updates:
        provisioned["Session store"] = env_updates[_SESSION_ENV]
        env_updates[_SESSION_ACTOR_ENV] = actor_id
    if traces_destination:
        provisioned["Traces"] = traces_destination
    if pip_index_url:
        for env in _PIP_INDEX_ENVS:
            env_updates[env] = pip_index_url
        provisioned["Package index"] = pip_index_url
    if auth_plan is not None:
        provisioned["User API scopes"] = ", ".join(auth_plan.desired_scopes) or "none"

    # 2. Patch the app.yaml manifest with the store identifiers.
    scaffolded = False
    if env_updates:
        scaffolded = _upsert_manifest_env(source_dir, env_updates)

    # 3. Roll out the deployment (Databricks Apps runtime).
    if auth_plan is None and not _deployment_exists(name, obj.profile):
        _databricks(["apps", "create", name], obj.profile)
        # `apps create` returns before the app's compute is up, but `apps deploy` requires it to be
        # RUNNING — so wait for it, or the first deploy races and fails ("not in RUNNING state").
        _wait_for_running(name, obj.profile)
    ws_path = workspace_path or f"/Workspace/Users/{client.current_user}/mason_deployments/{name}"
    # Don't ship uv.lock: it pins exact package URLs from whatever index the developer's machine
    # resolved against (often an internal proxy). The Apps build must resolve against its own
    # configured index, so let it lock fresh in-sandbox instead of inheriting the local lock.
    _databricks(["sync", str(source_dir), ws_path, "--exclude", "uv.lock"], obj.profile)
    _databricks(["apps", "deploy", name, "--source-code-path", ws_path], obj.profile)

    # 4. Give the app's SP access to its stores (best-effort, two steps): bind each store's database
    #    as a `postgres` resource (CONNECT), then GRANT the SP read/write on its tables. Without
    #    both, the app runs but the durable store path fails (can't connect, or can't read tables).
    grants_stores = bool(session_store or memory_store)
    grant_error: Optional[str] = None
    if grants_stores:
        sp = _app_service_principal(name, obj.profile)
        if sp is None:
            grant_error = "could not resolve the app's service principal."
        else:
            memory_database = _memory_store_database(client, memory_store) if memory_store else None
            grant_error = _grant_store_access(
                name, sp, client.current_user, session_store, memory_database, obj.profile
            )

    if obj.output == "json":
        payload = {
            "deployment": name,
            "workspace_path": ws_path,
            "env": env_updates,
            "store_grant": "skipped"
            if not grants_stores
            else ("granted" if grant_error is None else "failed"),
            "store_grant_error": grant_error,
        }
        if auth_plan is not None and scope_state is not None:
            payload["user_api_scopes"] = {
                "desired": list(auth_plan.desired_scopes),
                "requested": list(scope_state.requested_scopes),
                "effective": list(scope_state.effective_scopes),
                "changed": scope_state.changed,
            }
            payload["app_auth_integration_ids"] = list(auth_plan.app_auth_integration_ids)
            payload["warnings"] = auth_warnings
        render.emit_json(payload)
        return

    steps = [f"mason deployments logs {name}", f"mason deployments get {name}"]
    if scaffolded:
        steps.insert(
            0, f"Set a real `command:` in {source_dir / 'app.yaml'} (a placeholder was written)"
        )
    if grants_stores and grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its store tables; that grant couldn't "
            "be applied automatically (it requires store ownership). "
            f"Cause: {grant_error}",
        )
    if grants_stores and grant_error is None:
        provisioned["Store access"] = "granted to app service principal"
    render.success(
        f"Deployed agent '{name}'",
        fields={"Workspace path": ws_path, **provisioned},
        next_steps=steps,
    )


# --- mason deployments <lifecycle> ------------------------------------------


@click.group()
def deployments() -> None:
    """Manage agent deployments."""


def _deployment_status(a: dict) -> Optional[str]:
    for key in ("app_status", "compute_status"):
        section = a.get(key)
        if isinstance(section, dict) and field(section, "state"):
            return field(section, "state")
    return field(a, "state")


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List agent deployments in the workspace."""
    result = _databricks(["apps", "list", "-o", "json"], obj.profile, capture=True)
    data = json.loads(result.stdout or "[]")
    items = data.get("apps", data) if isinstance(data, dict) else data
    if obj.output == "json":
        render.emit_json(items)
        return
    rows = [
        [
            field(a, "name"),
            render.status_pill(_deployment_status(a)),
            field(a, "url"),
            timefmt.relative(field(a, "update_time")),
        ]
        for a in items
    ]
    render.resource_table(
        "Agent Deployments",
        [("Name", "left"), ("Status", "left"), ("URL", "left"), ("Updated", "left")],
        rows,
    )


@deployments.command("get")
@click.argument("name")
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    _validate_deployment_name(name)
    result = _databricks(["apps", "get", name, "-o", "json"], obj.profile, capture=True)
    data = json.loads(result.stdout or "{}")
    if obj.output == "json":
        render.emit_json(data)
        return
    url = field(data, "url")
    render.detail(
        "Agent Deployment",
        field(data, "name") or name,
        {
            "URL": url,
            "Description": field(data, "description"),
            "Created": timefmt.absolute(field(data, "create_time")),
            "Updated": timefmt.absolute(field(data, "update_time")),
        },
        status=_deployment_status(data),
        snippets=[("open", "bash", f"open {url}")] if url else None,
    )


@deployments.command("logs")
@click.argument("name")
@click.pass_obj
def deployments_logs(obj, name) -> None:
    """Stream a deployment's logs."""
    _validate_deployment_name(name)
    _databricks(["apps", "logs", name], obj.profile)


@deployments.command("start")
@click.argument("name")
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    _validate_deployment_name(name)
    _databricks(["apps", "start", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"started": name})
        return
    render.success(f"Started deployment '{name}'")


@deployments.command("stop")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_stop(obj, name, yes) -> None:
    """Stop a deployment."""
    _validate_deployment_name(name)
    _confirm_destroy(f"Stop deployment '{name}'", assume_yes=yes)
    _databricks(["apps", "stop", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"stopped": name})
        return
    render.success(f"Stopped deployment '{name}'")


@deployments.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_delete(obj, name, yes) -> None:
    """Delete a deployment."""
    _validate_deployment_name(name)
    _confirm_destroy(f"Delete deployment '{name}'", assume_yes=yes)
    _databricks(["apps", "delete", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")
