from __future__ import annotations

import importlib
import pickle
from dataclasses import FrozenInstanceError, asdict
from typing import Any, cast

import pytest

from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction


def _auth_module():
    return importlib.import_module("databricks_mason.runtime.auth")


def _value_reaches(value: Any, target: Any, seen: set[int]) -> bool:
    if id(value) in seen:
        return False
    seen.add(id(value))
    if value is target:
        return True
    if isinstance(target, str) and isinstance(value, str):
        return target in value
    if isinstance(value, dict):
        return any(
            _value_reaches(key, target, seen) or _value_reaches(item, target, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_value_reaches(item, target, seen) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict) and _value_reaches(attributes, target, seen):
        return True
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            try:
                attribute = getattr(value, slot)
            except AttributeError:
                continue
            if _value_reaches(attribute, target, seen):
                return True
    return False


def _assert_auth_traceback_cannot_reach(
    error: BaseException, module_name: str, *targets: Any
) -> None:
    production_locals: list[dict[str, Any]] = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__") == module_name:
            production_locals.append(frame.f_locals)
        traceback = traceback.tb_next
    assert production_locals
    for target in targets:
        for local_values in production_locals:
            assert not _value_reaches(local_values, target, set())


def test_deployed_context_lazily_caches_user_and_app_clients(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456")
    user_constructions: list[dict[str, Any]] = []
    app_client = object()
    app_constructions = 0

    class _WorkspaceClient:
        def __init__(self, **kwargs: Any) -> None:
            user_constructions.append(kwargs)

    def default_workspace_client():
        nonlocal app_constructions
        app_constructions += 1
        return app_client

    monkeypatch.setattr(auth, "WorkspaceClient", _WorkspaceClient)
    monkeypatch.setattr(auth, "_default_workspace_client", default_workspace_client)
    context = auth.RequestAuthContext.from_forwarded_token("request-user-token")

    user_client = context.client_for("user")

    assert context.client_for("user") is user_client
    assert context.client_for("app") is app_client
    assert context.client_for("app") is app_client
    assert user_constructions == [
        {
            "host": "https://workspace.example.com",
            "token": "request-user-token",
            "auth_type": "pat",
            "custom_headers": {"X-Databricks-Org-Id": "123456"},
        }
    ]
    assert app_constructions == 1


def test_deployed_user_requires_only_the_apps_forwarded_token(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("Authorization", "Bearer unrelated-authorization")
    monkeypatch.setattr(
        auth,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not fall back to App auth")),
    )
    context = auth.RequestAuthContext.from_forwarded_token(None)

    with pytest.raises(auth.MissingUserAuthorization) as exc_info:
        context.client_for("user")

    rendered = str(exc_info.value)
    assert "user authorization" in rendered.lower()
    assert "unrelated-authorization" not in rendered


def test_local_context_ignores_forwarded_token_and_uses_profile_resolution(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    constructions: list[object] = []

    def default_workspace_client():
        client = object()
        constructions.append(client)
        return client

    monkeypatch.setattr(auth, "_default_workspace_client", default_workspace_client)
    monkeypatch.setattr(
        auth,
        "WorkspaceClient",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"local auth must not consume forwarded token: {kwargs}")
        ),
    )
    context = auth.RequestAuthContext.from_forwarded_token("ignored-forwarded-token")

    user_client = context.client_for("user")
    app_client = context.client_for("app")

    assert context.client_for("user") is user_client
    assert context.client_for("app") is app_client
    assert constructions == [user_client, app_client]


def test_apps_run_local_uses_profile_resolution_with_app_metadata(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "app")
    monkeypatch.setenv("DATABRICKS_MASON_RUN_LOCAL", "1")
    constructions: list[object] = []

    def default_workspace_client():
        client = object()
        constructions.append(client)
        return client

    monkeypatch.setattr(auth, "_default_workspace_client", default_workspace_client)
    context = auth.RequestAuthContext.from_forwarded_token(None)

    user_client = context.client_for("user")
    app_client = context.client_for("app")

    assert constructions == [user_client, app_client]
    assert context.state_key("routing-session") == "routing-session"


def test_context_is_non_serializable_and_hides_forwarded_token(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    token = "secret-forwarded-token"
    context = auth.RequestAuthContext.from_forwarded_token(token)

    assert token not in repr(context)
    with pytest.raises(TypeError) as exc_info:
        pickle.dumps(context)
    assert token not in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        asdict(context)
    assert token not in str(exc_info.value)
    with pytest.raises(TypeError):
        asdict(auth.RequestAuthContext.from_forwarded_token(None))


def test_user_client_construction_failure_does_not_retain_token_in_exception(
    monkeypatch,
) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    token = "secret-forwarded-token"

    class _FailingWorkspaceClient:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError(f"SDK rejected {kwargs['token']}")

    monkeypatch.setattr(auth, "WorkspaceClient", _FailingWorkspaceClient)
    context = auth.RequestAuthContext.from_forwarded_token(token)

    with pytest.raises(auth.RequestAuthorizationError) as exc_info:
        context.client_for("user")

    assert token not in str(exc_info.value)
    assert token not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_auth_traceback_cannot_reach(exc_info.value, auth.__name__, token)


def test_user_client_constructor_base_exception_is_sanitized(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    token = "base-exception-token"

    class _ConstructorAbort(BaseException):
        pass

    class _FailingWorkspaceClient:
        def __init__(self, **kwargs: Any) -> None:
            raise _ConstructorAbort(f"SDK aborted with {kwargs['token']}")

    monkeypatch.setattr(auth, "WorkspaceClient", _FailingWorkspaceClient)
    context = auth.RequestAuthContext.from_forwarded_token(token)
    caught: BaseException | None = None

    try:
        context.client_for("user")
    except BaseException as error:
        caught = error

    assert caught is not None
    _assert_auth_traceback_cannot_reach(caught, auth.__name__, token)
    assert isinstance(caught, auth.RequestAuthorizationError)
    assert token not in str(caught)
    assert token not in repr(caught)
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert context.state_key("routing-session") == "routing-session"
    with pytest.raises(auth.MissingUserAuthorization):
        context.client_for("user")


@pytest.mark.parametrize("token", [" ", "\t", " \t "])
def test_deployed_whitespace_forwarded_token_is_missing(monkeypatch, token) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    constructions: list[dict[str, Any]] = []

    class _WorkspaceClient:
        def __init__(self, **kwargs: Any) -> None:
            constructions.append(kwargs)

    monkeypatch.setattr(auth, "WorkspaceClient", _WorkspaceClient)
    context = auth.RequestAuthContext.from_forwarded_token(token)

    with pytest.raises(auth.MissingUserAuthorization) as exc_info:
        context.client_for("user")

    _assert_auth_traceback_cannot_reach(exc_info.value, auth.__name__, token)
    assert constructions == []
    assert context.state_key("routing-session") == "routing-session"


def test_invalid_mode_failure_consumes_cached_user_authorization(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    token = "invalid-mode-token"
    user_client = object()
    monkeypatch.setattr(auth, "WorkspaceClient", lambda **kwargs: user_client)
    context = auth.RequestAuthContext.from_forwarded_token(token)

    assert context.client_for("user") is user_client
    assert context.client_for("user") is user_client
    assert context.state_key("routing-session") != "routing-session"

    with pytest.raises(ValueError, match="auth mode") as exc_info:
        context.client_for(cast(Any, "creator"))

    _assert_auth_traceback_cannot_reach(
        exc_info.value,
        auth.__name__,
        token,
        user_client,
    )
    assert context.state_key("routing-session") == "routing-session"
    with pytest.raises(auth.MissingUserAuthorization):
        context.client_for("user")


def test_app_client_failure_consumes_cached_user_authorization(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    token = "app-failure-token"
    user_client = object()
    monkeypatch.setattr(auth, "WorkspaceClient", lambda **kwargs: user_client)
    context = auth.RequestAuthContext.from_forwarded_token(token)

    assert context.client_for("user") is user_client
    assert context.state_key("routing-session") != "routing-session"
    monkeypatch.setattr(
        auth,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(RuntimeError("App client unavailable")),
    )

    with pytest.raises(RuntimeError, match="App client unavailable") as exc_info:
        context.client_for("app")

    _assert_auth_traceback_cannot_reach(
        exc_info.value,
        auth.__name__,
        token,
        user_client,
    )
    assert context.state_key("routing-session") == "routing-session"
    with pytest.raises(auth.MissingUserAuthorization):
        context.client_for("user")


def test_state_key_is_principal_workspace_app_and_session_bound(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    first = auth.RequestAuthContext.from_forwarded_token("token-a")
    second = auth.RequestAuthContext.from_forwarded_token("token-b")

    key = first.state_key("routing-session")

    assert key == "75db42126fb6dddaf851f38363f1fe52a396e85b2b97e104f689707156d76d12"
    assert first.state_key("routing-session") == key
    assert first.state_key("other-session") != key
    assert second.state_key("routing-session") != key

    monkeypatch.setenv("DATABRICKS_APP_NAME", "other-app")
    assert (
        auth.RequestAuthContext.from_forwarded_token("token-a").state_key("routing-session") != key
    )
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    monkeypatch.setenv("DATABRICKS_HOST", "https://other-workspace.example.com")
    assert (
        auth.RequestAuthContext.from_forwarded_token("token-a").state_key("routing-session") != key
    )


def test_tokenless_and_local_state_keys_retain_the_routing_session(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setenv("DATABRICKS_APP_NAME", "claims-app")
    assert auth.RequestAuthContext.from_forwarded_token(None).state_key("routing-session") == (
        "routing-session"
    )

    monkeypatch.delenv("DATABRICKS_APP_NAME")
    assert (
        auth.RequestAuthContext.from_forwarded_token("ignored-local-token").state_key(
            "routing-session"
        )
        == "routing-session"
    )


@pytest.mark.parametrize(
    ("integrations", "requires_user_credentials"),
    [
        ((), False),
        ((UCFunction(id="lookup", function="main.tools.lookup"),), False),
        ((MCPService(id="shared", service="main.tools.shared", auth="app"),), False),
        (
            (
                Sandbox(
                    id="python",
                    scopes=(Scope.volume("main.data.files"),),
                    auth="app",
                ),
            ),
            False,
        ),
        ((MCPService(id="user", service="main.tools.user", auth="user"),), True),
        (
            (
                MCPService(id="shared", service="main.tools.shared", auth="app"),
                MCPService(id="user", service="main.tools.user", auth="user"),
            ),
            True,
        ),
    ],
)
def test_invocation_auth_policy_is_immutable_and_derived_from_integrations(
    integrations,
    requires_user_credentials,
) -> None:
    auth = _auth_module()

    policy = auth.InvocationAuthPolicy.from_integrations(integrations)

    assert policy.requires_user_credentials is requires_user_credentials
    assert policy.allows_background is not requires_user_credentials
    with pytest.raises(FrozenInstanceError):
        policy.requires_user_credentials = False


def test_runtime_reexports_request_auth_contract() -> None:
    runtime = importlib.import_module("databricks_mason.runtime")
    auth = _auth_module()

    assert runtime.RequestAuthContext is auth.RequestAuthContext
    assert runtime.InvocationAuthPolicy is auth.InvocationAuthPolicy
    assert runtime.MissingUserAuthorization is auth.MissingUserAuthorization


def test_client_for_rejects_unknown_auth_mode_before_resolving_credentials(monkeypatch) -> None:
    auth = _auth_module()
    monkeypatch.setattr(
        auth,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve credentials")),
    )
    context = auth.RequestAuthContext.from_forwarded_token(None)

    with pytest.raises(ValueError, match="auth mode"):
        context.client_for(cast(Any, "creator"))
