"""Request-scoped Databricks authorization capabilities."""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from databricks.sdk import WorkspaceClient

from databricks_mason.integrations import AuthMode, Integration, MCPService, Sandbox
from databricks_mason.runtime.errors import (
    IntegrationClientResolutionError,
    InvalidAppAuthorization,
    MasonRuntimeError,
    MissingUserAuthorization,
    RequestAuthorizationError,
)
from databricks_mason.runtime.workspace import workspace_client as _default_workspace_client
from databricks_mason.runtime.workspace import workspace_headers

_UNSET = object()
_RUN_LOCAL_ENV = "DATABRICKS_MASON_RUN_LOCAL"


def is_deployed_app() -> bool:
    """Return whether Mason is running behind deployed Databricks Apps ingress."""

    return bool(os.getenv("DATABRICKS_APP_NAME", "").strip()) and (
        os.getenv(_RUN_LOCAL_ENV, "").strip() != "1"
    )


class _ForwardedToken:
    """A bearer wrapper that refuses the deepcopy path used by dataclass serialization."""

    def __init__(self, value: str | None = None) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        raise TypeError("RequestAuthContext cannot be serialized.")


def _user_workspace_client(
    *,
    host: str,
    token: str,
    headers: dict[str, str],
) -> WorkspaceClient | None:
    """Construct a user client without retaining credential locals in caller tracebacks."""

    kwargs: dict[str, Any] = {
        "host": host,
        "token": token,
        "auth_type": "pat",
    }
    if headers:
        kwargs["custom_headers"] = headers
    try:
        return WorkspaceClient(**kwargs)
    except BaseException:
        return None


@dataclass
class RequestAuthContext:
    """A non-serializable, request-scoped capability for Databricks clients."""

    _forwarded_token: _ForwardedToken = field(default_factory=_ForwardedToken, repr=False)
    _app_name: str = field(init=False, repr=False)
    _host: str = field(init=False, repr=False)
    _user_client: Any = field(default=_UNSET, init=False, repr=False)
    _app_client: Any = field(default=_UNSET, init=False, repr=False)

    def __post_init__(self) -> None:
        self._app_name = os.getenv("DATABRICKS_APP_NAME", "").strip() if is_deployed_app() else ""
        self._host = os.getenv("DATABRICKS_HOST", "").strip().rstrip("/")
        if not self._app_name:
            self._forwarded_token = _ForwardedToken()

    @classmethod
    def from_forwarded_token(cls, token: str | None) -> RequestAuthContext:
        """Create a context from the header value supplied by the Databricks Apps boundary."""

        if token is not None and not isinstance(token, str):
            raise TypeError("The Apps-forwarded credential must be text.")
        normalized = (token.strip() or None) if token is not None else None
        return cls(_forwarded_token=_ForwardedToken(normalized))

    def client_for(self, mode: AuthMode) -> WorkspaceClient:
        """Return the lazily cached client for one configured authorization mode."""

        try:
            return self._client_for(mode)
        except BaseException:
            self._clear_user_authorization()
            raise

    def _client_for(self, mode: AuthMode) -> WorkspaceClient:
        if mode not in ("user", "app"):
            raise ValueError(f"Unsupported request auth mode {mode!r}.")
        cache_name = "_user_client" if mode == "user" else "_app_client"
        cached = getattr(self, cache_name)
        if cached is not _UNSET:
            return cached

        if mode == "app" or not self._app_name:
            client = _default_workspace_client()
        else:
            if self._forwarded_token.value is None:
                raise MissingUserAuthorization
            client = _user_workspace_client(
                host=self._host,
                token=self._forwarded_token.value,
                headers=workspace_headers(),
            )
            if client is None:
                raise RequestAuthorizationError(
                    "Databricks request-user authorization could not be initialized."
                )

        setattr(self, cache_name, client)
        return client

    def _clear_user_authorization(self) -> None:
        self._forwarded_token = _ForwardedToken()
        self._user_client = _UNSET

    def state_key(self, routing_session: str) -> str:
        """Return a token-free state namespace bound to the active request principal."""

        forwarded_token = self._forwarded_token
        if not self._app_name or forwarded_token.value is None:
            return routing_session
        message = f"{self._host}|{self._app_name}|{routing_session}".encode()
        return hmac.new(forwarded_token.value.encode(), message, hashlib.sha256).hexdigest()

    def __getstate__(self) -> None:
        raise TypeError("RequestAuthContext cannot be serialized.")


@dataclass(frozen=True)
class InvocationAuthPolicy:
    """Credential-lifetime requirements for one declared integration sequence."""

    requires_user_credentials: bool

    @classmethod
    def from_integrations(cls, integrations: Sequence[Integration]) -> InvocationAuthPolicy:
        return cls(
            requires_user_credentials=any(
                bool(integration.required_user_scopes) for integration in integrations
            )
        )

    @property
    def allows_background(self) -> bool:
        return not self.requires_user_credentials


def _integration_auth_mode(integration: Integration) -> AuthMode:
    if isinstance(integration, (MCPService, Sandbox)):
        return integration.auth
    return "app"


def integration_client_resolver(
    integrations: Sequence[Integration],
    *,
    workspace_client: WorkspaceClient | None,
    workspace_client_for: Callable[[AuthMode], WorkspaceClient] | None,
    default_workspace_client: Callable[[], WorkspaceClient],
) -> Callable[[Integration], WorkspaceClient]:
    """Build a safe per-integration client resolver for framework adapters."""

    if workspace_client is not None and workspace_client_for is not None:
        raise ValueError("Pass only one of workspace_client or workspace_client_for.")

    if workspace_client is not None:

        def resolve_explicit(_mode: AuthMode) -> WorkspaceClient:
            return workspace_client

        resolver = resolve_explicit
    elif workspace_client_for is not None:
        resolver = workspace_client_for
    else:
        if is_deployed_app():
            if user_integration := next(
                (
                    integration
                    for integration in integrations
                    if _integration_auth_mode(integration) == "user"
                ),
                None,
            ):
                raise MissingUserAuthorization(user_integration.id)

        default_client: WorkspaceClient | None = None

        def resolve_default(_mode: AuthMode) -> WorkspaceClient:
            nonlocal default_client
            if default_client is None:
                default_client = default_workspace_client()
            return default_client

        resolver = resolve_default

    def resolve(integration: Integration) -> WorkspaceClient:
        mode = _integration_auth_mode(integration)
        resolution_error: MasonRuntimeError
        try:
            return resolver(mode)
        except MissingUserAuthorization:
            resolution_error = (
                MissingUserAuthorization(integration.id)
                if mode == "user"
                else InvalidAppAuthorization(integration.id)
            )
        except Exception:
            resolution_error = (
                IntegrationClientResolutionError(integration.id, mode)
                if mode == "user"
                else InvalidAppAuthorization(integration.id)
            )
        raise resolution_error

    return resolve
