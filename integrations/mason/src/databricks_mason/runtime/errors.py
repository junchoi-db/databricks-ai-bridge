"""Safe structured runtime errors and managed MCP authorization normalization."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from databricks.sdk.errors import PermissionDenied, Unauthenticated

if TYPE_CHECKING:
    from databricks_mason.integrations import AuthMode

MCP_USER_AUTHORIZATION_MISSING = "MCP_USER_AUTHORIZATION_MISSING"
MCP_USER_AUTHORIZATION_INVALID = "MCP_USER_AUTHORIZATION_INVALID"
MCP_APP_AUTHORIZATION_INVALID = "MCP_APP_AUTHORIZATION_INVALID"
MCP_PERMISSION_DENIED = "MCP_PERMISSION_DENIED"
MCP_USER_AUTH_BACKGROUND_UNSUPPORTED = "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED"
MCP_USER_AUTH_HITL_UNSUPPORTED = "MCP_USER_AUTH_HITL_UNSUPPORTED"
MCP_AUTHORIZATION_REQUIRED = "MCP_AUTHORIZATION_REQUIRED"

_AUTHORIZATION_PATH_PREFIX = "/explore/data/mcp-services/"
# Catalog Explorer authorization links carry OAuth state in their query. 2,048 characters
# is a conservative browser-compatible ceiling that bounds what Mason returns to a caller.
_MAX_AUTHORIZATION_URL_LENGTH = 2048
_ABSOLUTE_URI_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+")
_AUTHORIZATION_WORDING_PATTERN = re.compile(
    r"\b(?:authori[sz](?:ation|e|ed|ing)?|consent(?:ed|ing)?|oauth|log(?:[ -]?in)|sign(?:[ -]?in))\b",
    re.IGNORECASE,
)
_ELICITATION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"

_INVALID_AUTH_ERROR_CODES = frozenset(
    {
        "INVALID_AUTHENTICATION",
        "INVALID_TOKEN",
        "TOKEN_EXPIRED",
        "UNAUTHENTICATED",
        "UNAUTHORIZED",
    }
)
_PERMISSION_ERROR_CODES = frozenset({"FORBIDDEN", "PERMISSION_DENIED"})


class MasonRuntimeError(RuntimeError):
    """A caller-safe runtime failure with a stable HTTP/JSON contract."""

    code = "MASON_RUNTIME_ERROR"
    status = 500

    def __init__(
        self,
        message: str,
        *,
        integration_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.integration_id = integration_id
        self.data = copy.deepcopy(data) if data is not None else None

    def to_error_envelope(self) -> dict[str, dict[str, Any]]:
        """Return the stable JSON body consumed by generated HTTP runtimes."""

        payload: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.integration_id is not None:
            payload["integration_id"] = self.integration_id
        if self.data is not None:
            payload["data"] = copy.deepcopy(self.data)
        return {"error": payload}

    def to_observability_envelope(self) -> dict[str, dict[str, Any]]:
        """Return trace-safe error metadata without provider challenge payloads."""

        payload: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.integration_id is not None:
            payload["integration_id"] = self.integration_id
        return {"error": payload}

    def __repr__(self) -> str:
        # Keep structured data (which may include a validated OAuth URL query) out of
        # exception/log representations. Callers receive it only through the envelope.
        return f"{type(self).__name__}({str(self)!r})"


class RequestAuthorizationError(MasonRuntimeError):
    """A request authorization failure whose message is safe to show to a caller."""

    code = MCP_USER_AUTHORIZATION_INVALID
    status = 401


class MissingUserAuthorization(RequestAuthorizationError):
    """A deployed user-auth integration has no Apps-forwarded credential."""

    code = MCP_USER_AUTHORIZATION_MISSING

    def __init__(self, integration_id: str | None = None) -> None:
        if integration_id is None:
            message = (
                "User authorization is required, but no Databricks Apps-forwarded "
                "credential was provided."
            )
        else:
            message = (
                f"Integration {integration_id!r} requires user authorization, but no "
                "Databricks Apps-forwarded credential was provided."
            )
        super().__init__(message, integration_id=integration_id)


class InvalidUserAuthorization(RequestAuthorizationError):
    """The request-user credential was expired or rejected."""

    def __init__(self, integration_id: str) -> None:
        super().__init__(
            f"Request-user authorization for integration {integration_id!r} is invalid or expired.",
            integration_id=integration_id,
        )


class InvalidAppAuthorization(MasonRuntimeError):
    """The configured application credential could not be resolved or was rejected."""

    code = MCP_APP_AUTHORIZATION_INVALID
    status = 500

    def __init__(self, integration_id: str) -> None:
        super().__init__(
            f"App authorization for integration {integration_id!r} is invalid or unavailable.",
            integration_id=integration_id,
        )


class IntegrationClientResolutionError(RequestAuthorizationError):
    """An integration's configured credential could not be resolved safely."""

    def __init__(self, integration_id: str, mode: str) -> None:
        super().__init__(
            f"Could not resolve {mode} authorization for integration {integration_id!r}.",
            integration_id=integration_id,
        )


class MCPPermissionDenied(MasonRuntimeError):
    """The request principal lacks permission for one managed MCP integration."""

    code = MCP_PERMISSION_DENIED
    status = 403

    def __init__(self, integration_id: str) -> None:
        super().__init__(
            f"The request principal does not have permission to use integration {integration_id!r}.",
            integration_id=integration_id,
        )


class UserAuthBackgroundUnsupported(MasonRuntimeError):
    """A forwarded user credential cannot outlive its foreground request."""

    code = MCP_USER_AUTH_BACKGROUND_UNSUPPORTED
    status = 400

    def __init__(self, integration_id: str | None = None) -> None:
        super().__init__(
            "Background execution is not supported for user-authenticated MCP integrations.",
            integration_id=integration_id,
        )


class UserAuthHITLUnsupported(MasonRuntimeError):
    """The runtime cannot safely persist an OBO human-in-the-loop pause."""

    code = MCP_USER_AUTH_HITL_UNSUPPORTED
    status = 400

    def __init__(self, integration_id: str | None = None) -> None:
        super().__init__(
            "Human-in-the-loop pauses are not supported for user-authenticated MCP integrations.",
            integration_id=integration_id,
        )


class MCPAuthorizationRequired(MasonRuntimeError):
    """A managed MCP service requires an additional provider authorization step."""

    code = MCP_AUTHORIZATION_REQUIRED
    status = 401

    def __init__(
        self,
        integration_id: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Integration {integration_id!r} requires additional provider authorization.",
            integration_id=integration_id,
            data=data,
        )


def _trusted_authorization_url(url: str, workspace_url: str) -> str | None:
    if (
        not isinstance(url, str)
        or not isinstance(workspace_url, str)
        or not url
        or len(url) > _MAX_AUTHORIZATION_URL_LENGTH
        or url != url.strip()
        or any(character.isspace() or ord(character) < 0x20 for character in url)
    ):
        return None

    try:
        candidate = urlsplit(url)
        workspace = urlsplit(workspace_url)
        candidate_port = candidate.port
        workspace_port = workspace.port
    except ValueError:
        return None

    if (
        candidate.scheme.lower() != "https"
        or not candidate.netloc
        or candidate.username is not None
        or candidate.password is not None
        or candidate.hostname is None
        or workspace.hostname is None
        or candidate.hostname.casefold() != workspace.hostname.casefold()
        or candidate_port != workspace_port
        or not candidate.path.startswith(_AUTHORIZATION_PATH_PREFIX)
    ):
        return None
    return url


def _provider_authorization_error(
    source: BaseException,
    *,
    integration_id: str,
    workspace_url: str,
) -> MCPAuthorizationRequired | None:
    # Keep MCP imports lazy so the neutral request-auth module remains importable from
    # the light CLI installation, which does not install a framework runtime extra.
    from mcp import McpError
    from mcp.types import ElicitationRequiredErrorData

    if not isinstance(source, McpError) or source.error.code != -32042:
        return None

    raw_data = source.error.data
    if isinstance(raw_data, dict):
        raw_elicitations = raw_data.get("elicitations")
    else:
        raw_elicitations = getattr(raw_data, "elicitations", None)
    if not isinstance(raw_elicitations, list):
        return MCPAuthorizationRequired(integration_id)

    elicitations: list[dict[str, str]] = []
    for raw_elicitation in raw_elicitations:
        try:
            parsed = ElicitationRequiredErrorData.model_validate(
                {"elicitations": [raw_elicitation]}
            ).elicitations[0]
        except (IndexError, TypeError, ValueError):
            continue
        elicitation_id = parsed.elicitationId
        url = _trusted_authorization_url(parsed.url, workspace_url)
        if url is None or _ELICITATION_ID_PATTERN.fullmatch(elicitation_id) is None:
            continue
        elicitations.append(
            {
                "mode": "url",
                "message": "Additional authorization is required.",
                "url": url,
                "elicitationId": elicitation_id,
            }
        )
    if not elicitations:
        return MCPAuthorizationRequired(integration_id)
    return MCPAuthorizationRequired(
        integration_id,
        data={"elicitations": elicitations},
    )


def _exception_nodes(source: BaseException) -> Iterator[BaseException]:
    """Yield typed exception signals without inspecting messages, bodies, or headers."""

    pending = [source]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(item for item in nested if isinstance(item, BaseException))
        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _status_code(source: BaseException) -> int | None:
    direct = getattr(source, "status_code", None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    response = getattr(source, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _authorization_signal(source: BaseException) -> str | None:
    if isinstance(source, Unauthenticated):
        return "invalid"
    if isinstance(source, PermissionDenied):
        return "permission"
    if (status := _status_code(source)) == 401:
        return "invalid"
    if status == 403:
        return "permission"
    error_code = getattr(source, "error_code", None)
    if isinstance(error_code, str):
        normalized = error_code.upper()
        if normalized in _INVALID_AUTH_ERROR_CODES:
            return "invalid"
        if normalized in _PERMISSION_ERROR_CODES:
            return "permission"
    return None


def classify_managed_mcp_exception(
    source: BaseException,
    *,
    integration_id: str,
    workspace_url: str,
    auth_mode: AuthMode,
) -> MasonRuntimeError | None:
    """Classify trusted managed-MCP failures using typed/status signals only."""

    nodes = tuple(_exception_nodes(source))
    for node in nodes:
        if challenge := _provider_authorization_error(
            node,
            integration_id=integration_id,
            workspace_url=workspace_url,
        ):
            return challenge

    signals = {signal for node in nodes if (signal := _authorization_signal(node)) is not None}
    if signals == {"invalid"}:
        if auth_mode == "user":
            return InvalidUserAuthorization(integration_id)
        if auth_mode == "app":
            return InvalidAppAuthorization(integration_id)
        raise ValueError(f"Unsupported integration auth mode {auth_mode!r}.")
    if signals == {"permission"}:
        return MCPPermissionDenied(integration_id)
    return None


def classify_managed_mcp_result(
    result: Any,
    *,
    integration_id: str,
    workspace_url: str,
    auth_mode: AuthMode,
) -> MCPAuthorizationRequired | None:
    """Recognize the bounded flattened Catalog Explorer compatibility result."""

    from mcp.types import CallToolResult, TextContent

    if not isinstance(result, CallToolResult) or not result.isError:
        return None

    del auth_mode
    urls: list[str] = []
    recognized = False
    for content in result.content:
        if not isinstance(content, TextContent):
            continue
        if (
            _AUTHORIZATION_WORDING_PATTERN.search(content.text) is None
            or _AUTHORIZATION_PATH_PREFIX not in content.text
        ):
            continue
        recognized = True
        for match in _ABSOLUTE_URI_PATTERN.finditer(content.text):
            candidate = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
            if _AUTHORIZATION_PATH_PREFIX not in candidate:
                continue
            if (
                url := _trusted_authorization_url(candidate, workspace_url)
            ) is not None and url not in urls:
                urls.append(url)
    if not recognized:
        return None

    if not urls:
        return MCPAuthorizationRequired(integration_id)

    return MCPAuthorizationRequired(
        integration_id,
        data={
            "elicitations": [
                {
                    "mode": "url",
                    "message": "Additional authorization is required.",
                    "url": url,
                    "elicitationId": f"mason-authorization-{index}",
                }
                for index, url in enumerate(urls, start=1)
            ]
        },
    )


__all__ = [
    "MCP_USER_AUTHORIZATION_MISSING",
    "MCP_USER_AUTHORIZATION_INVALID",
    "MCP_APP_AUTHORIZATION_INVALID",
    "MCP_PERMISSION_DENIED",
    "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
    "MCP_USER_AUTH_HITL_UNSUPPORTED",
    "MCP_AUTHORIZATION_REQUIRED",
    "MasonRuntimeError",
    "RequestAuthorizationError",
    "MissingUserAuthorization",
    "InvalidUserAuthorization",
    "InvalidAppAuthorization",
    "IntegrationClientResolutionError",
    "MCPPermissionDenied",
    "UserAuthBackgroundUnsupported",
    "UserAuthHITLUnsupported",
    "MCPAuthorizationRequired",
    "classify_managed_mcp_exception",
    "classify_managed_mcp_result",
]
