"""Tests for framework-neutral structured runtime authorization errors."""

from __future__ import annotations

import importlib

import pytest
from databricks.sdk.errors import PermissionDenied, Unauthenticated
from mcp import McpError
from mcp.types import CallToolResult, ElicitationRequiredErrorData, ErrorData, TextContent

WORKSPACE_URL = "https://workspace.example.com"
AUTHORIZATION_URL = (
    f"{WORKSPACE_URL}/explore/data/mcp-services/system/ai/google_drive?oauth_state=sensitive-query"
)


def _errors():
    return importlib.import_module("databricks_mason.runtime.errors")


def _typed_authorization_error(
    url: str = AUTHORIZATION_URL,
    *,
    message: str = "Connect Google Drive",
    elicitation_id: str = "google-drive-oauth",
) -> McpError:
    return McpError(
        ErrorData(
            code=-32042,
            message=f"Unsafe upstream message containing {url}",
            data={
                "elicitations": [
                    {
                        "mode": "url",
                        "message": message,
                        "url": url,
                        "elicitationId": elicitation_id,
                        "access_token": "must-not-be-retained",
                        "_meta": {"authorization": "Bearer must-not-be-retained"},
                        "task": {"taskId": "must-not-be-retained"},
                    }
                ],
                "authorization": "Bearer must-not-be-retained",
            },
        )
    )


def _flattened_authorization_result(url: str) -> CallToolResult:
    return CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text",
                text=f"Google Drive authorization is required. Open {url}",
            )
        ],
    )


def test_runtime_authorization_errors_have_stable_codes_statuses_and_envelopes() -> None:
    errors = _errors()
    cases = [
        (
            errors.MissingUserAuthorization("drive"),
            "MCP_USER_AUTHORIZATION_MISSING",
            401,
        ),
        (
            errors.RequestAuthorizationError("Request authorization is invalid."),
            "MCP_USER_AUTHORIZATION_INVALID",
            401,
        ),
        (
            errors.InvalidAppAuthorization("drive"),
            "MCP_APP_AUTHORIZATION_INVALID",
            500,
        ),
        (
            errors.MCPPermissionDenied("drive"),
            "MCP_PERMISSION_DENIED",
            403,
        ),
        (
            errors.UserAuthBackgroundUnsupported(),
            "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            400,
        ),
        (
            errors.UserAuthHITLUnsupported(),
            "MCP_USER_AUTH_HITL_UNSUPPORTED",
            400,
        ),
        (
            errors.MCPAuthorizationRequired("drive"),
            "MCP_AUTHORIZATION_REQUIRED",
            401,
        ),
    ]

    for error, code, status in cases:
        assert error.code == code
        assert error.status == status
        assert error.message == str(error)
        assert error.to_error_envelope() == {
            "error": {
                "code": code,
                "message": str(error),
                **(
                    {"integration_id": error.integration_id}
                    if error.integration_id is not None
                    else {}
                ),
            }
        }

    assert isinstance(cases[0][0], errors.RequestAuthorizationError)


def test_typed_provider_challenge_preserves_only_safe_elicitation_fields() -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_exception(
        _typed_authorization_error(
            message=(f"Bearer must-not-be-retained; callback={AUTHORIZATION_URL}")
        ),
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.integration_id == "drive"
    assert error.data == {
        "elicitations": [
            {
                "mode": "url",
                "message": "Additional authorization is required.",
                "url": AUTHORIZATION_URL,
                "elicitationId": "google-drive-oauth",
            }
        ]
    }
    assert error.to_error_envelope()["error"]["data"] == error.data
    assert error.to_observability_envelope() == {
        "error": {
            "code": "MCP_AUTHORIZATION_REQUIRED",
            "message": "Integration 'drive' requires additional provider authorization.",
            "integration_id": "drive",
        }
    }
    ElicitationRequiredErrorData.model_validate(error.data)
    assert "must-not-be-retained" not in repr(error.data)
    assert "sensitive-query" not in repr(error.to_observability_envelope())
    for rendered in (str(error), repr(error)):
        assert "sensitive-query" not in rendered
        assert "must-not-be-retained" not in rendered


def test_malformed_typed_provider_data_is_not_retained() -> None:
    errors = _errors()
    source = McpError(
        ErrorData(
            code=-32042,
            message="Authorization required",
            data={
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Missing the required elicitation id",
                        "url": AUTHORIZATION_URL,
                        "secret": "must-not-be-retained",
                    }
                ]
            },
        )
    )

    error = errors.classify_managed_mcp_exception(
        source,
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert "must-not-be-retained" not in repr(error)


@pytest.mark.parametrize(
    "elicitation_id",
    ["oauth?id=must-not-be-retained", "x" * 129],
)
def test_typed_provider_challenge_rejects_unsafe_elicitation_ids(
    elicitation_id: str,
) -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_exception(
        _typed_authorization_error(elicitation_id=elicitation_id),
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert elicitation_id not in repr(error)


@pytest.mark.parametrize(
    "url",
    [
        "http://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://user@workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://other.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://workspace.example.com:443/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://workspace.example.com/explore/data/other/system/ai/google_drive?state=x",
        "https://[malformed",
        (
            "https://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state="
            + "x" * 2048
        ),
    ],
)
def test_typed_provider_challenge_does_not_return_untrusted_urls(url: str) -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_exception(
        _typed_authorization_error(url),
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert url not in repr(error)


def test_flattened_google_drive_result_becomes_a_safe_provider_challenge() -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_result(
        _flattened_authorization_result(AUTHORIZATION_URL),
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data == {
        "elicitations": [
            {
                "mode": "url",
                "message": "Additional authorization is required.",
                "url": AUTHORIZATION_URL,
                "elicitationId": "mason-authorization-1",
            }
        ]
    }
    assert "sensitive-query" not in str(error)
    assert "sensitive-query" not in repr(error)
    ElicitationRequiredErrorData.model_validate(error.data)


@pytest.mark.parametrize(
    "url",
    [
        "http://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "ftp://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://user@workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://other.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://workspace.example.com:443/explore/data/mcp-services/system/ai/google_drive?state=x",
        "https://[malformed/explore/data/mcp-services/system/ai/google_drive?state=x",
        "workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state=x",
        (
            "https://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?state="
            + "x" * 2048
        ),
    ],
)
def test_flattened_authorization_challenge_rejects_untrusted_url_data(url: str) -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_result(
        _flattened_authorization_result(url),
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert url not in str(error)
    assert url not in repr(error)


@pytest.mark.parametrize(
    "candidate",
    [
        "https://docs.example.com/troubleshooting/authentication",
        "https://workspace.example.com/explore/data/other/system/ai/google_drive?state=x",
        "https://[malformed",
    ],
)
def test_flattened_auth_wording_with_non_catalog_url_remains_an_ordinary_error(
    candidate: str,
) -> None:
    errors = _errors()
    result = CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text",
                text=f"Authorization header rejected; see {candidate}",
            )
        ],
    )

    assert (
        errors.classify_managed_mcp_result(
            result,
            integration_id="drive",
            workspace_url=WORKSPACE_URL,
            auth_mode="user",
        )
        is None
    )


def test_flattened_catalog_explorer_consent_challenge_is_safely_recognized() -> None:
    errors = _errors()
    candidate = (
        "ftp://workspace.example.com/explore/data/mcp-services/system/ai/google_drive?secret=x"
    )
    result = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=f"Consent is required at {candidate}")],
    )

    error = errors.classify_managed_mcp_result(
        result,
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPAuthorizationRequired)
    assert error.data is None
    assert candidate not in str(error)
    assert candidate not in repr(error)


def test_unrelated_flattened_tool_error_is_not_an_authorization_challenge() -> None:
    errors = _errors()
    result = CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text",
                text=(
                    "Query failed while reading the input URL "
                    "https://other.example.com/data?debug=ordinary-error"
                ),
            )
        ],
    )

    assert (
        errors.classify_managed_mcp_result(
            result,
            integration_id="drive",
            workspace_url=WORKSPACE_URL,
            auth_mode="user",
        )
        is None
    )


@pytest.mark.parametrize(
    ("source", "auth_mode", "error_name", "code", "status"),
    [
        (
            Unauthenticated("unsafe upstream request body"),
            "user",
            "InvalidUserAuthorization",
            "MCP_USER_AUTHORIZATION_INVALID",
            401,
        ),
        (
            Unauthenticated("unsafe upstream request body"),
            "app",
            "InvalidAppAuthorization",
            "MCP_APP_AUTHORIZATION_INVALID",
            500,
        ),
        (
            PermissionDenied("unsafe upstream request body"),
            "app",
            "MCPPermissionDenied",
            "MCP_PERMISSION_DENIED",
            403,
        ),
    ],
)
def test_typed_auth_failures_are_classified_without_upstream_text(
    source: Exception,
    auth_mode: str,
    error_name: str,
    code: str,
    status: int,
) -> None:
    errors = _errors()

    error = errors.classify_managed_mcp_exception(
        source,
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode=auth_mode,
    )

    assert isinstance(error, getattr(errors, error_name))
    assert error.code == code
    assert error.status == status
    assert "unsafe upstream request body" not in str(error)
    assert "unsafe upstream request body" not in repr(error)


def test_status_classification_does_not_read_response_body_or_headers() -> None:
    errors = _errors()

    class _Response:
        status_code = 403

        @property
        def headers(self):
            raise AssertionError("headers must not be inspected")

        @property
        def text(self):
            raise AssertionError("body must not be inspected")

    source = RuntimeError("unsafe body")
    source.response = _Response()  # type: ignore[attr-defined]

    error = errors.classify_managed_mcp_exception(
        source,
        integration_id="drive",
        workspace_url=WORKSPACE_URL,
        auth_mode="user",
    )

    assert isinstance(error, errors.MCPPermissionDenied)


def test_error_messages_are_not_used_as_authorization_signals() -> None:
    errors = _errors()

    assert (
        errors.classify_managed_mcp_exception(
            RuntimeError("401 permission denied, click https://attacker.example/secret"),
            integration_id="drive",
            workspace_url=WORKSPACE_URL,
            auth_mode="user",
        )
        is None
    )
