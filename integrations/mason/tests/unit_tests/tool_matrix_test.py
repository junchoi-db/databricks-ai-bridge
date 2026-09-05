"""Hermetic release-contract tests for the deployed Mason tool matrix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from types import ModuleType

import pytest
import tomli


def _load_tool_matrix() -> ModuleType:
    path = pathlib.Path(__file__).parents[1] / "e2e" / "tool_matrix.py"
    spec = importlib.util.spec_from_file_location("mason_tool_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool_matrix = _load_tool_matrix()


def _runner(tmp_path: pathlib.Path):
    return tool_matrix.Runner("df1", tmp_path / "output", tmp_path / "mason.whl")


def test_cli_and_direct_fixtures_render_explicit_auth_modes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    runner.uc_function = "main.mason_agent_tools_e2e.marker"
    runner.sandbox_table = "main.mason_agent_tools_e2e.app_sandbox_marker"
    calls: list[list[str]] = []

    def capture(argv, **_kwargs):
        calls.append(list(argv))

    monkeypatch.setattr(runner, "run", capture)
    runner._author_cli(tmp_path / "cli")

    assert [
        "tools",
        "add",
        "sandbox",
        "--scope",
        "table:main.mason_agent_tools_e2e.app_sandbox_marker",
        "--auth",
        "app",
    ] == calls[0][1:-2]
    assert [
        "tools",
        "add",
        "mcp",
        "system.ai.google_drive",
        "--auth",
        "user",
    ] == calls[1][1:-2]

    direct = tmp_path / "direct"
    (direct / "agent").mkdir(parents=True)
    runner._author_direct(direct)
    source = (direct / "agent" / "databricks_tools.py").read_text(encoding="utf-8")
    assert 'service="system.ai.google_drive"' in source
    assert 'auth="user"' in source
    assert 'Scope.table(\n                "main.mason_agent_tools_e2e.app_sandbox_marker"' in source
    assert 'auth="app"' in source
    assert "__SANDBOX_TABLE__" not in source


def test_generated_projects_pin_and_verify_the_exact_content_addressed_runtime_wheel(
    tmp_path: pathlib.Path,
) -> None:
    runner = _runner(tmp_path)
    runner.wheel.write_bytes(b"branch wheel bytes")
    runner.freshness_marker = "review-head-abc123"
    project = tmp_path / "langgraph-cli"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\ndependencies = '
        '["databricks-mason[runtime]>=0.1.1.dev0"]\n\n[tool.uv]\n',
        encoding="utf-8",
    )

    runner._pin_runtime_wheel(project)
    runner._write_python_tool(project)

    digest = tool_matrix._sha256(runner.wheel)
    relative_wheel = pathlib.Path("vendor") / digest / runner.wheel.name
    assert (project / relative_wheel).read_bytes() == b"branch wheel bytes"
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv.sources]" in pyproject
    assert f'path = "{relative_wheel.as_posix()}"' in pyproject
    assert tomli.loads(pyproject)["tool"]["uv"]["sources"]["databricks-mason"] == {
        "path": relative_wheel.as_posix()
    }
    probe = (project / "agent" / "tools" / "matrix_marker.py").read_text(encoding="utf-8")
    assert digest in probe
    assert "direct_url.json" in probe
    assert "review-head-abc123" in probe
    assert "[freshness-check review-head-abc123] hit matrix_marker" in probe
    assert runner.runtime_wheel_sources[project.name] == {
        "path": relative_wheel.as_posix(),
        "sha256": digest,
    }


def test_python_semantics_reject_a_stale_runtime_wheel_or_freshness_marker() -> None:
    current = "MASON_PYTHON_OK wheel_sha256=" + ("a" * 64) + " freshness=current-head"
    stale = {
        "output": [
            {
                "type": "ai",
                "content": "MASON_PYTHON_OK wheel_sha256="
                + ("b" * 64)
                + " freshness=previous-head",
            }
        ]
    }

    with pytest.raises(tool_matrix.MatrixError, match="current runtime wheel/freshness"):
        tool_matrix._evidence_actual("python", stale, expected_python_result=current)


def test_google_drive_semantics_are_validated_in_memory_but_evidence_is_fixed_and_redacted() -> (
    None
):
    sensitive_name = "board-acquisition-plan.pdf"
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "google_drive_list_recent",
                        "id": "call-drive",
                        "args": {"max_results": 3},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "google_drive_list_recent",
                "tool_call_id": "call-drive",
                "status": "success",
                "content": sensitive_name,
            },
            {"type": "ai", "content": "MASON_GOOGLE_DRIVE_OK count=1"},
        ]
    }

    actual = tool_matrix._evidence_actual("mcp", response)

    assert actual == tool_matrix.GOOGLE_DRIVE_REDACTED_ACTUAL
    assert sensitive_name not in actual


def test_google_drive_semantic_errors_never_echo_tool_output() -> None:
    sensitive_name = "private-roadmap-2030.docx"
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "google_drive_list_recent",
                        "id": "call-drive",
                        "args": {"max_results": 3},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "google_drive_list_recent",
                "tool_call_id": "call-drive",
                "status": "success",
                "content": sensitive_name,
            },
            {"type": "ai", "content": "I found a recent document."},
        ]
    }

    with pytest.raises(tool_matrix.MatrixError) as exc_info:
        tool_matrix._evidence_actual("mcp", response)

    assert sensitive_name not in str(exc_info.value)


def test_google_drive_call_and_summary_do_not_pass_with_an_error_tool_result() -> None:
    sensitive_error = "authorization URL with private query state"
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "google_drive_list_recent",
                        "id": "call-drive",
                        "args": {"max_results": 3},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "google_drive_list_recent",
                "tool_call_id": "call-drive",
                "status": "error",
                "content": sensitive_error,
            },
            {"type": "ai", "content": "MASON_GOOGLE_DRIVE_OK count=0"},
        ]
    }

    with pytest.raises(tool_matrix.MatrixError) as exc_info:
        tool_matrix._evidence_actual("mcp", response)

    assert sensitive_error not in str(exc_info.value)


def test_google_drive_retry_log_never_persists_raw_exception_detail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sensitive_detail = (
        "Google authorization: https://workspace.example/explore/data/mcp-services/drive"
        "?oauth_state=must-not-be-logged"
    )
    runner = _runner(tmp_path)

    def fail(*_args, **_kwargs):
        raise tool_matrix.MatrixError(sensitive_detail)

    monkeypatch.setattr(tool_matrix, "_monitored", fail)
    monkeypatch.setattr(tool_matrix.time, "sleep", lambda _seconds: None)

    with pytest.raises(tool_matrix.MatrixInvocationError):
        runner._invoke_with_retry(
            "deploy-langgraph-cli-mcp",
            "https://app.example/api/invocations",
            "privacy-safe prompt",
            {},
        )

    transcript = (tmp_path / "output" / "commands.log").read_text(encoding="utf-8")
    assert sensitive_detail not in transcript
    assert "oauth_state" not in transcript
    assert tool_matrix.MATRIX_EXECUTION_FAILED in transcript


def test_app_grants_cover_only_the_sandbox_table_and_fixed_uc_function() -> None:
    statements = tool_matrix._app_grant_statements(
        "app-sp-client-id",
        sandbox_table="main.mason_agent_tools_e2e.app_sandbox_marker",
        uc_function="main.mason_agent_tools_e2e.marker",
    )

    assert statements == (
        "GRANT USE CATALOG ON CATALOG `main` TO `app-sp-client-id`",
        "GRANT USE SCHEMA ON SCHEMA `main`.`mason_agent_tools_e2e` TO `app-sp-client-id`",
        "GRANT SELECT ON TABLE `main`.`mason_agent_tools_e2e`.`app_sandbox_marker` "
        "TO `app-sp-client-id`",
        "GRANT EXECUTE ON FUNCTION `main`.`mason_agent_tools_e2e`.`marker` TO `app-sp-client-id`",
    )
    assert all("google" not in statement.lower() for statement in statements)
    assert all("mcp service" not in statement.lower() for statement in statements)

    metadata = tool_matrix._auth_boundary_metadata("main.mason_agent_tools_e2e.app_sandbox_marker")
    assert metadata["google_drive"]["app_service_principal_granted"] is False
    assert metadata["sandbox"]["app_service_principal_granted"] is True


def test_app_sandbox_marker_without_a_successful_tool_result_does_not_pass() -> None:
    response = {
        "output": [
            {"type": "ai", "content": "MASON_APP_SANDBOX_OK"},
        ]
    }

    with pytest.raises(tool_matrix.MatrixError, match="successful Sandbox tool result"):
        tool_matrix._evidence_actual("sandbox", response)


def test_app_sandbox_accepts_a_matching_successful_tool_result() -> None:
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {"name": "sandbox", "id": "call-sandbox", "args": {"code": "spark.table"}}
                ],
            },
            {
                "type": "tool",
                "name": "sandbox",
                "tool_call_id": "call-sandbox",
                "status": "success",
                "content": "MASON_APP_SANDBOX_OK",
            },
        ]
    }

    assert "MASON_APP_SANDBOX_OK" in tool_matrix._evidence_actual("sandbox", response)


def test_deployed_sandbox_rejects_a_marker_from_any_other_code_path() -> None:
    principal_probe = (
        "from pyspark.sql import SparkSession\n"
        "spark = SparkSession.builder.getOrCreate()\n"
        'actual_principal = spark.sql("SELECT current_user() AS principal").first()["principal"]\n'
        'expected_principal = "app-sp-client-id"\n'
        "if actual_principal != expected_principal:\n"
        '    raise RuntimeError("unexpected App service principal")\n'
        'marker = spark.table("main.mason_agent_tools_e2e.marker").select("marker").limit(1).collect()[0]["marker"]\n'
        "print(marker)"
    )
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "run_code",
                        "id": "call-sandbox",
                        "args": {"code": "print('MASON_APP_SANDBOX_OK')"},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "run_code",
                "tool_call_id": "call-sandbox",
                "status": "success",
                "content": "MASON_APP_SANDBOX_OK",
            },
        ]
    }

    with pytest.raises(tool_matrix.MatrixError, match="principal probe"):
        tool_matrix._evidence_actual("sandbox", response, expected_sandbox_code=principal_probe)

    response["output"][0]["tool_calls"][0]["args"]["code"] = principal_probe
    assert "MASON_APP_SANDBOX_OK" in tool_matrix._evidence_actual(
        "sandbox", response, expected_sandbox_code=principal_probe
    )


def test_deployed_sandbox_rejects_an_unidentified_probe_with_an_unrelated_result() -> None:
    principal_probe = "print('principal-bound probe')"
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "run_code",
                        "args": {"code": principal_probe},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "run_code",
                "tool_call_id": "call-unrelated",
                "status": "success",
                "content": "MASON_APP_SANDBOX_OK",
            },
        ]
    }

    with pytest.raises(tool_matrix.MatrixError, match="principal probe"):
        tool_matrix._evidence_actual("sandbox", response, expected_sandbox_code=principal_probe)


def test_deployed_sandbox_accepts_one_wire_normalized_terminal_newline() -> None:
    principal_probe = "print('principal-bound probe')"
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "run_code",
                        "id": "call-sandbox",
                        "args": {"code": f"{principal_probe}\n"},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "run_code",
                "tool_call_id": "call-sandbox",
                "status": "success",
                "content": "MASON_APP_SANDBOX_OK",
            },
        ]
    }

    assert "MASON_APP_SANDBOX_OK" in tool_matrix._evidence_actual(
        "sandbox", response, expected_sandbox_code=principal_probe
    )


def test_deployed_sandbox_exercise_binds_success_to_the_app_principal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    runner.wheel.write_bytes(b"runtime wheel")
    runner.sandbox_table = "main.mason_agent_tools_e2e.marker"
    principal_probe = (
        "from pyspark.sql import SparkSession\n"
        "spark = SparkSession.builder.getOrCreate()\n"
        'actual_principal = spark.sql("SELECT current_user() AS principal").first()["principal"]\n'
        'expected_principal = "app-sp-client-id"\n'
        "if actual_principal != expected_principal:\n"
        '    raise RuntimeError("unexpected App service principal")\n'
        'marker = spark.table("main.mason_agent_tools_e2e.marker").select("marker").limit(1).collect()[0]["marker"]\n'
        "print(marker)"
    )
    response = {
        "output": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "run_code",
                        "id": "call-sandbox",
                        "args": {"code": principal_probe},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "run_code",
                "tool_call_id": "call-sandbox",
                "status": "success",
                "content": "MASON_APP_SANDBOX_OK",
            },
        ]
    }

    def invoke(_label, _url, prompt, _headers):
        assert principal_probe in prompt
        return response

    monkeypatch.setattr(tool_matrix, "TOOL_KINDS", ("sandbox",))
    monkeypatch.setattr(runner, "_invoke_with_retry", invoke)
    case = tool_matrix.ProjectCase("langgraph", "direct", tmp_path, "app")

    runner._exercise(
        case,
        "deploy",
        "https://app.example",
        {"Authorization": "Bearer redacted"},
        tmp_path / "deploy.log",
        app_name="app",
        expected_sandbox_principal="app-sp-client-id",
    )

    assert [(row.tool_kind, row.status) for row in runner.rows] == [("sandbox", "pass")]


def test_expected_evidence_is_exactly_sixteen_auth_annotated_cells(
    tmp_path: pathlib.Path,
) -> None:
    rows = []
    for framework, authoring, runtime, tool_kind in tool_matrix._expected_evidence_cells():
        rows.append(
            {
                "framework": framework,
                "authoring": authoring,
                "runtime": runtime,
                "tool_kind": tool_kind,
                "status": "pass",
                **tool_matrix._evidence_auth_metadata(tool_kind, runtime),
            }
        )
    wheel_bytes = b"exact runtime artifact"
    digest = hashlib.sha256(wheel_bytes).hexdigest()
    for row in rows:
        if row["tool_kind"] == "python":
            row["expected"] = tool_matrix._python_expected_result(digest, None)
    sources = {}
    for project_name in ("langgraph-cli", "langgraph-direct"):
        relative = f"vendor/{digest}/databricks_mason.whl"
        target = tmp_path / "projects" / project_name / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(wheel_bytes)
        sources[project_name] = {"path": relative, "sha256": digest}
    document = {
        "schema_version": 2,
        "wheel_sha256": digest,
        "runtime_wheel": {
            "sha256": digest,
            "project_sources": sources,
            "freshness_marker": None,
        },
        "auth_boundaries": tool_matrix._auth_boundary_metadata(
            "main.mason_agent_tools_e2e.app_sandbox_marker"
        ),
        "sandbox_table": "main.mason_agent_tools_e2e.app_sandbox_marker",
        "service_discovery": {
            "service": "system.ai.google_drive",
            "discovered": True,
        },
        "rows": rows,
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert len(rows) == 16
    dev_sandbox = next(
        row for row in rows if row["runtime"] == "dev" and row["tool_kind"] == "sandbox"
    )
    deploy_sandbox = next(
        row for row in rows if row["runtime"] == "deploy" and row["tool_kind"] == "sandbox"
    )
    dev_uc = next(
        row for row in rows if row["runtime"] == "dev" and row["tool_kind"] == "uc_function"
    )
    assert dev_sandbox["execution_principal"] == "selected developer profile"
    assert dev_sandbox["app_service_principal_granted"] is False
    assert deploy_sandbox["execution_principal"] == "dedicated Databricks App service principal"
    assert deploy_sandbox["app_service_principal_granted"] is True
    assert dev_uc["execution_principal"] == "selected developer profile"
    assert dev_uc["app_service_principal_granted"] is False
    assert tool_matrix.verify_evidence(path) == 0

    mcp_row = next(row for row in rows if row["tool_kind"] == "mcp")
    mcp_row["app_service_principal_granted"] = True
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert tool_matrix.verify_evidence(path) == 1

    mcp_row["app_service_principal_granted"] = False
    document["auth_boundaries"]["google_drive"]["app_service_principal_granted"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    assert tool_matrix.verify_evidence(path) == 1

    document["auth_boundaries"]["google_drive"]["app_service_principal_granted"] = False
    document.pop("runtime_wheel")
    path.write_text(json.dumps(document), encoding="utf-8")
    assert tool_matrix.verify_evidence(path) == 1


@pytest.mark.parametrize(
    ("status", "body", "expected_code"),
    [
        (401, {"error_code": "UNAUTHENTICATED"}, "APPS_INGRESS_AUTHORIZATION_FAILED"),
        (403, {"error_code": "PERMISSION_DENIED"}, "APPS_INGRESS_AUTHORIZATION_FAILED"),
        (
            401,
            {"error": {"code": "MCP_USER_AUTHORIZATION_MISSING", "message": "secret"}},
            "MCP_USER_AUTHORIZATION_MISSING",
        ),
        (
            401,
            {"error": {"code": "MCP_USER_AUTHORIZATION_INVALID", "message": "secret"}},
            "MCP_USER_AUTHORIZATION_INVALID",
        ),
        (
            403,
            {"error": {"code": "MCP_PERMISSION_DENIED", "message": "secret"}},
            "MCP_PERMISSION_DENIED",
        ),
        (
            401,
            {
                "error": {
                    "code": "MCP_AUTHORIZATION_REQUIRED",
                    "message": "secret provider URL",
                }
            },
            "MCP_AUTHORIZATION_REQUIRED",
        ),
    ],
)
def test_http_failures_keep_apps_ingress_and_mason_boundaries_distinct(
    status: int, body: dict[str, object], expected_code: str
) -> None:
    error = tool_matrix._safe_http_error(status, json.dumps(body))

    assert error.code == expected_code
    assert "secret" not in str(error)
