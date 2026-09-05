"""Unit tests for ``mason tools`` behavior not covered by code-first attachment tests."""

from __future__ import annotations

import json
import pathlib

from click.testing import CliRunner

from databricks_mason.integration_codegen import IntegrationRegistry, render_registry
from databricks_mason.integrations import MCPService
from databricks_mason.project_config import write_project_metadata
from databricks_mason.tools import tools


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(tmp_path: pathlib.Path, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    project.mkdir(parents=True)
    write_project_metadata(project, framework=framework, template=f"agent-{framework}")
    IntegrationRegistry.empty(project).write()
    return project


def test_generic_mcp_rejects_sandbox_scope(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--scope" in result.output
    assert IntegrationRegistry.load(project).integrations == []


def test_tools_list_emits_code_registry_records_as_json(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output
    sandbox = runner.invoke(
        tools,
        [
            "add",
            "sandbox",
            "--scope",
            "main.data.files",
            "--auth",
            "app",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )
    function = runner.invoke(
        tools,
        ["add", "uc-function", "main.tools.lookup", "--source", str(project)],
        obj=_Ctx(),
    )
    assert sandbox.exit_code == 0, sandbox.output
    assert function.exit_code == 0, function.output

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tools"] == [
        {
            "id": "web_search",
            "kind": "mcp",
            "source": "system.ai.web_search",
            "auth": "user",
        },
        {
            "id": "sandbox",
            "kind": "sandbox",
            "source": "volume:main.data.files",
            "auth": "app",
        },
        {
            "id": "lookup",
            "kind": "uc_function",
            "source": "main.tools.lookup",
            "auth": "app/default",
        },
    ]

    text_result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(),
    )
    assert text_result.exit_code == 0, text_result.output
    assert "AUTH" in text_result.output
    assert "user" in text_result.output
    assert "app" in text_result.output
    assert "app/default" in text_result.output


def test_tools_list_marks_legacy_missing_auth_as_unspecified(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    registry_path = project / "agent" / "databricks_tools.py"
    source = render_registry([MCPService(id="web", service="system.ai.web_search")])
    registry_path.write_text(
        source.replace('        auth="user",\n', ""),
        encoding="utf-8",
    )
    runner = CliRunner()

    json_result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )
    text_result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(),
    )

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["tools"][0]["auth"] == "unspecified"
    assert text_result.exit_code == 0, text_result.output
    assert "unspecified" in text_result.output
