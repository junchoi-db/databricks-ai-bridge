"""Unit tests for versioned Mason project deployment metadata."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
import tomli

from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import load_project_metadata, write_project_metadata


def _write_metadata(project: pathlib.Path, body: str) -> pathlib.Path:
    path = project / ".mason" / "project.toml"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_write_and_load_project_metadata_records_request_auth_contract_and_extra_scopes(
    tmp_path: pathlib.Path,
) -> None:
    path = write_project_metadata(
        tmp_path,
        framework="langgraph",
        template="agent-langgraph",
        request_auth_contract_version=1,
        extra_user_api_scopes=("sql", "files.files"),
    )

    with path.open("rb") as metadata_file:
        raw = tomli.load(metadata_file)
    assert raw == {
        "schema_version": 1,
        "framework": "langgraph",
        "template": "agent-langgraph",
        "request_auth_contract_version": 1,
        "extra_user_api_scopes": ["files.files", "sql"],
    }
    loaded = load_project_metadata(tmp_path)
    assert loaded.request_auth_contract_version == 1
    assert loaded.extra_user_api_scopes == ("files.files", "sql")


def test_load_legacy_project_metadata_keeps_missing_contract_detectable(
    tmp_path: pathlib.Path,
) -> None:
    _write_metadata(
        tmp_path,
        'schema_version = 1\nframework = "openai"\ntemplate = "agent-openai"\n',
    )

    loaded = load_project_metadata(tmp_path)

    assert loaded.request_auth_contract_version is None
    assert loaded.extra_user_api_scopes == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra_user_api_scopes": "sql"},
    ],
)
def test_write_project_metadata_rejects_invalid_auth_fields(
    tmp_path: pathlib.Path, overrides: dict[str, Any]
) -> None:
    with pytest.raises(AgentCliError):
        write_project_metadata(
            tmp_path,
            framework="langgraph",
            template="agent-langgraph",
            **overrides,
        )

    assert not (tmp_path / ".mason" / "project.toml").exists()


def test_write_project_metadata_omits_explicit_unknown_request_auth_contract(
    tmp_path: pathlib.Path,
) -> None:
    path = write_project_metadata(
        tmp_path,
        framework="langgraph",
        template="custom-langgraph",
        request_auth_contract_version=None,
    )

    with path.open("rb") as metadata_file:
        raw = tomli.load(metadata_file)
    assert "request_auth_contract_version" not in raw
    assert load_project_metadata(tmp_path).request_auth_contract_version is None


@pytest.mark.parametrize("schema_version", ["true", "1.0"])
def test_load_project_metadata_rejects_non_integer_schema_version(
    tmp_path: pathlib.Path, schema_version: str
) -> None:
    _write_metadata(
        tmp_path,
        f'schema_version = {schema_version}\nframework = "langgraph"\ntemplate = "agent-langgraph"\n',
    )

    with pytest.raises(AgentCliError, match="schema"):
        load_project_metadata(tmp_path)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("request_auth_contract_version = 2\nextra_user_api_scopes = []\n", "contract"),
        ("request_auth_contract_version = true\nextra_user_api_scopes = []\n", "contract"),
        ('request_auth_contract_version = "1"\nextra_user_api_scopes = []\n', "contract"),
        ('request_auth_contract_version = 1\nextra_user_api_scopes = "sql"\n', "scope"),
        ("request_auth_contract_version = 1\nextra_user_api_scopes = [1]\n", "scope"),
        ('request_auth_contract_version = 1\nextra_user_api_scopes = [""]\n', "scope"),
        ('request_auth_contract_version = 1\nextra_user_api_scopes = ["bad scope"]\n', "scope"),
        (
            'request_auth_contract_version = 1\nextra_user_api_scopes = ["sql", "sql"]\n',
            "duplicate",
        ),
    ],
)
def test_load_project_metadata_rejects_invalid_auth_fields(
    tmp_path: pathlib.Path, fields: str, message: str
) -> None:
    _write_metadata(
        tmp_path,
        'schema_version = 1\nframework = "langgraph"\ntemplate = "agent-langgraph"\n' + fields,
    )

    with pytest.raises(AgentCliError, match=message):
        load_project_metadata(tmp_path)
