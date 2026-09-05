"""Persistent Mason project metadata and legacy framework detection."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import tomli

from databricks_mason.errors import AgentCliError

_CONFIG_PATH = pathlib.Path(".mason/project.toml")
_SCHEMA_VERSION = 1
REQUEST_AUTH_CONTRACT_VERSION = 1
_SUPPORTED_FRAMEWORKS = {"langgraph", "openai"}


@dataclass(frozen=True)
class ProjectMetadata:
    """The template identity persisted by ``mason init``."""

    framework: str
    template: str | None
    request_auth_contract_version: int | None = None
    extra_user_api_scopes: tuple[str, ...] = ()


def _validate_request_auth_contract_version(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value != REQUEST_AUTH_CONTRACT_VERSION:
        raise AgentCliError(
            f"Unsupported Mason request-auth contract version {value!r}.",
            hint=f"Expected request_auth_contract_version = {REQUEST_AUTH_CONTRACT_VERSION}.",
        )
    return value


def _validate_extra_user_api_scopes(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentCliError("Mason project extra user API scopes must be a TOML array of strings.")
    if any(
        not isinstance(scope, str)
        or not scope
        or scope != scope.strip()
        or any(character.isspace() or not character.isprintable() for character in scope)
        for scope in value
    ):
        raise AgentCliError(
            "Each Mason project extra user API scope must be a non-empty string without whitespace."
        )
    if len(value) != len(set(value)):
        raise AgentCliError("Mason project extra user API scopes contain a duplicate scope.")
    return tuple(sorted(value))


def write_project_metadata(
    project: pathlib.Path,
    *,
    framework: str,
    template: str,
    request_auth_contract_version: int | None = REQUEST_AUTH_CONTRACT_VERSION,
    extra_user_api_scopes: tuple[str, ...] = (),
) -> pathlib.Path:
    """Write the metadata consumed by template-aware Mason commands."""
    contract_version = _validate_request_auth_contract_version(request_auth_contract_version)
    if not isinstance(extra_user_api_scopes, (list, tuple)):
        raise AgentCliError("Mason project extra user API scopes must be a sequence of strings.")
    scopes = _validate_extra_user_api_scopes(list(extra_user_api_scopes))
    contract_line = (
        f"request_auth_contract_version = {contract_version}\n"
        if contract_version is not None
        else ""
    )
    target = project / _CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"schema_version = {_SCHEMA_VERSION}\n"
        f"framework = {json.dumps(framework)}\n"
        f"template = {json.dumps(template)}\n"
        f"{contract_line}"
        f"extra_user_api_scopes = {json.dumps(list(scopes))}\n",
        encoding="utf-8",
    )
    return target


def _read_toml(path: pathlib.Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as input_file:
            value = tomli.load(input_file)
    except (OSError, tomli.TOMLDecodeError) as exc:
        raise AgentCliError(f"Could not read {description} at {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise AgentCliError(f"{description.capitalize()} at {path} must be a TOML table.")
    return value


def _validate_framework(framework: object) -> str:
    if not isinstance(framework, str) or framework not in _SUPPORTED_FRAMEWORKS:
        rendered = repr(framework) if isinstance(framework, str) else "missing"
        raise AgentCliError(
            f"Unsupported Mason framework {rendered}.",
            hint=f"Supported frameworks: {', '.join(sorted(_SUPPORTED_FRAMEWORKS))}.",
        )
    return framework


def _load_persisted_metadata(project: pathlib.Path) -> ProjectMetadata | None:
    path = project / _CONFIG_PATH
    if not path.is_file():
        return None
    data = _read_toml(path, "Mason project config")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
        raise AgentCliError(
            f"Unsupported Mason project config schema in {path}.",
            hint=f"Expected schema_version = {_SCHEMA_VERSION}.",
        )
    framework = _validate_framework(data.get("framework"))
    template = data.get("template")
    if not isinstance(template, str) or not template:
        raise AgentCliError(f"Mason project config at {path} must declare a template.")
    contract_version = _validate_request_auth_contract_version(
        data.get("request_auth_contract_version")
    )
    scopes = _validate_extra_user_api_scopes(data.get("extra_user_api_scopes", []))
    return ProjectMetadata(
        framework=framework,
        template=template,
        request_auth_contract_version=contract_version,
        extra_user_api_scopes=scopes,
    )


def _dependency_name(requirement: str) -> str:
    """Return a normalized distribution name from a PEP 508 requirement."""
    name = re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _infer_legacy_framework(project: pathlib.Path) -> ProjectMetadata:
    pyproject = project / "pyproject.toml"
    if not pyproject.is_file():
        raise AgentCliError(
            f"Could not determine the Mason framework for {project}.",
            hint="Run `mason init` to create project metadata or pass `--framework`.",
        )
    data = _read_toml(pyproject, "pyproject")
    project_table = data.get("project")
    dependencies = project_table.get("dependencies", []) if isinstance(project_table, dict) else []
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        dependencies = []
    packages = {_dependency_name(item) for item in dependencies}
    candidates = {
        framework
        for package, framework in (
            ("databricks-openai", "openai"),
            ("databricks-langchain", "langgraph"),
        )
        if package in packages
    }
    if len(candidates) != 1:
        detail = (
            "both framework dependencies are present"
            if candidates
            else "no framework dependency was found"
        )
        raise AgentCliError(
            f"Could not determine the Mason framework for {project}: {detail}.",
            hint="Pass `--framework openai` or `--framework langgraph`.",
        )
    return ProjectMetadata(framework=candidates.pop(), template=None)


def load_project_metadata(
    project: pathlib.Path,
    *,
    framework_override: str | None = None,
) -> ProjectMetadata:
    """Load init metadata, with dependency inference for projects created before it existed."""
    persisted = _load_persisted_metadata(project)
    if framework_override is not None:
        override = _validate_framework(framework_override)
        if persisted is not None and persisted.framework != override:
            raise AgentCliError(
                f"Framework override '{override}' conflicts with {persisted.framework!r} in "
                f"{project / _CONFIG_PATH}."
            )
        return persisted or ProjectMetadata(framework=override, template=None)
    return persisted or _infer_legacy_framework(project)
