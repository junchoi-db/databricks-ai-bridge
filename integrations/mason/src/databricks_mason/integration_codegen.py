"""Strict, non-executing reader and writer for ``DATABRICKS_TOOLS`` source."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import tempfile
from collections.abc import Iterable
from typing import cast

from databricks_mason.errors import AgentCliError
from databricks_mason.integrations import (
    AuthMode,
    Integration,
    MCPService,
    Sandbox,
    Scope,
    UCFunction,
)

_DEFAULT_RELATIVE_PATH = pathlib.Path("agent/databricks_tools.py")
_REGISTRY_PATHS = {
    "langgraph": _DEFAULT_RELATIVE_PATH,
    "openai": _DEFAULT_RELATIVE_PATH,
}
_ALIAS = "mason_integrations"


def registry_relative_path(framework: str) -> pathlib.Path:
    try:
        return _REGISTRY_PATHS[framework]
    except KeyError as exc:
        raise AgentCliError(f"Unsupported Mason framework {framework!r}.") from exc


def _string(node: ast.AST, description: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise AgentCliError(f"The CLI-owned integration registry requires a literal {description}.")
    return node.value


def _keywords(node: ast.Call) -> dict[str, ast.AST]:
    if node.args or any(keyword.arg is None for keyword in node.keywords):
        raise AgentCliError("The CLI-owned integration registry is not in canonical form.")
    names = [keyword.arg for keyword in node.keywords]
    if len(names) != len(set(names)):
        raise AgentCliError("The CLI-owned integration registry has duplicate keywords.")
    return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}


def _constructor(node: ast.AST, name: str) -> ast.Call:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == _ALIAS
        and node.func.attr == name
    ):
        raise AgentCliError("The CLI-owned integration registry is not in canonical form.")
    return node


def _scope(node: ast.AST) -> Scope:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == _ALIAS
        and node.func.value.attr == "Scope"
        and node.func.attr in {"table", "volume", "workspace"}
        and len(node.args) == 1
        and len(node.keywords) <= 1
        and all(keyword.arg == "permission" for keyword in node.keywords)
    ):
        raise AgentCliError("The CLI-owned integration registry has a non-canonical scope.")
    value = _string(node.args[0], "scope value")
    permission = "read_only"
    if node.keywords:
        permission = _string(node.keywords[0].value, "scope permission")
    return Scope(node.func.attr, value, permission)  # type: ignore[arg-type]


def _auth(values: dict[str, ast.AST]) -> tuple[AuthMode, bool]:
    if "auth" not in values:
        return "user", True
    value = _string(values["auth"], "integration auth mode")
    if value not in {"user", "app"}:
        raise AgentCliError(f"Unsupported integration auth mode {value!r}.")
    return cast(AuthMode, value), False


def _integration(node: ast.AST) -> tuple[Integration, bool]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        raise AgentCliError("The CLI-owned integration registry is not in canonical form.")
    name = node.func.attr
    if name == "Sandbox":
        call = _constructor(node, name)
        values = _keywords(call)
        if set(values) not in ({"id", "scopes"}, {"id", "scopes", "auth"}) or not isinstance(
            values["scopes"], ast.Tuple
        ):
            raise AgentCliError("The CLI-owned Sandbox integration is not in canonical form.")
        auth, legacy_auth = _auth(values)
        return (
            Sandbox(
                id=_string(values["id"], "integration id"),
                scopes=tuple(_scope(item) for item in values["scopes"].elts),
                auth=auth,
            ),
            legacy_auth,
        )
    if name == "MCPService":
        values = _keywords(_constructor(node, name))
        if set(values) not in ({"id", "service"}, {"id", "service", "auth"}):
            raise AgentCliError("The CLI-owned MCP integration is not in canonical form.")
        auth, legacy_auth = _auth(values)
        return (
            MCPService(
                id=_string(values["id"], "integration id"),
                service=_string(values["service"], "MCP service"),
                auth=auth,
            ),
            legacy_auth,
        )
    if name == "UCFunction":
        values = _keywords(_constructor(node, name))
        if set(values) != {"id", "function"}:
            raise AgentCliError("The CLI-owned UC function integration is not in canonical form.")
        return (
            UCFunction(
                id=_string(values["id"], "integration id"),
                function=_string(values["function"], "UC function"),
            ),
            False,
        )
    raise AgentCliError(f"Unsupported constructor {name!r} in the CLI-owned integration registry.")


def _parse(
    source_text: str,
) -> tuple[list[Integration], dict[str, int], frozenset[str]]:
    try:
        tree = ast.parse(source_text)
        compile(tree, "<databricks_tools.py>", "exec")
    except SyntaxError as exc:
        raise AgentCliError(
            f"Could not parse the CLI-owned integration registry: {exc.msg}."
        ) from exc
    statements = list(tree.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
    ):
        statements.pop(0)
    if len(statements) != 2:
        raise AgentCliError("The CLI-owned integration registry is not in canonical form.")
    import_node, assignment = statements
    if not (
        isinstance(import_node, ast.Import)
        and len(import_node.names) == 1
        and import_node.names[0].name == "databricks_mason.integrations"
        and import_node.names[0].asname == _ALIAS
        and isinstance(assignment, ast.AnnAssign)
        and isinstance(assignment.target, ast.Name)
        and assignment.target.id == "DATABRICKS_TOOLS"
        and _canonical_annotation(assignment.annotation)
        and isinstance(assignment.value, ast.Tuple)
    ):
        raise AgentCliError("The CLI-owned integration registry is not in canonical form.")
    parsed = [_integration(item) for item in assignment.value.elts]
    integrations = [integration for integration, _ in parsed]
    ids = [item.id for item in integrations]
    if len(ids) != len(set(ids)):
        raise AgentCliError("DATABRICKS_TOOLS integration ids must be unique.")
    lines = {
        item.id: node.lineno for item, node in zip(integrations, assignment.value.elts, strict=True)
    }
    legacy_auth_ids = frozenset(
        integration.id for integration, legacy_auth in parsed if legacy_auth
    )
    return integrations, lines, legacy_auth_ids


def _canonical_annotation(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tuple"
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[0], ast.Attribute)
        and isinstance(node.slice.elts[0].value, ast.Name)
        and node.slice.elts[0].value.id == _ALIAS
        and node.slice.elts[0].attr == "Integration"
        and isinstance(node.slice.elts[1], ast.Constant)
        and node.slice.elts[1].value is Ellipsis
    )


def _render_scope(scope: Scope, indent: str) -> list[str]:
    return [
        f"{indent}{_ALIAS}.Scope.{scope.kind}(",
        f"{indent}    {json.dumps(scope.value)},",
        f"{indent}    permission={json.dumps(scope.permission)},",
        f"{indent}),",
    ]


def _render_integration(integration: Integration) -> list[str]:
    prefix = f"    {_ALIAS}."
    if isinstance(integration, Sandbox):
        lines = [
            f"{prefix}Sandbox(",
            f"        id={json.dumps(integration.id)},",
            "        scopes=(",
        ]
        for scope in integration.scopes:
            lines.extend(_render_scope(scope, "            "))
        lines.extend(
            [
                "        ),",
                f"        auth={json.dumps(integration.auth)},",
                "    ),",
            ]
        )
        return lines
    if isinstance(integration, MCPService):
        return [
            f"{prefix}MCPService(",
            f"        id={json.dumps(integration.id)},",
            f"        service={json.dumps(integration.service)},",
            f"        auth={json.dumps(integration.auth)},",
            "    ),",
        ]
    return [
        f"{prefix}UCFunction(",
        f"        id={json.dumps(integration.id)},",
        f"        function={json.dumps(integration.function)},",
        "    ),",
    ]


def render_registry(integrations: Iterable[Integration]) -> str:
    lines = [
        '"""Databricks integrations selected for this application.\n\n',
        "Maintained by `mason tools`; the CLI parses this module without executing it.\n",
        '"""\n\n',
        f"import databricks_mason.integrations as {_ALIAS}\n\n",
        f"DATABRICKS_TOOLS: tuple[{_ALIAS}.Integration, ...] = (\n",
    ]
    for integration in integrations:
        lines.extend(f"{line}\n" for line in _render_integration(integration))
    lines.append(")\n")
    source_text = "".join(lines)
    ast.parse(source_text)
    return source_text


def _summary(integration: Integration) -> str:
    if isinstance(integration, Sandbox):
        scopes = ", ".join(f"{scope.resource} ({scope.permission})" for scope in integration.scopes)
        return f"sandbox (auth={integration.auth}): {scopes}"
    if isinstance(integration, MCPService):
        return f"MCP service (auth={integration.auth}): {integration.service}"
    return f"UC function: {integration.function}"


class IntegrationRegistry:
    """Mutable CLI view of one dedicated generated Python registry."""

    def __init__(
        self,
        root: pathlib.Path,
        integrations: list[Integration],
        definition_lines: dict[str, int] | None = None,
        *,
        relative_path: pathlib.Path = _DEFAULT_RELATIVE_PATH,
        legacy_auth_ids: frozenset[str] | None = None,
    ) -> None:
        self.root = root
        self.path = root / relative_path
        self.relative_path = relative_path
        self.integrations = integrations
        self._definition_lines = definition_lines or {}
        self.legacy_auth_ids = legacy_auth_ids or frozenset()

    @classmethod
    def empty(
        cls,
        root: pathlib.Path | str,
        *,
        relative_path: pathlib.Path = _DEFAULT_RELATIVE_PATH,
    ) -> IntegrationRegistry:
        return cls(
            pathlib.Path(root).expanduser().resolve(),
            [],
            relative_path=relative_path,
        )

    @classmethod
    def load(
        cls,
        root: pathlib.Path | str,
        *,
        relative_path: pathlib.Path = _DEFAULT_RELATIVE_PATH,
    ) -> IntegrationRegistry:
        project_root = pathlib.Path(root).expanduser().resolve()
        path = project_root / relative_path
        try:
            source_text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise AgentCliError(
                f"Could not find Databricks integration registry at {path}."
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise AgentCliError(
                f"Could not read Databricks integration registry at {path}: {exc}."
            ) from exc
        integrations, lines, legacy_auth_ids = _parse(source_text)
        return cls(
            project_root,
            integrations,
            lines,
            relative_path=relative_path,
            legacy_auth_ids=legacy_auth_ids,
        )

    def add(self, integration: Integration) -> bool:
        for existing in self.integrations:
            if existing.id != integration.id:
                continue
            if existing.id in self.legacy_auth_ids:
                raise AgentCliError(
                    f"Integration id {integration.id!r} has unspecified legacy auth.",
                    hint='Edit its declaration to add auth="user" or auth="app".',
                )
            if existing == integration:
                return False
            raise AgentCliError(
                f"Integration id {integration.id!r} already exists with a different configuration "
                f"(existing: {_summary(existing)}; requested: {_summary(integration)}).",
                hint="Use --name to choose a different id.",
            )
        self.integrations.append(integration)
        return True

    def write(self) -> pathlib.Path:
        if self.legacy_auth_ids:
            rendered = ", ".join(repr(item) for item in sorted(self.legacy_auth_ids))
            raise AgentCliError(
                f"Cannot write the integration registry while legacy entries omit auth: "
                f'{rendered}. Add auth="user" or auth="app" to each declaration.'
            )
        source_text = render_registry(self.integrations)
        temporary: pathlib.Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = pathlib.Path(output.name)
                output.write(source_text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise AgentCliError(
                f"Could not write Databricks integration registry at {self.path}: {exc}."
            ) from exc
        _, self._definition_lines, self.legacy_auth_ids = _parse(source_text)
        return self.path

    def definition_line(self, integration_id: str) -> int:
        try:
            return self._definition_lines[integration_id]
        except KeyError as exc:
            raise AgentCliError(f"Integration {integration_id!r} has not been written.") from exc
