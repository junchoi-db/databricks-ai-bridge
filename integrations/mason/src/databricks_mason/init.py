"""`mason init` — scaffold a local agent project from a mason template.

Fetches one template directory out of its git repo (a sparse, blobless clone so only the
chosen template is materialized) and drops it into a local target directory, ready for
`mason deploy --source <dir>`.

`--framework` selects which template to lay down; each framework knows its own repo, ref, and
path (see `_TEMPLATES`). `--repo` / `--ref` override those, e.g. to pull from a fork or branch
before a template has merged to its canonical repo.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as _installed_distribution
from importlib.metadata import version as _installed_version
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import IntegrationRegistry, registry_relative_path
from databricks_mason.project_config import REQUEST_AUTH_CONTRACT_VERSION, write_project_metadata

# Each framework's template has its own home: the git repo, ref, and path-within-repo to fetch.
# Both basic templates live in this repo, versioned in lockstep with the CLI (see below).
# `--repo` / `--ref` override the repo/ref here, e.g. to pull from a fork or branch before merge.
_MASON_REPO = "https://github.com/databricks/databricks-ai-bridge.git"
_TEMPLATES: dict[str, dict[str, str]] = {
    "openai": {
        "repo": _MASON_REPO,
        "ref": "main",
        "path": "integrations/mason/templates/agent-openai",
    },
    "langgraph": {
        "repo": _MASON_REPO,
        "ref": "main",
        "path": "integrations/mason/templates/agent-langgraph",
    },
}

# Frameworks whose template lives in this repo and is released in lockstep with the CLI: a scaffold
# they produce pins `databricks-mason[runtime]` at this package's version, so init fetches the
# template tagged for the installed CLI (see `_template_ref`) rather than `main`. That keeps a
# user's scaffold from outrunning the `databricks-mason` they have installed.
_VERSIONED_TEMPLATES = frozenset({"langgraph", "openai"})

# The release workflow tags each published version `databricks-mason-v<version>`.
_RELEASE_TAG_PREFIX = "databricks-mason-v"

_CHAT_APP_TEMPLATES = {
    "langgraph": "integrations/mason/templates/ui/agent-langgraph",
    "openai": "integrations/mason/templates/ui/agent-openai",
}

_TEMPLATE_RUNTIME_PATH = pathlib.Path("runtime/runtime.py")
_REQUEST_AUTH_CONTRACT_MARKER = "REQUEST_AUTH_CONTRACT_VERSION"
_RUNTIME_EXTRAS = {"langgraph": "runtime", "openai": "runtime-openai"}


def _template_request_auth_contract_version(template: pathlib.Path) -> int | None:
    """Read a supported literal request-auth marker without importing template code."""
    try:
        source = (template / _TEMPLATE_RUNTIME_PATH).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return None

    assignments: list[ast.AST | None] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            matching_targets = [
                target
                for target in statement.targets
                if isinstance(target, ast.Name) and target.id == _REQUEST_AUTH_CONTRACT_MARKER
            ]
            if matching_targets:
                assignments.append(statement.value if len(statement.targets) == 1 else None)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == _REQUEST_AUTH_CONTRACT_MARKER
        ):
            assignments.append(statement.value)
        elif (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == _REQUEST_AUTH_CONTRACT_MARKER
        ):
            assignments.append(None)

    if len(assignments) != 1:
        return None
    value = assignments[0]
    if (
        not isinstance(value, ast.Constant)
        or type(value.value) is not int
        or value.value != REQUEST_AUTH_CONTRACT_VERSION
    ):
        return None
    return value.value


def _template_ref(framework: str) -> str:
    """The git ref to fetch a framework's template from, absent a `--ref` override.

    For a versioned framework, fetch the tag matching the installed CLI so the scaffold's pinned
    `databricks-mason` matches what the user has. Fall back to the default ref when the version
    isn't a published release. Direct/editable installs and local `+` versions have no matching
    tag, so those keep fetching `main`; a prerelease such as `0.1.2.dev0` installed from an index
    is still a published release and uses its tag.
    """
    default_ref = _TEMPLATES[framework]["ref"]
    if framework not in _VERSIONED_TEMPLATES:
        return default_ref
    try:
        installed = _installed_version("databricks-mason")
    except PackageNotFoundError:
        return default_ref
    if _is_direct_url_install() or "+" in installed:
        return default_ref
    return f"{_RELEASE_TAG_PREFIX}{installed}"


def _is_direct_url_install() -> bool:
    """Whether Mason came from a local/direct URL rather than a package index."""

    try:
        direct_url = _installed_distribution("databricks-mason").read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    return bool(direct_url and direct_url.strip())


def _pin_template_runtime(dest: pathlib.Path, framework: str) -> str | None:
    """Pin a generated release scaffold to the exact installed Mason runtime."""

    pyproject = dest / "pyproject.toml"
    if framework not in _VERSIONED_TEMPLATES or not pyproject.is_file():
        return None
    try:
        installed = _installed_version("databricks-mason")
    except PackageNotFoundError:
        return None
    if _is_direct_url_install() or "+" in installed:
        return None

    extra = _RUNTIME_EXTRAS[framework]
    source = pyproject.read_text(encoding="utf-8")
    dependency = re.compile(
        rf'(?P<quote>["\'])databricks-mason\[{re.escape(extra)}\][^"\']*(?P=quote)'
    )
    matches = list(dependency.finditer(source))
    if len(matches) != 1:
        raise AgentCliError(
            f"Expected one databricks-mason[{extra}] dependency in {pyproject}.",
            hint="The selected template is not compatible with this Mason release.",
        )
    quote = matches[0].group("quote")
    pinned = f"databricks-mason[{extra}]=={installed}"
    updated = source[: matches[0].start()] + f"{quote}{pinned}{quote}" + source[matches[0].end() :]
    pyproject.write_text(updated, encoding="utf-8")
    return pinned


def _git(args: list[str], *, cwd: Optional[pathlib.Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AgentCliError(
            f"`git {' '.join(args)}` failed (exit {result.returncode})", hint=detail
        )
    return result


def _fetch_template(
    repo: str,
    ref: str,
    template_dir: str,
    dest: pathlib.Path,
    overlay_dirs: tuple[str, ...] = (),
) -> None:
    """Sparse-clone a template and optional overlays from `repo`@`ref` into `dest`."""
    with tempfile.TemporaryDirectory(prefix="mason-init-") as tmp:
        clone = pathlib.Path(tmp) / "repo"
        _git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                repo,
                str(clone),
            ]
        )
        template_dirs = (template_dir, *overlay_dirs)
        _git(["sparse-checkout", "set", *template_dirs], cwd=clone)
        for index, path in enumerate(template_dirs):
            src = clone / path
            if not src.is_dir():
                raise AgentCliError(
                    f"Template '{path}' not found in {repo}@{ref}.",
                    hint=(
                        "It may not have merged yet — pass --repo/--ref to target a fork or branch."
                    ),
                )
            shutil.copytree(src, dest, dirs_exist_ok=index > 0)


def _write_env(dest: pathlib.Path, profile: str) -> bool:
    """Seed a local `.env` from `.env.example` with DATABRICKS_CONFIG_PROFILE=<profile>.

    Returns True if a `.env` was written. Skips if `.env` already exists (never clobbers). The
    template reads DATABRICKS_CONFIG_PROFILE for local model auth, so this makes the scaffolded
    project runnable with `uv run start-server` without a manual `cp .env.example .env` step.
    """
    env_path = dest / ".env"
    if env_path.exists():
        return False
    example = dest / ".env.example"
    base = example.read_text() if example.exists() else ""
    lines, replaced = [], False
    for line in base.splitlines():
        if line.startswith("DATABRICKS_CONFIG_PROFILE="):
            lines.append(f"DATABRICKS_CONFIG_PROFILE={profile}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"DATABRICKS_CONFIG_PROFILE={profile}")
    env_path.write_text("\n".join(lines) + "\n")
    return True


@click.command(name="init")
@click.argument("directory", required=False)
@click.option(
    "--framework",
    type=click.Choice(sorted(_TEMPLATES)),
    default="langgraph",
    show_default=True,
    help="Which basic agent template to scaffold.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `uv run start-server` works "
    "immediately (defaults to the profile from -p / `mason login`).",
)
@click.option(
    "--disable-chat-app",
    is_flag=True,
    help="Scaffold the API-only backend, without the browser chat app.",
)
@click.option(
    "--enable-chat-app",
    is_flag=True,
    hidden=True,
    help="Deprecated: the chat app is included by default; this flag is a no-op.",
)
@click.option("--repo", default=None, help="Override the git repo URL to fetch the template from.")
@click.option("--ref", default=None, help="Override the branch, tag, or ref to fetch.")
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    framework: str,
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
    repo: Optional[str],
    ref: Optional[str],
) -> None:
    """Scaffold a local agent project from a mason template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `uv run start-server` right away.
    """
    # The chat app is included by default for frameworks that have one; --disable-chat-app opts out.
    # (--enable-chat-app is a deprecated no-op kept for back-compat.)
    chat_app_enabled = framework in _CHAT_APP_TEMPLATES and not disable_chat_app

    spec = _TEMPLATES[framework]
    template_path = spec["path"]
    dest = (
        pathlib.Path(directory)
        if directory
        else pathlib.Path(pathlib.PurePosixPath(template_path).name)
    )

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )

    overlay_dirs = (_CHAT_APP_TEMPLATES[framework],) if chat_app_enabled else ()
    _fetch_template(
        repo or spec["repo"],
        ref or _template_ref(framework),
        template_path,
        dest,
        overlay_dirs,
    )
    _pin_template_runtime(dest, framework)

    template_name = pathlib.PurePosixPath(template_path).name
    request_auth_contract_version = _template_request_auth_contract_version(dest)
    write_project_metadata(
        dest,
        framework=framework,
        template=template_name,
        request_auth_contract_version=request_auth_contract_version,
    )
    relative_registry = registry_relative_path(framework)
    registry_path = dest / relative_registry
    if registry_path.is_file():
        IntegrationRegistry.load(dest, relative_path=relative_registry)
    else:
        IntegrationRegistry.empty(dest, relative_path=relative_registry).write()
    env_profile = profile or obj.profile
    wrote_env = _write_env(dest, env_profile) if env_profile else False

    if obj.output == "json":
        render.emit_json(
            {
                "framework": framework,
                "template": template_name,
                "directory": str(dest),
                "chat_app_enabled": chat_app_enabled,
                "env_profile": env_profile if wrote_env else None,
                "request_auth_contract_version": request_auth_contract_version,
                "extra_user_api_scopes": [],
            }
        )
        return

    fields = {"Framework": framework, "Directory": str(dest)}
    if chat_app_enabled:
        fields["Chat app"] = "enabled"
    steps = [f"cd {dest}"]
    if wrote_env:
        fields["Profile (.env)"] = env_profile
    else:
        # No profile resolved, so no .env was seeded — call out the auth step explicitly rather
        # than burying it, since running locally fails without a Databricks profile.
        steps += [
            "cp .env.example .env",
            "Set DATABRICKS_CONFIG_PROFILE in .env (or re-run `mason init --profile <profile>`)",
        ]
    steps += ["mason dev        # run locally"]
    if chat_app_enabled:
        steps.append("Open http://localhost:8000")
    steps.append(f"mason deploy <name> --source {dest}")
    render.success(f"Scaffolded '{template_name}'", fields=fields, next_steps=steps)
