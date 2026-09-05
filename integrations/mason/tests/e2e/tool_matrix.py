#!/usr/bin/env python3
"""Run the LangGraph × CLI/direct × dev/deploy × tool E2E matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from databricks.sdk import WorkspaceClient

FRAMEWORKS = ("langgraph",)
AUTHORING_PATHS = ("cli", "direct")
RUNTIMES = ("dev", "deploy")
TOOL_KINDS = ("sandbox", "mcp", "python", "uc_function")

GOOGLE_DRIVE_SERVICE = "system.ai.google_drive"
GOOGLE_DRIVE_TOOL = "google_drive_list_recent"
GOOGLE_DRIVE_MAX_RESULTS = 3
GOOGLE_DRIVE_REDACTED_ACTUAL = (
    "MASON_GOOGLE_DRIVE_OK count=<validated in memory>; raw Drive output redacted"
)

APPS_INGRESS_AUTHORIZATION_FAILED = "APPS_INGRESS_AUTHORIZATION_FAILED"
MATRIX_HTTP_ERROR = "MATRIX_HTTP_ERROR"
MATRIX_EXECUTION_FAILED = "MATRIX_EXECUTION_FAILED"
_MASON_RUNTIME_ERROR_MESSAGES = {
    "MCP_USER_AUTHORIZATION_MISSING": (
        "Mason received no Databricks Apps-forwarded request-user credential."
    ),
    "MCP_USER_AUTHORIZATION_INVALID": (
        "Mason rejected an invalid or expired request-user credential."
    ),
    "MCP_PERMISSION_DENIED": (
        "The request user lacks permission for the configured MCP integration."
    ),
    "MCP_AUTHORIZATION_REQUIRED": (
        "Google Drive requires an additional provider authorization step."
    ),
}

PROMPTS = {
    "sandbox": (
        "You must call the sandbox tool and use Spark to read the marker column from exactly one "
        "row of {sandbox_table}. Print and return only that value. Do not synthesize the marker."
    ),
    "mcp": (
        "You must call google_drive_list_recent with max_results=3. Inspect the result only to "
        "count returned items. Do not reveal or repeat names, titles, URLs, IDs, owners, snippets, "
        "or file content. Return exactly `MASON_GOOGLE_DRIVE_OK count=N`, where N is an integer "
        "from 0 through 3, and return nothing else."
    ),
    "python": (
        "You must call the matrix_marker Python tool with value 'matrix'. Return its exact result."
    ),
    "uc_function": "",
}

EXPECTED = {
    "sandbox": "MASON_APP_SANDBOX_OK",
    "python": "MASON_PYTHON_OK",
    "uc_function": "MASON_UC_OK:matrix",
    "mcp": (
        "google_drive_list_recent(max_results=3) and a privacy-safe "
        "MASON_GOOGLE_DRIVE_OK count=N summary"
    ),
}

_EVIDENCE_AUTH = {
    "sandbox": {
        "auth_mode": "app",
        "execution_principal": "dedicated Databricks App service principal",
        "app_service_principal_granted": True,
    },
    "mcp": {
        "auth_mode": "user",
        "execution_principal": "request user",
        "app_service_principal_granted": False,
    },
    "python": {
        "auth_mode": "local",
        "execution_principal": "agent process",
        "app_service_principal_granted": False,
    },
    "uc_function": {
        "auth_mode": "app/default",
        "execution_principal": "environment default (App service principal when deployed)",
        "app_service_principal_granted": True,
    },
}


class MatrixError(RuntimeError):
    """A reproducible setup or execution failure."""


class MatrixInvocationError(MatrixError):
    """A privacy-safe invocation failure with a stable evidence code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclasses.dataclass
class EvidenceRow:
    framework: str
    authoring: str
    runtime: str
    tool_kind: str
    auth_mode: str
    execution_principal: str
    app_service_principal_granted: bool
    status: str
    command: str
    expected: str
    actual: str
    duration_seconds: float
    artifact_paths: list[str]
    app_name: str | None = None
    app_url: str | None = None
    error_code: str | None = None
    error: str | None = None


@dataclasses.dataclass
class ProjectCase:
    framework: str
    authoring: str
    path: pathlib.Path
    app_name: str


class Transcript:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        line = text.rstrip() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def command(self, argv: Sequence[str], cwd: pathlib.Path | None = None) -> None:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        self.write(f"$ {prefix}{shlex.join(list(argv))}")

    def file_step(self, path: pathlib.Path, description: str) -> None:
        self.write(f"# write {path}: {description}")


def _sandbox_principal_probe_code(sandbox_table: str, expected_principal: str) -> str:
    """Return the exact Sandbox program that proves which App principal executed it."""
    return (
        "from pyspark.sql import SparkSession\n"
        "spark = SparkSession.builder.getOrCreate()\n"
        'actual_principal = spark.sql("SELECT current_user() AS principal").first()["principal"]\n'
        f"expected_principal = {json.dumps(expected_principal)}\n"
        "if actual_principal != expected_principal:\n"
        '    raise RuntimeError("unexpected App service principal")\n'
        f"marker = spark.table({json.dumps(sandbox_table)})"
        '.select("marker").limit(1).collect()[0]["marker"]\n'
        "print(marker)"
    )


def _sandbox_principal_probe_prompt(code: str) -> str:
    return (
        "You must call the sandbox run_code tool exactly once with language `python` and pass "
        "the following code exactly, without edits:\n"
        f"```python\n{code}\n```\n"
        "Return only the tool's marker value. Do not synthesize the marker."
    )


class Runner:
    def __init__(
        self,
        profile: str,
        output: pathlib.Path,
        wheel: pathlib.Path,
        template_repo: str | None = None,
        template_ref: str | None = None,
        app_auth_profile: str | None = None,
        freshness_marker: str | None = None,
    ):
        self.profile = profile
        self.output = output
        self.wheel = wheel.resolve()
        self.template_repo = template_repo
        self.template_ref = template_ref
        self.app_auth_profile = app_auth_profile or profile
        if (
            freshness_marker is not None
            and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", freshness_marker) is None
        ):
            raise MatrixError(
                "--freshness-marker must contain 1-64 letters, digits, dots, underscores, or hyphens."
            )
        self.freshness_marker = freshness_marker
        self.transcript = Transcript(output / "commands.log")
        self.runner_venv = output / "runner-venv"
        self.mason = self.runner_venv / "bin" / "mason"
        self.rows: list[EvidenceRow] = []
        self.apps: list[str] = []
        self.uc_function: str | None = None
        self.sandbox_table: str | None = None
        self.warehouse_id: str | None = None
        self.host: str | None = None
        self.headers: dict[str, str] = {}
        self.runtime_wheel_sources: dict[str, dict[str, str]] = {}
        self.google_drive_discovered = False

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 300,
        log: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if log:
            self.transcript.command(argv, cwd)
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log and result.stdout.strip():
            self.transcript.write(result.stdout)
        if log and result.stderr.strip():
            self.transcript.write(result.stderr)
        if check and result.returncode != 0:
            raise MatrixError(
                f"Command failed ({result.returncode}): {shlex.join(list(argv))}\n"
                f"{result.stderr or result.stdout}"
            )
        return result

    def run_long(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 1800,
    ) -> str:
        self.transcript.command(argv, cwd)
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s; log: {log_path}")
                if elapsed >= next_tick:
                    last = _last_nonempty_line(log_path)
                    self.transcript.write(
                        f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | running | {last}"
                    )
                    next_tick += 60.0
                time.sleep(2)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        self.transcript.write(output)
        if process.returncode != 0:
            raise MatrixError(f"{label} failed ({process.returncode}); log: {log_path}")
        self.transcript.write(f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | success")
        return output

    def databricks(self, args: Sequence[str], *, timeout: float = 300) -> dict[str, Any]:
        result = self.run(
            ["databricks", *args, "--profile", self.profile, "--output", "json"],
            timeout=timeout,
        )
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MatrixError(f"Databricks CLI returned invalid JSON: {result.stdout}") from exc

    def bootstrap(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.run(["uv", "venv", str(self.runner_venv)], timeout=300)
        self.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.runner_venv / "bin" / "python"),
                str(self.wheel),
            ],
            timeout=600,
        )
        self.run([str(self.mason), "tools", "--help"])
        self.discover_google_drive_service()
        workspace_client = WorkspaceClient(profile=self.profile)
        app_auth_client = WorkspaceClient(profile=self.app_auth_profile)
        if not workspace_client.config.host:
            raise MatrixError(f"Could not resolve a host from profile {self.profile!r}.")
        if not app_auth_client.config.host:
            raise MatrixError(
                f"Could not resolve a host from App auth profile {self.app_auth_profile!r}."
            )
        self.host = workspace_client.config.host.rstrip("/")
        app_auth_host = app_auth_client.config.host.rstrip("/")
        if app_auth_host != self.host:
            raise MatrixError(
                f"App auth profile {self.app_auth_profile!r} targets {app_auth_host}, "
                f"not {self.host}."
            )
        if app_auth_client.config.auth_type == "pat":
            raise MatrixError(
                f"App auth profile {self.app_auth_profile!r} uses a PAT. "
                "Databricks Apps /api routes require OAuth; run `databricks auth login` "
                "for a profile on the same workspace."
            )
        authorization = app_auth_client.config.authenticate().get("Authorization")
        if not authorization:
            raise MatrixError(
                f"Could not resolve credentials from App auth profile {self.app_auth_profile!r}."
            )
        self.headers = {"Authorization": authorization}

    def discover_google_drive_service(self) -> None:
        argv = [
            str(self.mason),
            "--profile",
            self.profile,
            "--output",
            "json",
            "mcp",
            "list",
            "--schema",
            "system.ai",
        ]
        self.transcript.command(argv)
        result = self.run(argv, log=False)
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MatrixError("MCP service discovery returned invalid JSON.") from exc
        services = document.get("mcp_services") if isinstance(document, dict) else None
        if not isinstance(services, list) or not any(
            isinstance(service, dict) and service.get("name") == GOOGLE_DRIVE_SERVICE
            for service in services
        ):
            raise MatrixError(f"MCP service discovery did not return {GOOGLE_DRIVE_SERVICE}.")
        self.google_drive_discovered = True
        self.transcript.write(f"# service discovery: {GOOGLE_DRIVE_SERVICE} is available")

    def select_warehouse(self, override: str | None) -> str:
        if override:
            self.warehouse_id = override
        else:
            warehouses = self.databricks(["warehouses", "list"])
            if not isinstance(warehouses, list) or not warehouses:
                raise MatrixError("df1 has no SQL warehouse available for UC function setup.")
            running = next(
                (item for item in warehouses if item.get("state") == "RUNNING"), warehouses[0]
            )
            self.warehouse_id = str(running["id"])
        self.run_long(
            "warehouse-start",
            [
                "databricks",
                "warehouses",
                "start",
                self.warehouse_id,
                "--profile",
                self.profile,
                "--timeout",
                "20m",
            ],
            timeout=1250,
        )
        return self.warehouse_id

    def sql(self, statement: str, *, timeout: float = 600) -> dict[str, Any]:
        if self.warehouse_id is None:
            raise MatrixError("SQL warehouse was not selected.")
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": statement,
            "wait_timeout": "30s",
            "on_wait_timeout": "CONTINUE",
        }
        response = self.databricks(
            ["api", "post", "/api/2.0/sql/statements", "--json", json.dumps(payload)],
            timeout=60,
        )
        statement_id = response.get("statement_id")
        while response.get("status", {}).get("state") in {"PENDING", "RUNNING"}:
            if not statement_id:
                raise MatrixError(f"SQL response has no statement_id: {response}")
            if timeout <= 0:
                raise MatrixError(f"SQL statement timed out: {statement_id}")
            time.sleep(10)
            timeout -= 10
            response = self.databricks(
                ["api", "get", f"/api/2.0/sql/statements/{statement_id}"], timeout=60
            )
        if response.get("status", {}).get("state") != "SUCCEEDED":
            raise MatrixError(f"SQL failed: {json.dumps(response, indent=2)}")
        return response

    def create_uc_resources(self, schema: str) -> tuple[str, str]:
        catalog, separator, schema_name = schema.partition(".")
        if not separator or not catalog or not schema_name or "." in schema_name:
            raise MatrixError("--uc-schema must be a two-part catalog.schema name.")
        self.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema_name}`")
        suffix = uuid.uuid4().hex[:8]
        function_name = f"mason_uc_{suffix}"
        table_name = f"mason_app_sandbox_{suffix}"
        self.uc_function = f"{catalog}.{schema_name}.{function_name}"
        self.sandbox_table = f"{catalog}.{schema_name}.{table_name}"
        exposed_tool_name = self.uc_function.replace(".", "__")
        if len(exposed_tool_name) > 64:
            raise MatrixError(
                "The UC function's MCP tool name would exceed 64 characters: "
                f"{exposed_tool_name!r}. Use a shorter --uc-schema."
            )
        self.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Deterministic Mason E2E marker tool' "
            "RETURN concat('MASON_UC_OK:', value)"
        )
        self.sql(
            f"CREATE OR REPLACE TABLE `{catalog}`.`{schema_name}`.`{table_name}` "
            "AS SELECT 'MASON_APP_SANDBOX_OK' AS marker"
        )
        return self.uc_function, self.sandbox_table

    def create_projects(self) -> list[ProjectCase]:
        if self.uc_function is None or self.sandbox_table is None:
            raise MatrixError("UC function and App Sandbox table were not created.")
        projects_root = self.output / "projects"
        projects_root.mkdir(parents=True, exist_ok=True)
        run_suffix = uuid.uuid4().hex[:6]
        cases: list[ProjectCase] = []
        for framework in FRAMEWORKS:
            for authoring in AUTHORING_PATHS:
                project = projects_root / f"{framework}-{authoring}"
                init_args = [
                    str(self.mason),
                    "--profile",
                    self.profile,
                    "init",
                    "--framework",
                    framework,
                    "--profile",
                    self.profile,
                ]
                if self.template_repo:
                    init_args.extend(["--repo", self.template_repo])
                if self.template_ref:
                    init_args.extend(["--ref", self.template_ref])
                init_args.append(str(project))
                self.run_long(
                    f"init-{framework}-{authoring}",
                    init_args,
                    timeout=600,
                )
                self._pin_runtime_wheel(project)
                if authoring == "cli":
                    self._author_cli(project)
                else:
                    self._author_direct(project)
                self._write_python_tool(project)
                app_name = f"mason-tools-{framework[:2]}-{authoring[:2]}-{run_suffix}"
                cases.append(ProjectCase(framework, authoring, project, app_name))
        return cases

    def _pin_runtime_wheel(self, project: pathlib.Path) -> None:
        """Make dev and Apps builds resolve the exact content-addressed branch wheel."""

        digest = _sha256(self.wheel)
        relative_wheel = pathlib.Path("vendor") / digest / self.wheel.name
        target = project / relative_wheel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.wheel, target)

        pyproject_path = project / "pyproject.toml"
        pyproject = pyproject_path.read_text(encoding="utf-8")
        if "[tool.uv.sources]" in pyproject:
            raise MatrixError(
                f"Generated project {project} already has [tool.uv.sources]; "
                "refusing to overwrite its databricks-mason source."
            )
        source = (
            f'\n[tool.uv.sources]\ndatabricks-mason = {{ path = "{relative_wheel.as_posix()}" }}\n'
        )
        pyproject_path.write_text(pyproject.rstrip() + "\n" + source, encoding="utf-8")
        self.runtime_wheel_sources[project.name] = {
            "path": relative_wheel.as_posix(),
            "sha256": digest,
        }
        self.transcript.file_step(
            target,
            f"content-addressed databricks-mason runtime wheel sha256={digest}",
        )

    def _author_cli(self, project: pathlib.Path) -> None:
        if self.sandbox_table is None:
            raise MatrixError("App Sandbox table was not created.")
        commands = [
            [
                "tools",
                "add",
                "sandbox",
                "--scope",
                f"table:{self.sandbox_table}",
                "--auth",
                "app",
            ],
            ["tools", "add", "mcp", GOOGLE_DRIVE_SERVICE, "--auth", "user"],
            [
                "tools",
                "add",
                "uc-function",
                self.uc_function or "",
                "--name",
                "mason_uc_marker",
            ],
        ]
        for args in commands:
            self.run([str(self.mason), *args, "--source", str(project)])

    def _author_direct(self, project: pathlib.Path) -> None:
        fixture = pathlib.Path(__file__).parent / "fixtures" / "direct_databricks_tools.py"
        registry = (
            fixture.read_text(encoding="utf-8")
            .replace("__UC_FUNCTION__", self.uc_function or "")
            .replace("__SANDBOX_TABLE__", self.sandbox_table or "")
        )
        target = project / "agent" / "databricks_tools.py"
        self.transcript.file_step(target, "direct authoring; no mason tools command")
        target.write_text(registry, encoding="utf-8")

    def _write_python_tool(self, project: pathlib.Path) -> None:
        digest = _sha256(self.wheel)
        result = _python_expected_result(digest, self.freshness_marker)
        freshness_print = (
            f"    print('[freshness-check {self.freshness_marker}] hit matrix_marker', flush=True)\n"
            if self.freshness_marker
            else ""
        )
        body = (
            "import hashlib\n"
            "import importlib.metadata\n"
            "import json\n\n"
            "import pathlib\n"
            "import urllib.parse\n\n"
            "from langchain_core.tools import tool\n\n\n"
            f"_EXPECTED_MASON_WHEEL_SHA256 = {digest!r}\n\n\n"
            "def _assert_mason_runtime_wheel() -> None:\n"
            "    direct_url = importlib.metadata.distribution('databricks-mason').read_text(\n"
            "        'direct_url.json'\n"
            "    )\n"
            "    if direct_url is None:\n"
            "        raise RuntimeError('Installed databricks-mason has no direct wheel provenance.')\n"
            "    provenance = json.loads(direct_url)\n"
            "    archive_info = provenance.get('archive_info', {})\n"
            "    hashes = archive_info.get('hashes', {})\n"
            "    actual = hashes.get('sha256')\n"
            "    if actual is None and isinstance(archive_info.get('hash'), str):\n"
            "        algorithm, separator, value = archive_info['hash'].partition('=')\n"
            "        actual = value if separator and algorithm == 'sha256' else None\n"
            "    if actual is None:\n"
            "        parsed = urllib.parse.urlsplit(provenance.get('url', ''))\n"
            "        if parsed.scheme != 'file':\n"
            "            raise RuntimeError('Installed databricks-mason is not from a local wheel.')\n"
            "        wheel = pathlib.Path(urllib.parse.unquote(parsed.path))\n"
            "        digest = hashlib.sha256()\n"
            "        with wheel.open('rb') as source:\n"
            "            for chunk in iter(lambda: source.read(1024 * 1024), b''):\n"
            "                digest.update(chunk)\n"
            "        actual = digest.hexdigest()\n"
            "    if actual != _EXPECTED_MASON_WHEEL_SHA256:\n"
            "        raise RuntimeError('Installed databricks-mason wheel does not match E2E artifact.')\n\n\n"
            "@tool\n"
            "def matrix_marker(value: str) -> str:\n"
            '    """Verify and return the deterministic Mason E2E runtime marker."""\n'
            f"{freshness_print}"
            "    _assert_mason_runtime_wheel()\n"
            f"    return {result!r}\n"
        )
        target = project / "agent" / "tools" / "matrix_marker.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.transcript.file_step(target, "user-owned deterministic MASON_PYTHON_OK implementation")
        target.write_text(body, encoding="utf-8")

    def run_dev(self, case: ProjectCase, port: int) -> None:
        label = f"dev-{case.framework}-{case.authoring}"
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.mason),
            "--profile",
            self.profile,
            "dev",
            "--source",
            str(case.path),
            "--app-port",
            str(port),
            "--prepare-environment",
        ]
        self.transcript.command(argv)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            self._wait_for_local(process, port, label, log_path)
            self._exercise(case, "dev", f"http://127.0.0.1:{port}", {}, log_path)
        except Exception as exc:
            self._record_runtime_failure(case, "dev", exc, log_path)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            self.transcript.write(
                f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | stopped"
            )

    def _wait_for_local(
        self,
        process: subprocess.Popen[str],
        port: int,
        label: str,
        log_path: pathlib.Path,
    ) -> None:
        started = time.monotonic()
        next_tick = 60.0
        while True:
            if process.poll() is not None:
                raise MatrixError(
                    f"{label} exited {process.returncode}: {_last_lines(log_path, 30)}"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - started
            if elapsed > 1200:
                raise MatrixError(f"{label} did not become reachable: {_last_lines(log_path, 30)}")
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | starting | "
                    f"{_last_nonempty_line(log_path)}"
                )
                next_tick += 60
            time.sleep(5)

    def deploy(self, case: ProjectCase) -> None:
        label = f"deploy-{case.framework}-{case.authoring}"
        log_path = self.output / "logs" / f"{label}.log"
        try:
            self.run_long(
                label,
                [
                    str(self.mason),
                    "--profile",
                    self.profile,
                    "deploy",
                    case.app_name,
                    "--source",
                    str(case.path),
                ],
                timeout=2400,
            )
            self.apps.append(case.app_name)
            app = self._wait_for_app(case.app_name)
            expected_sandbox_principal = self._grant_app_resources(app)
            url = str(app.get("url") or "").rstrip("/")
            if not url:
                raise MatrixError(f"App {case.app_name} has no URL: {app}")
            self._exercise(
                case,
                "deploy",
                url,
                self.headers,
                log_path,
                app_name=case.app_name,
                expected_sandbox_principal=expected_sandbox_principal,
            )
        except Exception as exc:
            self._record_runtime_failure(case, "deploy", exc, log_path, case.app_name)

    def _wait_for_app(self, name: str) -> dict[str, Any]:
        started = time.monotonic()
        next_tick = 0.0
        while time.monotonic() - started < 1200:
            app = self.databricks(["apps", "get", name])
            compute = app.get("compute_status", {})
            state = compute.get("state") if isinstance(compute, dict) else None
            if state == "ACTIVE" and app.get("url"):
                return app
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                self.transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | app-{name} | {state or 'UNKNOWN'}"
                )
                next_tick += 60
            time.sleep(15)
        raise MatrixError(f"App {name} did not become ACTIVE.")

    def _grant_app_resources(self, app: dict[str, Any]) -> str:
        principal = app.get("service_principal_client_id")
        if not principal or self.uc_function is None or self.sandbox_table is None:
            raise MatrixError(f"App response has no service_principal_client_id: {app}")
        for statement in _app_grant_statements(
            str(principal),
            sandbox_table=self.sandbox_table,
            uc_function=self.uc_function,
        ):
            self.sql(statement)
        return str(principal)

    def _exercise(
        self,
        case: ProjectCase,
        runtime: str,
        base_url: str,
        headers: dict[str, str],
        log_path: pathlib.Path,
        app_name: str | None = None,
        expected_sandbox_principal: str | None = None,
    ) -> None:
        invocation_url = f"{base_url}{'/api' if runtime == 'deploy' else ''}/invocations"
        for tool_kind in TOOL_KINDS:
            started = time.monotonic()
            prompt = PROMPTS[tool_kind]
            expected_sandbox_code = None
            if tool_kind == "sandbox":
                if self.sandbox_table is None:
                    raise MatrixError("App Sandbox table was not created.")
                if runtime == "deploy":
                    if not expected_sandbox_principal:
                        raise MatrixError(
                            "A deployed Sandbox check requires the App service principal id."
                        )
                    expected_sandbox_code = _sandbox_principal_probe_code(
                        self.sandbox_table, expected_sandbox_principal
                    )
                    prompt = _sandbox_principal_probe_prompt(expected_sandbox_code)
                else:
                    prompt = prompt.format(sandbox_table=self.sandbox_table)
            if tool_kind == "uc_function":
                if self.uc_function is None:
                    raise MatrixError("UC function was not created.")
                exposed_tool_name = self.uc_function.replace(".", "__")
                prompt = (
                    f"You must call the tool named {exposed_tool_name} with value 'matrix'. "
                    "Do not call matrix_marker. Return the called tool's exact result."
                )
            command = _curl_command(invocation_url, prompt, bool(headers))
            expected = (
                _python_expected_result(_sha256(self.wheel), self.freshness_marker)
                if tool_kind == "python"
                else EXPECTED[tool_kind]
            )
            try:
                response = self._invoke_with_retry(
                    f"{runtime}-{case.framework}-{case.authoring}-{tool_kind}",
                    invocation_url,
                    prompt,
                    headers,
                )
                actual = _evidence_actual(
                    tool_kind,
                    response,
                    expected_python_result=expected if tool_kind == "python" else None,
                    expected_sandbox_code=expected_sandbox_code,
                )
                status, error_code, error = "pass", None, None
            except Exception as exc:
                actual = ""
                error_code, error = _safe_evidence_error(exc)
                status = "fail"
            auth_metadata = _evidence_auth_metadata(tool_kind, runtime)
            self.rows.append(
                EvidenceRow(
                    framework=case.framework,
                    authoring=case.authoring,
                    runtime=runtime,
                    tool_kind=tool_kind,
                    **auth_metadata,
                    status=status,
                    command=command,
                    expected=expected,
                    actual=actual,
                    duration_seconds=round(time.monotonic() - started, 3),
                    artifact_paths=[str(log_path)],
                    app_name=app_name,
                    app_url=base_url if runtime == "deploy" else None,
                    error_code=error_code,
                    error=error,
                )
            )
            self._write_evidence()

    def _invoke_with_retry(
        self, label: str, url: str, prompt: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                return _monitored(
                    label,
                    lambda: _http_json(
                        url, {"input": [{"role": "user", "content": prompt}]}, headers
                    ),
                    self.transcript,
                    timeout=360,
                )
            except Exception as exc:
                last = exc
                error_code, safe_message = _safe_evidence_error(exc)
                self.transcript.write(
                    f"attempt {attempt}/3 | {label} | {error_code} | {safe_message}"
                )
                if attempt < 3:
                    time.sleep(15)
        error_code, safe_message = _safe_evidence_error(last or MatrixError("unknown failure"))
        raise MatrixInvocationError(
            error_code,
            f"{label} failed after 3 attempts. {safe_message}",
        )

    def _record_runtime_failure(
        self,
        case: ProjectCase,
        runtime: str,
        exc: Exception,
        log_path: pathlib.Path,
        app_name: str | None = None,
    ) -> None:
        existing = {
            row.tool_kind
            for row in self.rows
            if row.framework == case.framework
            and row.authoring == case.authoring
            and row.runtime == runtime
        }
        for tool_kind in TOOL_KINDS:
            if tool_kind in existing:
                continue
            error_code, error = _safe_evidence_error(exc)
            auth_metadata = _evidence_auth_metadata(tool_kind, runtime)
            expected = (
                _python_expected_result(_sha256(self.wheel), self.freshness_marker)
                if tool_kind == "python"
                else EXPECTED[tool_kind]
            )
            self.rows.append(
                EvidenceRow(
                    framework=case.framework,
                    authoring=case.authoring,
                    runtime=runtime,
                    tool_kind=tool_kind,
                    **auth_metadata,
                    status="fail",
                    command="runtime setup",
                    expected=expected,
                    actual="",
                    duration_seconds=0.0,
                    artifact_paths=[str(log_path)],
                    app_name=app_name,
                    error_code=error_code,
                    error=error,
                )
            )
        self._write_evidence()

    def _write_evidence(self) -> None:
        payload = {
            "schema_version": 2,
            "profile": self.profile,
            "app_auth_profile": self.app_auth_profile,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wheel": str(self.wheel),
            "wheel_sha256": _sha256(self.wheel),
            "runtime_wheel": {
                "sha256": _sha256(self.wheel),
                "project_sources": self.runtime_wheel_sources,
                "freshness_marker": self.freshness_marker,
            },
            "uc_function": self.uc_function,
            "sandbox_table": self.sandbox_table,
            "warehouse_id": self.warehouse_id,
            "auth_boundaries": _auth_boundary_metadata(self.sandbox_table),
            "service_discovery": {
                "service": GOOGLE_DRIVE_SERVICE,
                "discovered": self.google_drive_discovered,
            },
            "rows": [dataclasses.asdict(row) for row in self.rows],
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def cleanup(self) -> None:
        for app in self.apps:
            self.run(
                ["databricks", "apps", "delete", app, "--profile", self.profile],
                timeout=600,
                check=False,
            )
        if self.uc_function:
            catalog, schema, function_name = self.uc_function.split(".")
            try:
                self.sql(f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`")
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC function | {exc}")
        if self.sandbox_table:
            catalog, schema, table_name = self.sandbox_table.split(".")
            try:
                self.sql(f"DROP TABLE IF EXISTS `{catalog}`.`{schema}`.`{table_name}`")
            except Exception as exc:
                self.transcript.write(f"cleanup warning | App Sandbox table | {exc}")


def _last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _monitored(
    label: str,
    operation: Callable[[], dict[str, Any]],
    transcript: Transcript,
    *,
    timeout: float,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started = time.monotonic()
        while True:
            try:
                return future.result(
                    timeout=min(60, max(1, timeout - (time.monotonic() - started)))
                )
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                transcript.write(
                    f"tick {dt.datetime.now(dt.timezone.utc):%H:%M} | {label} | running | {elapsed:.0f}s"
                )
                if elapsed >= timeout:
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s") from None


def _http_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=340) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise _safe_http_error(exc.code, detail) from exc
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MatrixInvocationError(
            MATRIX_HTTP_ERROR,
            "The invocation returned a non-JSON response; the response body was not retained.",
        ) from exc
    if not isinstance(value, dict):
        raise MatrixInvocationError(
            MATRIX_HTTP_ERROR,
            "The invocation returned an unexpected JSON shape; the response body was not retained.",
        )
    return value


def _safe_http_error(status: int, detail: str) -> MatrixInvocationError:
    """Classify an HTTP failure without retaining its response body or provider URL."""

    try:
        document = json.loads(detail)
    except json.JSONDecodeError:
        document = None
    error = document.get("error") if isinstance(document, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and code in _MASON_RUNTIME_ERROR_MESSAGES:
        return MatrixInvocationError(code, _MASON_RUNTIME_ERROR_MESSAGES[code])
    if status in {401, 403}:
        return MatrixInvocationError(
            APPS_INGRESS_AUTHORIZATION_FAILED,
            "Databricks Apps ingress rejected the caller before Mason handled the request; "
            "verify App CAN USE and OAuth scopes.",
        )
    return MatrixInvocationError(
        MATRIX_HTTP_ERROR,
        f"The invocation returned HTTP {status}; the response body was not retained.",
    )


def _safe_evidence_error(source: BaseException) -> tuple[str, str]:
    if isinstance(source, MatrixInvocationError):
        return source.code, str(source)
    return (
        MATRIX_EXECUTION_FAILED,
        "The matrix step failed; inspect the referenced local log. No response body was retained.",
    )


def _walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _bounded_google_drive_calls(response: dict[str, Any]) -> tuple[set[str], bool]:
    call_ids: set[str] = set()
    has_unidentified_call = False
    for value in _walk_json(response):
        if not isinstance(value, dict) or value.get("name") != GOOGLE_DRIVE_TOOL:
            continue
        arguments = value.get("args", value.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict) and arguments.get("max_results") == GOOGLE_DRIVE_MAX_RESULTS:
            call_id = value.get("id")
            if isinstance(call_id, str) and call_id:
                call_ids.add(call_id)
            else:
                has_unidentified_call = True
    return call_ids, has_unidentified_call


def _tool_result_has_error(value: dict[str, Any]) -> bool:
    if value.get("status") in {"error", "failed", "failure"}:
        return True
    if value.get("isError") is True or value.get("is_error") is True:
        return True
    content = value.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return False
    if isinstance(content, dict):
        return (
            content.get("isError") is True
            or content.get("is_error") is True
            or isinstance(content.get("error"), dict)
        )
    return False


def _google_drive_result_succeeded(
    response: dict[str, Any], call_ids: set[str], has_unidentified_call: bool
) -> bool:
    for value in _walk_json(response):
        if not isinstance(value, dict) or (
            value.get("type") != "tool" and value.get("role") != "tool"
        ):
            continue
        result_call_id = value.get("tool_call_id")
        matches_call = isinstance(result_call_id, str) and result_call_id in call_ids
        matches_unidentified = has_unidentified_call and value.get("name") == GOOGLE_DRIVE_TOOL
        if (matches_call or matches_unidentified) and not _tool_result_has_error(value):
            return True
    return False


def _assistant_text(value: dict[str, Any]) -> str | None:
    role = value.get("role")
    message_type = value.get("type")
    if role != "assistant" and message_type not in {"ai", "assistant"}:
        return None
    content = value.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        blocks = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if blocks:
            return "".join(blocks).strip()
    return None


def _sandbox_result_succeeded(
    response: dict[str, Any], *, expected_code: str | None = None
) -> bool:
    tool_names = {"sandbox", "run_code"}
    call_ids: set[str] = set()
    has_unidentified_call = False
    for value in _walk_json(response):
        if (
            not isinstance(value, dict)
            or value.get("name") not in tool_names
            or not ({"args", "arguments"} & value.keys())
        ):
            continue
        arguments = value.get("args", value.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if expected_code is not None:
            if not isinstance(arguments, dict):
                continue
            actual_code = arguments.get("code")
            if actual_code not in (expected_code, f"{expected_code}\n"):
                continue
        call_id = value.get("id")
        if isinstance(call_id, str) and call_id:
            call_ids.add(call_id)
        elif expected_code is None:
            has_unidentified_call = True
    for value in _walk_json(response):
        if not isinstance(value, dict) or (
            value.get("type") != "tool" and value.get("role") != "tool"
        ):
            continue
        result_call_id = value.get("tool_call_id")
        matches_call = isinstance(result_call_id, str) and result_call_id in call_ids
        matches_unidentified = (
            expected_code is None and has_unidentified_call and value.get("name") in tool_names
        )
        if (
            (matches_call or matches_unidentified)
            and not _tool_result_has_error(value)
            and EXPECTED["sandbox"] in json.dumps(value.get("content"), default=str)
        ):
            return True
    return False


def _assert_google_drive_semantics(response: dict[str, Any]) -> None:
    call_ids, has_unidentified_call = _bounded_google_drive_calls(response)
    if not call_ids and not has_unidentified_call:
        raise MatrixError(
            "Google Drive validation did not observe "
            "google_drive_list_recent(max_results=3); response content was discarded."
        )
    if not _google_drive_result_succeeded(response, call_ids, has_unidentified_call):
        raise MatrixError(
            "Google Drive validation did not observe a matching non-error tool result; "
            "response content was discarded."
        )
    summary = re.compile(r"MASON_GOOGLE_DRIVE_OK count=([0-3])")
    for value in _walk_json(response):
        if isinstance(value, dict) and (text := _assistant_text(value)) is not None:
            match = summary.fullmatch(text)
            if match is not None and int(match.group(1)) <= GOOGLE_DRIVE_MAX_RESULTS:
                return
    raise MatrixError(
        "Google Drive validation did not observe the exact privacy-safe success/count summary; "
        "response content was discarded."
    )


def _assert_semantics(
    tool_kind: str,
    response: dict[str, Any],
    *,
    expected_python_result: str | None = None,
    expected_sandbox_code: str | None = None,
) -> None:
    if tool_kind == "mcp":
        _assert_google_drive_semantics(response)
        return
    if tool_kind == "sandbox":
        if not _sandbox_result_succeeded(response, expected_code=expected_sandbox_code):
            requirement = (
                " from the exact App service-principal probe" if expected_sandbox_code else ""
            )
            raise MatrixError(
                "Sandbox validation did not observe a successful Sandbox tool result"
                f"{requirement} with the marker."
            )
        return
    serialized = json.dumps(response, sort_keys=True, default=str)
    if tool_kind == "python":
        if expected_python_result is None or expected_python_result not in serialized:
            raise MatrixError(
                "Python validation did not observe the current runtime wheel/freshness marker."
            )
        return
    if tool_kind == "uc_function":
        marker = EXPECTED[tool_kind]
        if marker not in serialized:
            raise MatrixError(f"Missing semantic marker {marker!r}: {serialized[:2000]}")
        return
    raise MatrixError(f"Unsupported tool kind {tool_kind!r}.")


def _evidence_actual(
    tool_kind: str,
    response: dict[str, Any],
    *,
    expected_python_result: str | None = None,
    expected_sandbox_code: str | None = None,
) -> str:
    _assert_semantics(
        tool_kind,
        response,
        expected_python_result=expected_python_result,
        expected_sandbox_code=expected_sandbox_code,
    )
    if tool_kind == "mcp":
        return GOOGLE_DRIVE_REDACTED_ACTUAL
    return json.dumps(response, sort_keys=True, default=str)[:6000]


def _split_uc_name(value: str) -> tuple[str, str, str]:
    parts = value.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise MatrixError(f"Expected a three-part Unity Catalog name, got {value!r}.")
    return parts[0], parts[1], parts[2]


def _app_grant_statements(
    principal: str,
    *,
    sandbox_table: str,
    uc_function: str,
) -> tuple[str, ...]:
    table_catalog, table_schema, table_name = _split_uc_name(sandbox_table)
    function_catalog, function_schema, function_name = _split_uc_name(uc_function)
    if (table_catalog, table_schema) != (function_catalog, function_schema):
        raise MatrixError("The App Sandbox table and UC function must share the E2E schema.")
    quoted_principal = f"`{principal.replace('`', '``')}`"
    return (
        f"GRANT USE CATALOG ON CATALOG `{table_catalog}` TO {quoted_principal}",
        f"GRANT USE SCHEMA ON SCHEMA `{table_catalog}`.`{table_schema}` TO {quoted_principal}",
        f"GRANT SELECT ON TABLE `{table_catalog}`.`{table_schema}`.`{table_name}` "
        f"TO {quoted_principal}",
        f"GRANT EXECUTE ON FUNCTION `{function_catalog}`.`{function_schema}`.`{function_name}` "
        f"TO {quoted_principal}",
    )


def _python_expected_result(wheel_sha256: str, freshness_marker: str | None) -> str:
    result = f"MASON_PYTHON_OK wheel_sha256={wheel_sha256}"
    if freshness_marker:
        result += f" freshness={freshness_marker}"
    return result


def _evidence_auth_metadata(tool_kind: str, runtime: str) -> dict[str, Any]:
    if runtime not in RUNTIMES:
        raise MatrixError(f"Unsupported runtime {runtime!r}.")
    try:
        metadata = dict(_EVIDENCE_AUTH[tool_kind])
    except KeyError as exc:
        raise MatrixError(f"Unsupported tool kind {tool_kind!r}.") from exc
    if runtime == "dev" and tool_kind in {"sandbox", "mcp", "uc_function"}:
        metadata["execution_principal"] = "selected developer profile"
        metadata["app_service_principal_granted"] = False
    return metadata


def _auth_boundary_metadata(sandbox_table: str | None) -> dict[str, dict[str, Any]]:
    return {
        "google_drive": {
            "service": GOOGLE_DRIVE_SERVICE,
            "auth": "user",
            "execution_principal": "request user",
            "app_service_principal_granted": False,
            "raw_tool_output_persisted": False,
        },
        "sandbox": {
            "service": "system.ai.sandbox",
            "auth": "app",
            "execution_principal": "dedicated Databricks App service principal",
            "app_service_principal_granted": True,
            "scope": f"table:{sandbox_table}" if sandbox_table else None,
        },
        "uc_function": {
            "auth": "app/default",
            "app_service_principal_granted": True,
        },
    }


def _expected_evidence_cells() -> set[tuple[str, str, str, str]]:
    return {
        (framework, authoring, runtime, tool)
        for framework in FRAMEWORKS
        for authoring in AUTHORING_PATHS
        for runtime in RUNTIMES
        for tool in TOOL_KINDS
    }


def _runtime_wheel_evidence_error(
    document: dict[str, Any], evidence_path: pathlib.Path
) -> str | None:
    expected_sha = document.get("wheel_sha256")
    runtime_wheel = document.get("runtime_wheel")
    if (
        not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or not isinstance(runtime_wheel, dict)
        or runtime_wheel.get("sha256") != expected_sha
    ):
        return "top-level runtime wheel SHA is missing or inconsistent"
    freshness_marker = runtime_wheel.get("freshness_marker")
    if freshness_marker is not None and (
        not isinstance(freshness_marker, str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,64}", freshness_marker) is None
    ):
        return "freshness marker is invalid"
    sources = runtime_wheel.get("project_sources")
    expected_projects = {
        f"{framework}-{authoring}" for framework in FRAMEWORKS for authoring in AUTHORING_PATHS
    }
    if not isinstance(sources, dict) or set(sources) != expected_projects:
        return "runtime wheel sources do not cover every generated project"
    for project_name, source in sources.items():
        if not isinstance(source, dict) or source.get("sha256") != expected_sha:
            return f"runtime wheel SHA is inconsistent for {project_name}"
        relative_value = source.get("path")
        if not isinstance(relative_value, str):
            return f"runtime wheel path is missing for {project_name}"
        relative_path = pathlib.PurePosixPath(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parts[:2] != ("vendor", expected_sha)
            or relative_path.suffix != ".whl"
        ):
            return f"runtime wheel path is not content-addressed for {project_name}"
        artifact = evidence_path.parent / "projects" / project_name / pathlib.Path(relative_value)
        if not artifact.is_file() or _sha256(artifact) != expected_sha:
            return f"runtime wheel artifact is missing or mismatched for {project_name}"
    return None


def _release_metadata_error(document: dict[str, Any]) -> str | None:
    if document.get("schema_version") != 2:
        return "evidence schema version 2 is required"
    boundaries = document.get("auth_boundaries")
    if not isinstance(boundaries, dict):
        return "auth boundary metadata is missing"
    google_drive = boundaries.get("google_drive")
    if not isinstance(google_drive, dict) or any(
        (
            google_drive.get("service") != GOOGLE_DRIVE_SERVICE,
            google_drive.get("auth") != "user",
            google_drive.get("execution_principal") != "request user",
            google_drive.get("app_service_principal_granted") is not False,
            google_drive.get("raw_tool_output_persisted") is not False,
        )
    ):
        return "Google Drive must remain user-authenticated, redacted, and ungranted to the App SP"
    sandbox = boundaries.get("sandbox")
    sandbox_table = document.get("sandbox_table")
    if (
        not isinstance(sandbox, dict)
        or sandbox.get("service") != "system.ai.sandbox"
        or sandbox.get("auth") != "app"
        or sandbox.get("app_service_principal_granted") is not True
        or not isinstance(sandbox_table, str)
        or sandbox.get("scope") != f"table:{sandbox_table}"
    ):
        return "Sandbox must remain an App-authenticated table-scoped fixture"
    uc_function = boundaries.get("uc_function")
    if not isinstance(uc_function, dict) or uc_function != {
        "auth": "app/default",
        "app_service_principal_granted": True,
    }:
        return "UC Function must retain its fixed App/default deployment path"
    discovery = document.get("service_discovery")
    if not isinstance(discovery, dict) or discovery != {
        "service": GOOGLE_DRIVE_SERVICE,
        "discovered": True,
    }:
        return "Google Drive service discovery evidence is missing"
    return None


def _curl_command(invocation_url: str, prompt: str, authenticated: bool) -> str:
    auth = " -H 'Authorization: Bearer <redacted>'" if authenticated else ""
    body = json.dumps({"input": [{"role": "user", "content": prompt}]})
    return (
        f"curl -sS -X POST {shlex.quote(invocation_url)}"
        f" -H 'Content-Type: application/json'{auth} --data {shlex.quote(body)}"
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(path: pathlib.Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows", [])
    expected = _expected_evidence_cells()
    actual = {
        (row["framework"], row["authoring"], row["runtime"], row["tool_kind"]) for row in rows
    }
    duplicates = len(rows) - len(actual)
    passed = sum(row.get("status") == "pass" for row in rows)
    failed = sum(row.get("status") == "fail" for row in rows)
    skipped = len(expected - actual)
    metadata_mismatches = []
    for row in rows:
        expected_metadata = _evidence_auth_metadata(row["tool_kind"], row["runtime"])
        expected_python = None
        runtime_wheel = document.get("runtime_wheel")
        if row["tool_kind"] == "python" and isinstance(runtime_wheel, dict):
            wheel_sha = runtime_wheel.get("sha256")
            marker = runtime_wheel.get("freshness_marker")
            if isinstance(wheel_sha, str) and (marker is None or isinstance(marker, str)):
                expected_python = _python_expected_result(wheel_sha, marker)
        if any(row.get(key) != value for key, value in expected_metadata.items()) or (
            row["tool_kind"] == "python" and row.get("expected") != expected_python
        ):
            metadata_mismatches.append(
                (row["framework"], row["authoring"], row["runtime"], row["tool_kind"])
            )
    wheel_error = _runtime_wheel_evidence_error(document, path)
    release_metadata_error = _release_metadata_error(document)
    sys.stdout.write(f"{passed} passed, {failed} failed, {skipped} skipped\n")
    if (
        actual != expected
        or duplicates
        or passed != len(expected)
        or metadata_mismatches
        or wheel_error is not None
        or release_metadata_error is not None
    ):
        if expected - actual:
            sys.stdout.write(f"missing cells: {sorted(expected - actual)}\n")
        if duplicates:
            sys.stdout.write(f"duplicate rows: {duplicates}\n")
        if metadata_mismatches:
            sys.stdout.write(f"invalid auth metadata: {sorted(metadata_mismatches)}\n")
        if wheel_error is not None:
            sys.stdout.write(f"invalid runtime wheel evidence: {wheel_error}\n")
        if release_metadata_error is not None:
            sys.stdout.write(f"invalid release metadata: {release_metadata_error}\n")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="df1")
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--warehouse-id")
    parser.add_argument("--uc-schema", default="main.mason_agent_tools_e2e")
    parser.add_argument("--template-repo")
    parser.add_argument("--template-ref")
    parser.add_argument(
        "--app-auth-profile",
        help="OAuth profile for deployed App /api calls; defaults to --profile.",
    )
    parser.add_argument(
        "--freshness-marker",
        help=(
            "Optional 1-64 character live-run marker returned by the Python cell; "
            "the runtime always verifies the supplied wheel SHA."
        ),
    )
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args()
    if args.verify_evidence is None and (args.wheel is None or args.output is None):
        parser.error("--wheel and --output are required unless --verify-evidence is used")
    if bool(args.template_repo) != bool(args.template_ref):
        parser.error("--template-repo and --template-ref must be provided together")
    return args


def main() -> int:
    args = parse_args()
    if args.verify_evidence:
        return verify_evidence(args.verify_evidence)
    runner = Runner(
        args.profile,
        args.output.resolve(),
        args.wheel.resolve(),
        args.template_repo,
        args.template_ref,
        args.app_auth_profile,
        args.freshness_marker,
    )
    succeeded = False
    try:
        runner.bootstrap()
        runner.select_warehouse(args.warehouse_id)
        runner.create_uc_resources(args.uc_schema)
        cases = runner.create_projects()
        for index, case in enumerate(cases):
            runner.run_dev(case, 8400 + index)
        for case in cases:
            runner.deploy(case)
        runner._write_evidence()
        succeeded = verify_evidence(runner.output / "evidence.json") == 0
        return 0 if succeeded else 1
    finally:
        if not args.keep_resources and succeeded:
            runner.cleanup()
        elif not succeeded:
            runner.transcript.write(
                "Resources retained after failure for diagnosis; rerun cleanup after fixing."
            )


if __name__ == "__main__":
    sys.exit(main())
