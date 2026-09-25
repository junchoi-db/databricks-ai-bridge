from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient

from scripts.common import EVIDENCE_DIR, ROOT, SETUP_STATE_PATH, StateStore, redact
from scripts.setup_df1 import EXPECTED_USER, PROFILE


REQUIRED_SCOPES = {"ai-gateway", "files", "genie"}
DEPLOY_SOURCE = Path(tempfile.gettempdir()) / "agentbricks-genie-report-deploy"
CLI = ROOT.parents[1] / ".venv" / "bin" / "ab"


def render_manifest(state: dict[str, Any]) -> str:
    manifest = (ROOT / "agent.toml").read_text(encoding="utf-8")
    manifest = manifest.replace("0" * 32, state["space_id"])
    manifest = manifest.replace(
        "supervisor_agent.not_deployed.reports",
        state["volume"],
    )
    if "0" * 32 in manifest or "supervisor_agent.not_deployed.reports" in manifest:
        raise RuntimeError("deployment manifest still contains sentinel resources")
    return manifest


def render_app_yaml(state: dict[str, Any]) -> str:
    base = (ROOT / "app.yaml").read_text(encoding="utf-8").rstrip()
    return (
        f"{base}\nenv:\n"
        "  - name: REPORT_SCHEMA_NAME\n"
        f"    value: {state['schema_name']}\n"
        "  - name: MLFLOW_DISABLE_AGENT_HINT\n"
        "    value: '1'\n"
    )


def prepare_deploy_source(state: dict[str, Any]) -> Path:
    if DEPLOY_SOURCE.exists():
        shutil.rmtree(DEPLOY_SOURCE)
    shutil.copytree(
        ROOT,
        DEPLOY_SOURCE,
        ignore=shutil.ignore_patterns(
            ".demo-state",
            ".env",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "*.pyc",
        ),
    )
    (DEPLOY_SOURCE / "agent.toml").write_text(render_manifest(state), encoding="utf-8")
    (DEPLOY_SOURCE / "app.yaml").write_text(render_app_yaml(state), encoding="utf-8")
    return DEPLOY_SOURCE


def _record_command(name: str, completed: subprocess.CompletedProcess[str]) -> None:
    StateStore(EVIDENCE_DIR / f"{name}.json").save(
        redact(
            {
                "arguments": completed.args,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    )


def _run(arguments: list[str], *, cwd: Path, timeout: int = 1800) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    _record_command(
        f"command-{datetime.now(UTC):%Y%m%dT%H%M%S}-{arguments[-1]}", completed
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n{completed.stderr[-4000:]}"
        )
    return completed.stdout


def _privileges(payload: dict[str, Any]) -> set[str]:
    return {
        str(item.get("privilege", "")).upper()
        for assignment in payload.get("privilege_assignments", [])
        for item in assignment.get("privileges", [])
    }


def assert_no_app_data_grants(
    client: WorkspaceClient, state: dict[str, Any], principal: str
) -> dict[str, Any]:
    table = client.grants.get_effective(
        "table", state["table"], principal=principal
    ).as_dict()
    volume = client.grants.get_effective(
        "volume", state["volume"], principal=principal
    ).as_dict()
    forbidden_table = {"SELECT", "OWN", "ALL_PRIVILEGES", "ALL PRIVILEGES"}
    forbidden_volume = {
        "READ_VOLUME",
        "WRITE_VOLUME",
        "OWN",
        "ALL_PRIVILEGES",
        "ALL PRIVILEGES",
    }
    if _privileges(table) & forbidden_table:
        raise RuntimeError(
            "App service principal unexpectedly has table data privileges"
        )
    if _privileges(volume) & forbidden_volume:
        raise RuntimeError(
            "App service principal unexpectedly has Volume data privileges"
        )
    return {"table": table, "volume": volume}


def deploy() -> dict[str, Any]:
    store = StateStore(SETUP_STATE_PATH)
    state = store.load()
    source = prepare_deploy_source(state)
    if not CLI.exists():
        raise RuntimeError(f"Agent Bricks CLI not found at {CLI}")

    _run(
        [
            str(CLI),
            "--profile",
            PROFILE,
            "-o",
            "json",
            "deploy",
            state["app"],
            "--source",
            str(source),
            "--allow-user-scope-update",
        ],
        cwd=ROOT,
    )

    client = WorkspaceClient(profile=PROFILE)
    current = client.current_user.me()
    if current.user_name != EXPECTED_USER:
        raise RuntimeError(f"df1 caller mismatch: {current.user_name}")
    app_name = f"agent-bricks-{state['app']}"
    deadline = time.monotonic() + 900
    while True:
        app = client.apps.get(app_name)
        payload = app.as_dict()
        compute_state = str(payload.get("compute_status", {}).get("state", ""))
        if compute_state == "ACTIVE" and payload.get("url"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"App {app_name} did not become ACTIVE: {compute_state}")
        time.sleep(15)

    configured = set(app.user_api_scopes or [])
    effective = set(app.effective_user_api_scopes or [])
    if not REQUIRED_SCOPES <= configured or not REQUIRED_SCOPES <= effective:
        raise RuntimeError(
            f"App scopes did not converge: configured={sorted(configured)}, effective={sorted(effective)}"
        )
    if app.forward_user_access_token is False:
        raise RuntimeError("App has user-token forwarding disabled")
    principal = app.service_principal_client_id
    if not principal:
        raise RuntimeError("deployed App has no service principal client ID")
    negative_grants = assert_no_app_data_grants(client, state, principal)

    state["deployment"] = {
        "app_name": app_name,
        "app_url": app.url,
        "service_principal_client_id": principal,
        "service_principal_id": app.service_principal_id,
        "configured_scopes": sorted(configured),
        "effective_scopes": sorted(effective),
        "source_commit": "81c427266af394216f327fb5c0159db332f1ae64",
        "deployed_at": datetime.now(UTC).isoformat(),
        "negative_data_grants": negative_grants,
    }
    store.save(state)
    return state


def main() -> None:
    print(json.dumps(deploy()["deployment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
