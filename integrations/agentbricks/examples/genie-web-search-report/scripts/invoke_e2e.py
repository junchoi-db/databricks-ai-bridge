from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient

from agent.report_contract import canonical_official_url
from agent.tools.report_tools import REQUIRED_SECTIONS
from scripts.common import (
    EVIDENCE_DIR,
    RESULT_PATH,
    ROOT,
    SETUP_STATE_PATH,
    StateStore,
    redact,
)
from scripts.deploy_df1 import CLI, REQUIRED_SCOPES, assert_no_app_data_grants
from scripts.setup_df1 import EXPECTED_USER, PROFILE


REPORT_PROMPT = """Create the complete Databricks web search report now. You must read the fixed
Slack thread, query the configured native Genie asset catalog, validate with official web search,
and use sandbox to write and read back both governed Volume artifacts. Follow every developer
instruction and return only the final completion JSON after all checks pass."""
_REF = re.compile(r"\[((?:slack|genie|web)-\d{3})\]")


@dataclass(frozen=True)
class Verification:
    passed: bool
    errors: list[str]
    counts: dict[str, int]


def verify_report(markdown: str, evidence_text: str, *, run_id: str) -> Verification:
    errors: list[str] = []
    if run_id not in markdown:
        errors.append("report is missing run_id")
    for section in REQUIRED_SECTIONS:
        if section not in markdown:
            errors.append(f"missing section: {section}")

    try:
        payload = json.loads(evidence_text)
    except json.JSONDecodeError as error:
        return Verification(False, [f"invalid evidence JSON: {error}"], {})
    if payload.get("run_id") != run_id:
        errors.append("evidence run_id mismatch")
    if payload.get("errors"):
        errors.append(f"agent validation errors: {payload['errors']}")
    items = payload.get("evidence") or []
    counts: dict[str, int] = {}
    known: set[str] = set()
    genie_has_ids = False
    slack_has_thread = False
    for item in items:
        kind = str(item.get("source_kind"))
        counts[kind] = counts.get(kind, 0) + 1
        known.add(str(item.get("evidence_id")))
        metadata = item.get("source_metadata") or {}
        if kind == "slack" and "1789056476.567349" in json.dumps(metadata):
            slack_has_thread = True
        if kind == "genie_asset":
            serialized = json.dumps(metadata)
            genie_has_ids = (
                "conversation_id" in serialized and "message_id" in serialized
            )
        if kind == "official_web":
            try:
                canonical_official_url(str(item.get("canonical_uri") or ""))
            except ValueError as error:
                errors.append(f"invalid official citation: {error}")
    for kind in ("slack", "genie_asset", "official_web"):
        if not counts.get(kind):
            errors.append(f"missing source kind: {kind}")
    if counts.get("slack") and not slack_has_thread:
        errors.append("Slack evidence is missing the fixed thread timestamp")
    if counts.get("genie_asset") and not genie_has_ids:
        errors.append("Genie evidence is missing conversation_id/message_id")
    for reference in sorted(set(_REF.findall(markdown)) - known):
        errors.append(f"unknown evidence id: {reference}")
    return Verification(not errors, errors, counts)


def _invoke(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    body = {
        "id": run_id,
        "input": {
            "session_id": run_id,
            "messages": [{"role": "user", "content": REPORT_PROMPT}],
        },
        "background": True,
    }
    completed = subprocess.run(
        [
            str(CLI),
            "--profile",
            PROFILE,
            "-o",
            "json",
            "endpoint",
            "invoke",
            state["deployment"]["app_name"],
            "--path",
            "/api/invocations",
            "--timeout",
            "60",
            "--json",
            json.dumps(body),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=70,
        check=False,
    )
    StateStore(EVIDENCE_DIR / f"invoke-{run_id}.json").save(
        redact(
            {
                "arguments": completed.args,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    )
    if completed.returncode:
        raise RuntimeError(f"deployed invocation failed: {completed.stderr[-6000:]}")
    response = json.loads(completed.stdout)
    status = str(response.get("body", {}).get("status", "")).lower()
    if status == "completed":
        return response
    if status not in {"queued", "active"}:
        raise RuntimeError(f"unexpected background invocation response: {response}")

    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        time.sleep(15)
        polled = subprocess.run(
            [
                str(CLI),
                "--profile",
                PROFILE,
                "-o",
                "json",
                "endpoint",
                "invoke",
                state["deployment"]["app_name"],
                "--method",
                "GET",
                "--path",
                f"/api/invocations/{run_id}",
                "--timeout",
                "60",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=70,
            check=False,
        )
        if polled.returncode:
            raise RuntimeError(
                f"invocation status poll failed: {polled.stderr[-6000:]}"
            )
        response = json.loads(polled.stdout)
        status = str(response.get("body", {}).get("status", "")).lower()
        if status == "completed":
            return response
        if status == "failed":
            raise RuntimeError(f"background invocation failed: {response}")
    raise TimeoutError(f"background invocation {run_id} did not finish within 1200s")


def run_e2e() -> dict[str, Any]:
    state = StateStore(SETUP_STATE_PATH).load()
    if "deployment" not in state:
        raise RuntimeError("run deploy_df1.py before invoke_e2e.py")
    client = WorkspaceClient(profile=PROFILE)
    current = client.current_user.me()
    if current.user_name != EXPECTED_USER:
        raise RuntimeError(f"df1 caller mismatch: {current.user_name}")
    run_id = str(uuid4())
    started = datetime.now(UTC)
    response = _invoke(state, run_id)
    report = client.files.download(state["report_path"]).contents.read().decode("utf-8")
    evidence = (
        client.files.download(state["evidence_path"]).contents.read().decode("utf-8")
    )
    verification = verify_report(report, evidence, run_id=run_id)
    deployment = client.apps.get(state["deployment"]["app_name"])
    scopes = set(deployment.effective_user_api_scopes or [])
    if not REQUIRED_SCOPES <= scopes:
        verification.errors.append(
            f"missing effective App scopes: {sorted(REQUIRED_SCOPES - scopes)}"
        )
    negative_grants = assert_no_app_data_grants(
        client, state, state["deployment"]["service_principal_client_id"]
    )
    result = {
        "passed": not verification.errors,
        "run_id": run_id,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "app_name": state["deployment"]["app_name"],
        "app_url": state["deployment"]["app_url"],
        "report_path": state["report_path"],
        "evidence_path": state["evidence_path"],
        "verification": asdict(verification),
        "effective_scopes": sorted(scopes),
        "negative_data_grants": negative_grants,
        "invocation_response": redact(response),
    }
    result["verification"]["passed"] = result["passed"]
    StateStore(RESULT_PATH).save(result)
    if not result["passed"]:
        raise RuntimeError(f"E2E verification failed: {verification.errors}")
    return result


def main() -> None:
    print(json.dumps(run_e2e(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
