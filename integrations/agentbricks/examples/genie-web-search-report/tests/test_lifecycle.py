import json
import subprocess

import scripts.invoke_e2e as invoke_e2e
from scripts.deploy_df1 import DEPLOY_SOURCE, ROOT, render_app_yaml, render_manifest
from scripts.invoke_e2e import _invoke, verify_report


def _state() -> dict:
    return {
        "suffix": "ab12cd34",
        "schema_name": "mason_genie_web_search_demo_ab12cd34",
        "schema": "supervisor_agent.mason_genie_web_search_demo_ab12cd34",
        "volume": "supervisor_agent.mason_genie_web_search_demo_ab12cd34.reports",
        "space_id": "a" * 32,
    }


def test_deploy_source_is_outside_gitignored_project_state() -> None:
    assert ROOT not in DEPLOY_SOURCE.parents


def test_invoke_uses_background_transport(monkeypatch, tmp_path) -> None:
    requests: list[dict] = []

    def fake_run(arguments, **_kwargs):
        requests.append(json.loads(arguments[arguments.index("--json") + 1]))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps({"body": {"status": "completed"}}),
            stderr="",
        )

    monkeypatch.setattr(invoke_e2e, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(invoke_e2e.subprocess, "run", fake_run)

    _invoke({"deployment": {"app_name": "agent-bricks-demo"}}, "run-1")

    assert requests[0]["background"] is True


def test_render_manifest_injects_exact_space_and_volume() -> None:
    manifest = render_manifest(_state())

    assert "a" * 32 in manifest
    assert (
        "volume:supervisor_agent.mason_genie_web_search_demo_ab12cd34.reports"
        in manifest
    )
    assert manifest.count('auth = "user"') == 4
    assert "00000000000000000000000000000000" not in manifest
    assert "supervisor_agent.not_deployed.reports" not in manifest


def test_render_app_yaml_injects_report_schema() -> None:
    rendered = render_app_yaml(_state())

    assert "REPORT_SCHEMA_NAME" in rendered
    assert "mason_genie_web_search_demo_ab12cd34" in rendered
    assert "MLFLOW_DISABLE_AGENT_HINT" in rendered


def test_verify_report_requires_all_source_kinds() -> None:
    markdown = "\n".join(
        [
            "run-1",
            "## Executive summary",
            "## Internal field report",
            "## Cataloged assets",
            "## Official documentation findings",
            "## Discrepancies and limitations",
            "## Sources",
            "Official [web-001]",
        ]
    )
    evidence = {
        "run_id": "run-1",
        "errors": [],
        "evidence": [
            {
                "evidence_id": "web-001",
                "source_kind": "official_web",
                "canonical_uri": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "source_metadata": {"host": "docs.databricks.com"},
            },
            {
                "evidence_id": "genie-001",
                "source_kind": "genie_asset",
                "canonical_uri": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "source_metadata": {"conversation_id": "c-1", "message_id": "m-1"},
            },
        ],
    }

    result = verify_report(markdown, json.dumps(evidence), run_id="run-1")

    assert result.passed is False
    assert "missing source kind: slack" in result.errors


def test_verify_report_accepts_complete_artifacts() -> None:
    markdown = "\n".join(
        [
            "run-1",
            "## Executive summary",
            "## Internal field report [slack-001]",
            "## Cataloged assets [genie-001]",
            "## Official documentation findings [web-001]",
            "## Discrepancies and limitations",
            "## Sources",
        ]
    )
    evidence = {
        "run_id": "run-1",
        "errors": [],
        "evidence": [
            {
                "evidence_id": "slack-001",
                "source_kind": "slack",
                "canonical_uri": "https://databricks.slack.com/archives/C088VN8U4E5/p1790353973998359",
                "source_metadata": {"thread_ts": "1789056476.567349"},
            },
            {
                "evidence_id": "genie-001",
                "source_kind": "genie_asset",
                "canonical_uri": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "source_metadata": {"conversation_id": "c-1", "message_id": "m-1"},
            },
            {
                "evidence_id": "web-001",
                "source_kind": "official_web",
                "canonical_uri": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "source_metadata": {"host": "docs.databricks.com"},
            },
        ],
    }

    result = verify_report(markdown, json.dumps(evidence), run_id="run-1")

    assert result.passed is True
    assert result.errors == []
