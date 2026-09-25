import json

from agent.tools import all_tools
from agent.tools.report_tools import EvidenceLedger, reset_ledger, validation_payload


def test_web_evidence_separates_rejected_domains() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    accepted, rejected = ledger.add_web_results(
        [
            {
                "title": "Official",
                "url": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "excerpt": "Managed MCP",
            },
            {
                "title": "Other",
                "url": "https://example.com/post",
                "excerpt": "Unofficial",
            },
        ]
    )

    assert [item.evidence_id for item in accepted] == ["web-001"]
    assert rejected == [
        {
            "title": "Other",
            "url": "https://example.com/post",
            "reason": "unapproved host",
        }
    ]


def test_report_rejects_unknown_evidence_ids() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    errors = ledger.validate_report("Finding [web-999]\n\n## Sources\n")

    assert "unknown evidence id: web-999" in errors


def test_report_requires_current_run_id() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    errors = ledger.validate_report("## Sources\n")

    assert "missing run_id: run-1" in errors


def test_evidence_metadata_removes_credentials_recursively() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    items = ledger.add_slack_results(
        [
            {
                "title": "Field report",
                "url": "https://databricks.slack.com/archives/C088VN8U4E5/p1790353973998359",
                "excerpt": "Observed behavior",
                "message_ts": "1789056476.567349",
                "metadata": {
                    "headers": {"authorization": "Bearer secret"},
                    "cookie": "secret",
                },
            }
        ]
    )

    serialized = json.dumps(items[0].source_metadata)
    assert "1789056476.567349" in serialized
    assert "secret" not in serialized
    assert "authorization" not in serialized
    assert "cookie" not in serialized


def test_reset_ledger_isolates_invocations() -> None:
    first = reset_ledger("run-1")
    second = reset_ledger("run-2")

    assert first is not second
    assert second.run_id == "run-2"


def test_report_tools_auto_register() -> None:
    tools = {tool.name: tool for tool in all_tools()}

    assert {
        "record_slack_evidence",
        "record_genie_assets",
        "record_web_evidence",
        "validate_report",
    } <= set(tools)
    assert set(tools["record_slack_evidence"].params_json_schema["required"]) == {
        "results_json",
        "thread_ts",
    }
    assert set(tools["record_genie_assets"].params_json_schema["required"]) == {
        "results_json",
        "conversation_id",
        "message_id",
    }


def test_genie_evidence_preserves_conversation_identifiers() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    items = ledger.add_genie_results(
        [
            {
                "title": "Asset",
                "url": "https://docs.databricks.com/aws/en/generative-ai/mcp",
                "description": "Managed MCP",
                "conversation_id": "conversation-1",
                "message_id": "message-1",
            }
        ]
    )

    assert items[0].source_metadata == {
        "conversation_id": "conversation-1",
        "message_id": "message-1",
    }


def test_validation_payload_returns_fixed_paths(monkeypatch) -> None:
    monkeypatch.setenv("REPORT_SCHEMA_NAME", "mason_genie_web_search_demo_ab12cd34")
    reset_ledger("run-1")
    markdown = "\n\n".join(
        [
            "run-1",
            "## Executive summary",
            "## Internal field report",
            "## Cataloged assets",
            "## Official documentation findings",
            "## Discrepancies and limitations",
            "## Sources",
        ]
    )

    payload = validation_payload(markdown)

    assert payload["errors"] == []
    assert payload["run_id"] == "run-1"
    assert payload["report_path"].endswith("/databricks_web_search_report.md")
    assert payload["evidence_path"].endswith("/evidence.json")
