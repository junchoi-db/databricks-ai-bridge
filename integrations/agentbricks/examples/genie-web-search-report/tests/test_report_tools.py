import json

from agent.tools import all_tools
from agent.tools.report_tools import EvidenceLedger, reset_ledger


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


def test_evidence_metadata_removes_credentials_recursively() -> None:
    ledger = EvidenceLedger(run_id="run-1")

    items = ledger.add_slack_results(
        [
            {
                "title": "Field report",
                "url": "https://databricks.slack.com/archives/C088VN8U4E5/p1790353973998359",
                "excerpt": "Observed behavior",
                "metadata": {
                    "thread_ts": "1789056476.567349",
                    "headers": {"authorization": "Bearer secret"},
                    "cookie": "secret",
                },
            }
        ]
    )

    serialized = json.dumps(items[0].source_metadata)
    assert "thread_ts" in serialized
    assert "secret" not in serialized
    assert "authorization" not in serialized
    assert "cookie" not in serialized


def test_reset_ledger_isolates_invocations() -> None:
    first = reset_ledger("run-1")
    second = reset_ledger("run-2")

    assert first is not second
    assert second.run_id == "run-2"


def test_report_tools_auto_register() -> None:
    names = {tool.name for tool in all_tools()}

    assert {
        "record_slack_evidence",
        "record_genie_assets",
        "record_web_evidence",
        "validate_report",
    } <= names
