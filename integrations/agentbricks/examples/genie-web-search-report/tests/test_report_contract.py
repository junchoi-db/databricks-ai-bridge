import pytest

from agent.report_contract import (
    EvidenceItem,
    canonical_official_url,
    validate_evidence,
    volume_paths,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://docs.databricks.com/aws/en/generative-ai/mcp/",
            "https://docs.databricks.com/aws/en/generative-ai/mcp",
        ),
        (
            "https://sub.docs.databricks.com/page",
            "https://sub.docs.databricks.com/page",
        ),
        (
            "https://learn.microsoft.com/en-us/azure/databricks/generative-ai/",
            "https://learn.microsoft.com/en-us/azure/databricks/generative-ai",
        ),
    ],
)
def test_accepts_official_documentation(url: str, expected: str) -> None:
    assert canonical_official_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.databricks.com/page",
        "https://docs.databricks.com.evil.example/page",
        "https://learn.microsoft.com/en-us/windows/page",
        "https://user:password@docs.databricks.com/page",
        "not-a-url",
    ],
)
def test_rejects_non_official_or_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        canonical_official_url(url)


def test_volume_paths_are_fixed_to_demo_schema() -> None:
    paths = volume_paths("mason_genie_web_search_demo_ab12cd34")

    assert paths.report == (
        "/Volumes/supervisor_agent/mason_genie_web_search_demo_ab12cd34/"
        "reports/databricks_web_search_report.md"
    )
    assert paths.evidence == (
        "/Volumes/supervisor_agent/mason_genie_web_search_demo_ab12cd34/reports/evidence.json"
    )


def test_volume_paths_reject_non_demo_schema() -> None:
    with pytest.raises(ValueError, match="demo schema"):
        volume_paths("other_schema")


def test_validate_evidence_requires_matching_source_authority() -> None:
    item = EvidenceItem(
        evidence_id="web-001",
        source_kind="official_web",
        title="Managed MCP",
        canonical_uri="https://docs.databricks.com/aws/en/generative-ai/mcp",
        excerpt="Official documentation",
        retrieved_at="2026-09-25T00:00:00+00:00",
        source_metadata={"host": "docs.databricks.com"},
        authority="internal_field_report",
    )

    with pytest.raises(ValueError, match="authority"):
        validate_evidence([item])
