from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit


CATALOG = "supervisor_agent"
SCHEMA_PREFIX = "mason_genie_web_search_demo_"
VOLUME = "reports"

SourceKind = Literal["slack", "genie_asset", "official_web"]
Authority = Literal[
    "internal_field_report", "governed_asset_catalog", "official_documentation"
]

_AUTHORITY_BY_SOURCE: dict[SourceKind, Authority] = {
    "slack": "internal_field_report",
    "genie_asset": "governed_asset_catalog",
    "official_web": "official_documentation",
}


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_kind: SourceKind
    title: str
    canonical_uri: str
    excerpt: str
    retrieved_at: str
    source_metadata: dict[str, str]
    authority: Authority


@dataclass(frozen=True)
class VolumePaths:
    report: str
    evidence: str


def canonical_official_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("official documentation URL must be credential-free HTTPS")

    host = parsed.hostname.lower().rstrip(".")
    is_databricks_docs = host == "docs.databricks.com" or host.endswith(
        ".docs.databricks.com"
    )
    is_azure_docs = host == "learn.microsoft.com" and parsed.path.lower().startswith(
        "/en-us/azure/databricks/"
    )
    if not (is_databricks_docs or is_azure_docs):
        raise ValueError("unapproved host")

    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, parsed.query, ""))


def volume_paths(schema_name: str) -> VolumePaths:
    if not schema_name.startswith(SCHEMA_PREFIX):
        raise ValueError("report paths require a demo schema")
    root = f"/Volumes/{CATALOG}/{schema_name}/{VOLUME}"
    return VolumePaths(
        report=f"{root}/databricks_web_search_report.md",
        evidence=f"{root}/evidence.json",
    )


def validate_evidence(items: list[EvidenceItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.evidence_id in seen:
            raise ValueError(f"duplicate evidence id: {item.evidence_id}")
        seen.add(item.evidence_id)
        expected = _AUTHORITY_BY_SOURCE[item.source_kind]
        if item.authority != expected:
            raise ValueError(
                f"authority {item.authority!r} does not match {item.source_kind!r}"
            )
        if item.source_kind == "official_web":
            canonical_official_url(item.canonical_uri)
