from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from agents import function_tool

from agent.report_contract import (
    EvidenceItem,
    canonical_official_url,
    validate_evidence,
    volume_paths,
)


REQUIRED_SECTIONS = (
    "## Executive summary",
    "## Internal field report",
    "## Cataloged assets",
    "## Official documentation findings",
    "## Discrepancies and limitations",
    "## Sources",
)
_EVIDENCE_REF = re.compile(r"\[((?:slack|genie|web)-\d{3})\]")
_SENSITIVE_KEYS = {"authorization", "cookie", "headers", "token"}
_current_ledger: ContextVar[EvidenceLedger | None] = ContextVar(
    "report_evidence_ledger", default=None
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if str(key).lower() in _SENSITIVE_KEYS:
            continue
        if isinstance(item, (dict, list)):
            nested = _strip_sensitive(item)
            result[str(key)] = json.dumps(nested, sort_keys=True)
        elif item is not None:
            result[str(key)] = str(item)
    return result


def _strip_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_sensitive(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    return value


class EvidenceLedger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.items: list[EvidenceItem] = []
        self.rejected: list[dict[str, str]] = []

    def _add(
        self,
        *,
        prefix: str,
        source_kind: str,
        authority: str,
        title: str,
        uri: str,
        excerpt: str,
        metadata: Any,
    ) -> EvidenceItem:
        count = (
            sum(item.evidence_id.startswith(f"{prefix}-") for item in self.items) + 1
        )
        evidence = EvidenceItem(
            evidence_id=f"{prefix}-{count:03d}",
            source_kind=source_kind,  # type: ignore[arg-type]
            title=title.strip()[:300],
            canonical_uri=uri,
            excerpt=excerpt.strip()[:2000],
            retrieved_at=_now(),
            source_metadata=_safe_metadata(metadata),
            authority=authority,  # type: ignore[arg-type]
        )
        validate_evidence([evidence])
        self.items.append(evidence)
        return evidence

    def add_slack_results(self, results: list[dict[str, Any]]) -> list[EvidenceItem]:
        return [
            self._add(
                prefix="slack",
                source_kind="slack",
                authority="internal_field_report",
                title=str(result.get("title") or "Slack field report"),
                uri=str(result["url"]),
                excerpt=str(result["excerpt"]),
                metadata=result.get("metadata"),
            )
            for result in results
        ]

    def add_genie_results(self, results: list[dict[str, Any]]) -> list[EvidenceItem]:
        return [
            self._add(
                prefix="genie",
                source_kind="genie_asset",
                authority="governed_asset_catalog",
                title=str(result["title"]),
                uri=str(result["url"]),
                excerpt=str(result.get("excerpt") or result.get("description") or ""),
                metadata=result.get("metadata"),
            )
            for result in results
        ]

    def add_web_results(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[EvidenceItem], list[dict[str, str]]]:
        accepted: list[EvidenceItem] = []
        rejected: list[dict[str, str]] = []
        for result in results:
            title = str(result.get("title") or "Untitled result")
            url = str(result.get("url") or "")
            try:
                canonical = canonical_official_url(url)
            except ValueError as error:
                rejection = {"title": title, "url": url, "reason": str(error)}
                rejected.append(rejection)
                self.rejected.append(rejection)
                continue
            accepted.append(
                self._add(
                    prefix="web",
                    source_kind="official_web",
                    authority="official_documentation",
                    title=title,
                    uri=canonical,
                    excerpt=str(result.get("excerpt") or result.get("snippet") or ""),
                    metadata={
                        **(result.get("metadata") or {}),
                        "host": canonical.split("/", 3)[2],
                    },
                )
            )
        return accepted, rejected

    def validate_report(self, markdown: str) -> list[str]:
        errors = [
            f"missing section: {section}"
            for section in REQUIRED_SECTIONS
            if section not in markdown
        ]
        known = {item.evidence_id for item in self.items}
        for evidence_id in sorted(set(_EVIDENCE_REF.findall(markdown)) - known):
            errors.append(f"unknown evidence id: {evidence_id}")
        return errors

    def as_dict(self) -> dict[str, Any]:
        validate_evidence(self.items)
        return {
            "run_id": self.run_id,
            "evidence": [asdict(item) for item in self.items],
            "rejected": list(self.rejected),
        }


def reset_ledger(run_id: str) -> EvidenceLedger:
    ledger = EvidenceLedger(run_id)
    _current_ledger.set(ledger)
    return ledger


def current_ledger() -> EvidenceLedger:
    ledger = _current_ledger.get()
    if ledger is None:
        raise RuntimeError("report evidence ledger is not initialized")
    return ledger


def _decode_results(results_json: str) -> list[dict[str, Any]]:
    value = json.loads(results_json)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("results_json must encode a list of objects")
    return value


def validation_payload(markdown: str) -> dict[str, Any]:
    ledger = current_ledger()
    paths = volume_paths(os.environ["REPORT_SCHEMA_NAME"])
    return {
        "errors": ledger.validate_report(markdown),
        **ledger.as_dict(),
        "report_path": paths.report,
        "evidence_path": paths.evidence,
    }


@function_tool
def record_slack_evidence(results_json: str) -> str:
    """Record normalized observations from the fixed Slack thread as internal field evidence."""
    items = current_ledger().add_slack_results(_decode_results(results_json))
    return json.dumps([asdict(item) for item in items], sort_keys=True)


@function_tool
def record_genie_assets(results_json: str) -> str:
    """Record normalized asset rows returned by the configured native Genie Agent."""
    items = current_ledger().add_genie_results(_decode_results(results_json))
    return json.dumps([asdict(item) for item in items], sort_keys=True)


@function_tool
def record_web_evidence(results_json: str) -> str:
    """Validate and record official documentation results; reject every unapproved URL."""
    accepted, rejected = current_ledger().add_web_results(_decode_results(results_json))
    return json.dumps(
        {"accepted": [asdict(item) for item in accepted], "rejected": rejected},
        sort_keys=True,
    )


@function_tool
def validate_report(markdown: str) -> str:
    """Validate required report sections and every inline evidence reference before writing."""
    return json.dumps(validation_payload(markdown), sort_keys=True)
