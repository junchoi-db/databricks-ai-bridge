from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

from scripts.common import SETUP_STATE_PATH, StateStore


PROFILE = "df1"
EXPECTED_USER = "jun.choi@databricks.com"
CATALOG = "supervisor_agent"
SCHEMA_PREFIX = "mason_genie_web_search_demo_"
WAREHOUSE_ID = "59d2ebcf58480621"
_SUFFIX = re.compile(r"^[0-9a-f]{8}$")


@dataclass(frozen=True)
class ResourceNames:
    suffix: str
    schema_name: str
    schema: str
    table: str
    volume: str
    report_path: str
    evidence_path: str
    app: str
    genie_title: str


def resource_names(suffix: str) -> ResourceNames:
    if not _SUFFIX.fullmatch(suffix):
        raise ValueError("resource suffix must be 8 lowercase hexadecimal characters")
    schema_name = f"{SCHEMA_PREFIX}{suffix}"
    schema = f"{CATALOG}.{schema_name}"
    volume_root = f"/Volumes/{CATALOG}/{schema_name}/reports"
    return ResourceNames(
        suffix=suffix,
        schema_name=schema_name,
        schema=schema,
        table=f"{schema}.web_search_assets",
        volume=f"{schema}.reports",
        report_path=f"{volume_root}/databricks_web_search_report.md",
        evidence_path=f"{volume_root}/evidence.json",
        app=f"mason-genie-report-{suffix}",
        genie_title=f"Mason web search docs {suffix}",
    )


def asset_rows() -> list[dict[str, Any]]:
    return [
        {
            "asset_id": "docs-web-search",
            "topic": "web_search",
            "title": "Web search tool for agents",
            "url": "https://docs.databricks.com/aws/en/generative-ai/agent-framework/web-search",
            "description": "Official guidance for using governed web search from Databricks agents.",
            "keywords": ["web search", "allowed domains", "citations"],
        },
        {
            "asset_id": "docs-managed-mcp",
            "topic": "managed_mcp",
            "title": "Databricks managed MCP servers",
            "url": "https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp",
            "description": "Official managed MCP server concepts and supported services.",
            "keywords": ["managed MCP", "AI Gateway", "tools"],
        },
        {
            "asset_id": "docs-obo",
            "topic": "obo",
            "title": "Authorize access to Databricks resources",
            "url": "https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m",
            "description": "Official OAuth and delegated authorization background.",
            "keywords": ["OAuth", "on behalf of", "user authorization"],
        },
        {
            "asset_id": "docs-inference-tables",
            "topic": "audit",
            "title": "AI Gateway-enabled inference tables",
            "url": "https://docs.databricks.com/aws/en/ai-gateway/inference-tables",
            "description": "Official guidance for logging and auditing AI Gateway inference traffic.",
            "keywords": ["inference tables", "audit", "AI Gateway"],
        },
        {
            "asset_id": "docs-app-auth",
            "topic": "apps_oauth",
            "title": "Databricks Apps authorization",
            "url": "https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth",
            "description": "Official Databricks Apps user authorization and resource access guidance.",
            "keywords": ["Databricks Apps", "OAuth scopes", "user token forwarding"],
        },
    ]


def build_serialized_space(table: str) -> dict[str, Any]:
    return {
        "version": 2,
        "data_sources": {"tables": [{"identifier": table}]},
        "instructions": {
            "text_instructions": [
                {
                    "id": "web_search_asset_catalog",
                    "content": [
                        "Use only the web_search_assets table. Return asset_id, topic, title, url, "
                        "description, and keywords for assets relevant to the user's question."
                    ],
                }
            ]
        },
    }


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _insert_statement(table: str, rows: list[dict[str, Any]]) -> str:
    values = []
    for row in rows:
        keywords = ", ".join(_sql_string(item) for item in row["keywords"])
        values.append(
            "("
            + ", ".join(
                [
                    _sql_string(row["asset_id"]),
                    _sql_string(row["topic"]),
                    _sql_string(row["title"]),
                    _sql_string(row["url"]),
                    _sql_string(row["description"]),
                    f"array({keywords})",
                ]
            )
            + ")"
        )
    return f"INSERT INTO {table} VALUES\n" + ",\n".join(values)


def _execute_sql(client: WorkspaceClient, statement: str) -> dict[str, Any]:
    response = client.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        wait_timeout="50s",
    ).as_dict()
    state = response.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"SQL failed ({state}): {response.get('status')}")
    return response


def setup() -> dict[str, Any]:
    store = StateStore(SETUP_STATE_PATH)
    client = WorkspaceClient(profile=PROFILE)
    current = client.current_user.me()
    if current.user_name != EXPECTED_USER:
        raise RuntimeError(f"df1 caller mismatch: {current.user_name}")

    if SETUP_STATE_PATH.exists():
        state = store.load()
        if (
            state.get("workspace_host") != client.config.host
            or state.get("caller") != EXPECTED_USER
        ):
            raise RuntimeError(
                "existing setup state belongs to another workspace or principal"
            )
        return state

    names = resource_names(uuid4().hex[:8])
    state: dict[str, Any] = {
        **asdict(names),
        "created_at": datetime.now(UTC).isoformat(),
        "profile": PROFILE,
        "caller": EXPECTED_USER,
        "caller_id": current.id,
        "workspace_host": client.config.host,
        "warehouse_id": WAREHOUSE_ID,
        "resources": [],
    }
    store.save(state)

    client.schemas.create(
        name=names.schema_name,
        catalog_name=CATALOG,
        comment="Mason Genie web search report demo",
    )
    state["resources"].append({"kind": "schema", "name": names.schema})
    store.save(state)

    _execute_sql(
        client,
        f"""CREATE TABLE {names.table} (
asset_id STRING,
topic STRING,
title STRING,
url STRING,
description STRING,
keywords ARRAY<STRING>
) USING DELTA""",
    )
    state["resources"].append({"kind": "table", "name": names.table})
    store.save(state)
    _execute_sql(client, _insert_statement(names.table, asset_rows()))
    control = _execute_sql(client, f"SELECT count(*) AS asset_count FROM {names.table}")
    if control.get("result", {}).get("data_array") != [[str(len(asset_rows()))]]:
        raise RuntimeError("seeded asset row count did not match")
    state["seed_control"] = control
    store.save(state)

    client.volumes.create(
        catalog_name=CATALOG,
        schema_name=names.schema_name,
        name="reports",
        volume_type=VolumeType.MANAGED,
        comment="Mason Genie report artifacts",
    )
    state["resources"].append({"kind": "volume", "name": names.volume})
    store.save(state)

    space = client.genie.create_space(
        warehouse_id=WAREHOUSE_ID,
        title=names.genie_title,
        description="Curated official Databricks documentation asset catalog",
        serialized_space=json.dumps(build_serialized_space(names.table)),
    )
    state["space_id"] = space.space_id
    state["resources"].append({"kind": "genie", "name": space.space_id})
    store.save(state)
    return state


def main() -> None:
    print(json.dumps(setup(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
