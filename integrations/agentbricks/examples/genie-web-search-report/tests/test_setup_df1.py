import json
import re
import stat
from dataclasses import asdict
from types import SimpleNamespace

import scripts.setup_df1 as setup_df1
from scripts.common import StateStore
from scripts.setup_df1 import asset_rows, build_serialized_space, resource_names, setup


def test_resource_names_share_one_suffix() -> None:
    names = resource_names("ab12cd34")

    assert names.schema == "supervisor_agent.mason_genie_web_search_demo_ab12cd34"
    assert names.table == names.schema + ".web_search_assets"
    assert names.volume == names.schema + ".reports"
    assert names.report_path.endswith("/reports/databricks_web_search_report.md")
    assert names.evidence_path.endswith("/reports/evidence.json")
    assert names.app == "genie-ab12cd34"
    assert len(f"agent-bricks-{names.app}") <= 30


def test_resource_names_reject_unsafe_suffix() -> None:
    try:
        resource_names("../unsafe")
    except ValueError as error:
        assert "8 lowercase hexadecimal" in str(error)
    else:
        raise AssertionError("unsafe suffix was accepted")


def test_seed_assets_are_official_and_cover_required_topics() -> None:
    rows = asset_rows()

    assert {row["topic"] for row in rows} >= {
        "web_search",
        "managed_mcp",
        "obo",
        "audit",
        "apps_oauth",
    }
    assert all(row["url"].startswith("https://docs.databricks.com/") for row in rows)
    assert len({row["asset_id"] for row in rows}) == len(rows)


def test_genie_space_uses_only_asset_table() -> None:
    payload = build_serialized_space("supervisor_agent.schema.web_search_assets")

    assert payload["data_sources"] == {
        "tables": [{"identifier": "supervisor_agent.schema.web_search_assets"}]
    }
    instructions = payload["instructions"]["text_instructions"]
    assert len(instructions) == 1
    assert re.fullmatch(r"[0-9a-f]{32}", instructions[0]["id"])
    assert "web_search_assets" in instructions[0]["content"][0]


def test_state_store_writes_private_atomic_json(tmp_path) -> None:
    path = tmp_path / "nested" / "setup-state.json"
    store = StateStore(path)

    store.save({"suffix": "ab12cd34", "resources": []})

    assert json.loads(path.read_text()) == {"suffix": "ab12cd34", "resources": []}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(path.parent.glob("*.tmp")) == []


def test_setup_resumes_missing_genie_space(monkeypatch, tmp_path) -> None:
    names = resource_names("ab12cd34")
    state_path = tmp_path / "setup-state.json"
    StateStore(state_path).save(
        {
            **asdict(names),
            "workspace_host": "https://df1.example.com",
            "caller": setup_df1.EXPECTED_USER,
            "resources": [
                {"kind": "schema", "name": names.schema},
                {"kind": "table", "name": names.table},
                {"kind": "volume", "name": names.volume},
            ],
            "seed_control": {"status": {"state": "SUCCEEDED"}},
        }
    )

    created: list[dict] = []
    client = SimpleNamespace(
        config=SimpleNamespace(host="https://df1.example.com"),
        current_user=SimpleNamespace(
            me=lambda: SimpleNamespace(user_name=setup_df1.EXPECTED_USER)
        ),
        genie=SimpleNamespace(
            create_space=lambda **kwargs: (
                created.append(kwargs) or SimpleNamespace(space_id="a" * 32)
            )
        ),
    )
    monkeypatch.setattr(setup_df1, "SETUP_STATE_PATH", state_path)
    monkeypatch.setattr(setup_df1, "WorkspaceClient", lambda profile: client)

    resumed = setup()

    assert resumed["space_id"] == "a" * 32
    assert created[0]["title"] == names.genie_title
