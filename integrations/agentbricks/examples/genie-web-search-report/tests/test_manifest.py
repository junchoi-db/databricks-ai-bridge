from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_all_managed_tools_use_request_user_auth() -> None:
    manifest = tomllib.loads((ROOT / "agent.toml").read_text(encoding="utf-8"))
    tools = {item["id"]: item for item in manifest["tools"]}

    assert set(tools) == {"slack", "genie_assets", "web_search", "report_sandbox"}
    assert {item["auth"] for item in tools.values()} == {"user"}
    assert tools["slack"]["source"] == {"kind": "mcp", "service": "system.ai.slack"}
    assert tools["web_search"]["source"] == {
        "kind": "mcp",
        "service": "system.ai.web_search",
    }
    assert tools["genie_assets"]["source"]["kind"] == "genie_agent"
    downscope = tools["report_sandbox"]["policy"]["downscope"]
    assert downscope == [
        {
            "resource": "volume:supervisor_agent.not_deployed.reports",
            "permission": "read_write",
        }
    ]


def test_readme_documents_complete_df1_lifecycle() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for command in (
        "scripts/setup_df1.py",
        "scripts/deploy_df1.py",
        "scripts/invoke_e2e.py",
        "scripts/teardown_df1.py",
    ):
        assert command in readme
    for integration in (
        "system.ai.slack",
        "genie-agent",
        "system.ai.web_search",
        "system.ai.sandbox",
    ):
        assert integration in readme
    assert "1789056476.567349" in readme
    assert "ai-gateway" in readme
    assert "genie" in readme
    assert "files" in readme
    assert "leaves the successful resources deployed" in readme
