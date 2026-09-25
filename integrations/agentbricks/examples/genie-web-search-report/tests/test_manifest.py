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
