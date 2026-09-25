from unittest.mock import MagicMock

from agent.agent import INSTRUCTIONS, MAX_TURNS, SLACK_THREAD_URL, create_agent


def test_prompt_fixes_source_order_and_slack_thread() -> None:
    assert SLACK_THREAD_URL in INSTRUCTIONS
    assert INSTRUCTIONS.index("Slack") < INSTRUCTIONS.index("Genie")
    assert INSTRUCTIONS.index("Genie") < INSTRUCTIONS.index("web search")


def test_prompt_forbids_blind_genie_resubmission() -> None:
    lowered = INSTRUCTIONS.lower()

    assert "same conversation_id and message_id" in INSTRUCTIONS
    assert "never resubmit" in lowered
    assert "indeterminate_submission" in lowered


def test_prompt_requires_sandbox_artifacts() -> None:
    assert "databricks_web_search_report.md" in INSTRUCTIONS
    assert "evidence.json" in INSTRUCTIONS
    assert "read both files back" in INSTRUCTIONS
    assert MAX_TURNS == 30


def test_prompt_separates_field_report_from_official_docs() -> None:
    assert "internal field report" in INSTRUCTIONS.lower()
    assert "official documentation" in INSTRUCTIONS.lower()
    assert "allowed_domains is not a security boundary" in INSTRUCTIONS


def test_create_agent_uses_protocol_instructions(monkeypatch) -> None:
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "all_tools", lambda: [])
    monkeypatch.setattr(agent_module, "memory_tools", lambda _actor: [])
    monkeypatch.setattr(agent_module, "genie_tools", lambda **_kwargs: [])
    constructed = MagicMock(return_value=object())
    monkeypatch.setattr(agent_module, "Agent", constructed)

    result = create_agent("actor-1")

    assert result is constructed.return_value
    assert (
        constructed.call_args.kwargs["name"] == "Databricks documentation report agent"
    )
    assert constructed.call_args.kwargs["instructions"] == INSTRUCTIONS
