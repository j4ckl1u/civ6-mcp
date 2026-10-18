from types import SimpleNamespace

import pytest

from scripts.auto_play import (
    _allowed_tools,
    _compact_history,
    _ensure_mcp_port_available,
    _ensure_ollama_model,
    _parse_arguments,
    _single_tool_call_message,
    _tool_definition,
    _turn_advance,
)


def test_tool_definition_uses_mcp_schema():
    tool = SimpleNamespace(
        name="unit_action",
        description="Move a unit.",
        inputSchema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"Unused": {"type": "string"}},
            "type": "object",
            "properties": {"unit_id": {"type": "integer"}},
            "required": ["unit_id"],
        },
    )

    definition = _tool_definition(tool)

    assert definition["function"]["name"] == "unit_action"
    assert definition["function"]["parameters"] == {
        "type": "object",
        "properties": {"unit_id": {"type": "integer"}},
        "required": ["unit_id"],
    }


def test_dangerous_game_lifecycle_tools_are_not_exposed():
    tools = [
        SimpleNamespace(name="get_units"),
        SimpleNamespace(name="load_game_save"),
        SimpleNamespace(name="run_lua"),
        SimpleNamespace(name="delete_unit"),
    ]

    assert [tool.name for tool in _allowed_tools(tools)] == ["get_units"]


def test_mcp_port_check_reports_existing_server(monkeypatch):
    class ConnectedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "scripts.auto_play.socket.create_connection",
        lambda *_args, **_kwargs: ConnectedSocket(),
    )

    with pytest.raises(RuntimeError, match="only one active MCP connection"):
        _ensure_mcp_port_available(8000)


def test_ollama_check_reports_missing_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"models":[{"name":"other:latest"}]}'

    monkeypatch.setattr("scripts.auto_play.urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="ollama pull qwen7b-civ6:latest"):
        _ensure_ollama_model("http://localhost:11434", "qwen7b-civ6:latest", 2)


def test_compact_history_keeps_system_prompt_and_complete_tool_turn():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "old"},
        {"role": "tool", "tool_name": "old_tool", "content": "old result"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "recent"}}]},
        {"role": "tool", "tool_name": "recent", "content": "recent result"},
    ]

    compacted = _compact_history(messages, max_messages=3)

    assert compacted[0] == messages[0]
    assert "shortened" in compacted[1]["content"]
    assert compacted[2:] == messages[4:]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Turn 12 -> 13\nAI moved.", 1),
        ("Turn 13 -> 13\nBlocked.", 0),
        ("GAME OVER — VICTORY!", 0),
    ],
)
def test_turn_advance_detects_only_increment(text, expected):
    assert _turn_advance(text) == expected


def test_parse_arguments_accepts_json_and_rejects_non_objects():
    assert _parse_arguments('{"unit_id": 4}', "unit_action") == {"unit_id": 4}

    with pytest.raises(RuntimeError, match="must be an object"):
        _parse_arguments("[]", "unit_action")


def test_single_tool_call_message_defers_remaining_calls():
    first_call = {"id": "call-1", "function": {"name": "get_units"}}
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            first_call,
            {"id": "call-2", "function": {"name": "end_turn"}},
        ],
    }

    single = _single_tool_call_message(message)

    assert single["tool_calls"] == [first_call]
    assert len(message["tool_calls"]) == 2
