#!/usr/bin/env python3
"""Autonomously play the currently loaded Civ VI game with a local Ollama model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISABLED_TOOLS = {
    "delete_unit",
    "kill_game",
    "launch_game",
    "load_game_save",
    "load_save",
    "load_save_from_menu",
    "restart_and_load",
    "run_lua",
}
TURN_ADVANCE_RE = re.compile(r"\bTurn\s+(\d+)\s*->\s*(\d+)\b")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _tool_definition(tool: Any) -> dict[str, Any]:
    schema = dict(tool.inputSchema)
    schema.pop("$schema", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _allowed_tools(tools: list[Any]) -> list[Any]:
    return [tool for tool in tools if tool.name not in DISABLED_TOOLS]


def _compact_history(
    messages: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    if len(messages) <= max_messages:
        return messages

    keep_from = len(messages) - max_messages
    while keep_from < len(messages) and messages[keep_from].get("role") == "tool":
        keep_from += 1
    if keep_from >= len(messages):
        return messages

    return [
        messages[0],
        {
            "role": "user",
            "content": (
                "Conversation history was shortened to fit the model context. "
                "Re-orient with get_game_overview and get_diary, then continue "
                "the current game. Do not load or restart a save."
            ),
        },
        *messages[keep_from:],
    ]


def _turn_advance(text: str) -> int:
    match = TURN_ADVANCE_RE.search(text)
    if not match:
        return 0
    before, after = (int(value) for value in match.groups())
    return 1 if after > before else 0


def _result_text(result: Any) -> str:
    parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text" and isinstance(item.text, str)
    ]
    text = "\n".join(parts)
    if not text and getattr(result, "structuredContent", None) is not None:
        text = json.dumps(result.structuredContent, ensure_ascii=False)
    if getattr(result, "isError", False):
        return f"Tool error: {text or 'MCP tool returned an error.'}"
    return text


def _single_tool_call_message(message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    if len(calls) <= 1:
        return message
    selected = dict(message)
    selected["tool_calls"] = calls[:1]
    return selected


def _ollama_chat(
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    num_ctx: int,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"num_ctx": num_ctx},
        }
    ).encode("utf-8")
    request = Request(
        f"{url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {detail[:1000]}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach Ollama at {url}: {error.reason}") from error

    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise RuntimeError("Ollama returned a response without a message object.")
    return result["message"]


def _parse_arguments(value: Any, tool_name: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Model returned invalid JSON arguments for {tool_name}: {error}"
            ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Model arguments for {tool_name} must be an object.")
    return value


def _log_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _ensure_mcp_port_available(port: int = 8000) -> None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except (ConnectionRefusedError, TimeoutError, OSError):
        return
    raise RuntimeError(
        f"Port {port} is already in use. Another civ6-mcp server is probably "
        "running. Civilization VI's FireTuner supports only one active MCP "
        "connection; shut down the other Civ 6 MCP client/server and retry."
    )


def _ensure_ollama_model(
    url: str, model: str, timeout: int
) -> None:
    request = Request(f"{url.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code} while checking installed models: "
            f"{detail[:1000]}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach Ollama at {url}: {error.reason}") from error

    available = {
        item.get("name")
        for item in result.get("models", [])
        if isinstance(item, dict)
    }
    if model not in available and f"{model}:latest" not in available:
        names = ", ".join(sorted(name for name in available if name)) or "(none)"
        raise RuntimeError(
            f"Ollama model {model!r} is not installed at {url}. "
            f"Installed models: {names}. Pull it with `ollama pull {model}`."
        )


async def _run(args: argparse.Namespace) -> None:
    log_path = args.log or (
        PROJECT_ROOT
        / "logs"
        / f"auto_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_event(
        log_path,
        {
            "type": "run_started",
            "model": args.model,
            "ollama_url": args.ollama_url,
            "max_turns": args.max_turns,
            "started_at": datetime.now().astimezone().isoformat(),
        },
    )

    print("Checking whether another Civ 6 MCP server is active...", flush=True)
    _ensure_mcp_port_available()
    print("MCP port is available.", flush=True)

    print("Checking the local Ollama endpoint and model...", flush=True)
    _ensure_ollama_model(
        args.ollama_url,
        args.model,
        min(args.request_timeout, 10),
    )
    print(f"Ollama model {args.model} is available.", flush=True)

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (src_path, current_pythonpath) if value
    )
    env["PYTHONUTF8"] = "1"
    env["CIV_MCP_AGENT_MODEL"] = args.model
    env["CIV_MCP_AUTO_DISMISS_CONFIRMATIONS"] = "1"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-X", "utf8", "-m", "civ_mcp"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    instructions = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    system_prompt = (
        "You are authorized to autonomously play the currently loaded "
        "Civilization VI game until it ends or the configured turn limit is "
        "reached. Follow the gameplay playbook below. Do not start or launch "
        "a game, load/reload a save, restart/kill the game, execute raw Lua, "
        "or delete units. Those tools are unavailable. Resolve blockers, "
        "make decisions needed to play, and call end_turn with all five "
        "non-empty diary reflections. Stop when a tool reports GAME OVER.\n\n"
        f"{instructions}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Play the game that is already loaded. Start with "
                "get_game_overview and get_diary, then follow the full turn "
                f"loop. Advance no more than {args.max_turns} turns."
            ),
        },
    ]

    print(
        f"Starting the local MCP server; Ollama model: {args.model}",
        flush=True,
    )
    print(f"Transcript: {log_path}", flush=True)
    _log_event(log_path, {"type": "connecting_to_mcp"})
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            print("MCP process started; waiting for initialization...", flush=True)
            try:
                await asyncio.wait_for(session.initialize(), timeout=45)
            except TimeoutError as error:
                raise RuntimeError(
                    "MCP server did not initialize within 45 seconds. Check that "
                    "Civ 6 is running, the FireTuner port 4318 is reachable, and "
                    "no other Civ 6 MCP client is connected."
                ) from error
            print("MCP initialized; loading available tools...", flush=True)
            try:
                listed_tools = await asyncio.wait_for(
                    session.list_tools(), timeout=30
                )
            except TimeoutError as error:
                raise RuntimeError(
                    "MCP server initialized but did not return its tool list "
                    "within 30 seconds."
                ) from error
            available_tools = _allowed_tools(listed_tools.tools)
            tools = [_tool_definition(tool) for tool in available_tools]
            tool_names = {tool.name for tool in available_tools}
            print(
                f"Connected; exposing {len(tools)} tools "
                f"({len(listed_tools.tools) - len(tools)} restricted).",
                flush=True,
            )
            _log_event(
                log_path,
                {
                    "type": "mcp_connected",
                    "tools": len(tools),
                    "restricted_tools": len(listed_tools.tools) - len(tools),
                },
            )

            tool_calls_total = 0
            turns_advanced = 0
            consecutive_idle_responses = 0
            started = time.monotonic()

            while tool_calls_total < args.max_tool_calls:
                messages = _compact_history(messages, args.history_messages)
                assistant = await asyncio.to_thread(
                    _ollama_chat,
                    args.ollama_url,
                    args.model,
                    messages,
                    tools,
                    args.num_ctx,
                    args.request_timeout,
                )
                assistant_message = {
                    key: assistant[key]
                    for key in ("role", "content", "tool_calls")
                    if key in assistant
                }
                assistant_message.setdefault("role", "assistant")
                proposed_calls = assistant_message.get("tool_calls") or []
                assistant_message = _single_tool_call_message(assistant_message)
                messages.append(assistant_message)
                _log_event(
                    log_path,
                    {
                        "type": "assistant",
                        "message": assistant_message,
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                )
                print("Received model response.", flush=True)

                content = assistant_message.get("content", "")
                if content:
                    print(f"\nAgent: {content}")
                calls = assistant_message.get("tool_calls") or []
                if not calls:
                    consecutive_idle_responses += 1
                    if consecutive_idle_responses >= 2:
                        print("Model repeatedly returned no tool calls; stopping.")
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue the authorized autonomous game. Call "
                                "get_game_overview to check whether the game has "
                                "ended; if it is still active, continue playing "
                                "and use the available tools."
                            ),
                        }
                    )
                    continue
                consecutive_idle_responses = 0
                if len(proposed_calls) > 1:
                    print(
                        f"Model proposed {len(proposed_calls)} tool calls; "
                        "executing one at a time and rechecking the game state.",
                        flush=True,
                    )
                if tool_calls_total + 1 > args.max_tool_calls:
                    raise RuntimeError(
                        f"Reached the --max-tool-calls limit ({args.max_tool_calls})."
                    )

                stop_reason: str | None = None
                for call in calls:
                    function = call.get("function", {})
                    name = function.get("name")
                    if name not in tool_names:
                        raise RuntimeError(
                            f"Model requested unavailable tool: {name!r}"
                        )
                    arguments = _parse_arguments(
                        function.get("arguments", {}), name
                    )
                    tool_calls_total += 1
                    arguments_text = json.dumps(arguments, ensure_ascii=False)
                    print(f"\n[{tool_calls_total}] {name}({arguments_text})")
                    tool_started = time.monotonic()
                    if name == "end_turn":
                        call_task = asyncio.create_task(
                            session.call_tool(name, arguments)
                        )
                        while not call_task.done():
                            done, _ = await asyncio.wait(
                                {call_task}, timeout=30
                            )
                            if not done:
                                elapsed = int(time.monotonic() - tool_started)
                                print(
                                    f"end_turn is still processing "
                                    f"({elapsed}s). Civ 6 may take several "
                                    "minutes to resolve AI turns; waiting for "
                                    "the game result...",
                                    flush=True,
                                )
                                _log_event(
                                    log_path,
                                    {
                                        "type": "tool_still_running",
                                        "tool": name,
                                        "elapsed_seconds": elapsed,
                                    },
                                )
                        result = call_task.result()
                    else:
                        result = await session.call_tool(name, arguments)
                    result_text = _result_text(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": result_text,
                        }
                    )
                    _log_event(
                        log_path,
                        {
                            "type": "tool_call",
                            "tool": name,
                            "arguments": arguments,
                            "result": result_text,
                            "elapsed_seconds": round(
                                time.monotonic() - tool_started, 1
                            ),
                        },
                    )
                    if result_text:
                        print(result_text[:2000])
                        if len(result_text) > 2000:
                            print(f"... ({len(result_text)} characters total)")

                    if name == "end_turn":
                        turns_advanced += _turn_advance(result_text)
                        if turns_advanced >= args.max_turns:
                            stop_reason = (
                                f"Reached the limit of {args.max_turns} advanced turns."
                            )
                    if "GAME OVER" in result_text:
                        stop_reason = "The game reported GAME OVER."
                    if stop_reason:
                        break

                if stop_reason:
                    print(f"\nStopping: {stop_reason}")
                    break
            else:
                raise RuntimeError(
                    f"Reached the --max-tool-calls limit ({args.max_tool_calls})."
                )

    print(f"Run finished. Turns advanced: {turns_advanced}. Log: {log_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Let a local Ollama model play the currently loaded Civ VI game "
            "through this repository's MCP server."
        )
    )
    parser.add_argument("--model", default="qwen7b-civ6:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-turns", type=_positive_int, default=330)
    parser.add_argument("--max-tool-calls", type=_positive_int, default=20000)
    parser.add_argument("--num-ctx", type=_positive_int, default=32768)
    parser.add_argument("--history-messages", type=_positive_int, default=40)
    parser.add_argument("--request-timeout", type=_positive_int, default=600)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.history_messages < 20:
        parser.error("--history-messages must be at least 20")
    return args


def main() -> None:
    try:
        asyncio.run(_run(_parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted by user; the currently loaded game was not reloaded.")
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
