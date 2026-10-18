import asyncio
from unittest.mock import AsyncMock, patch

from civ_mcp.game_lifecycle import dismiss_popup
from civ_mcp.spectator import _POPUP_POLL_LUA


def test_popup_watcher_polls_for_advisor_popup():
    assert "LookUpControl('/InGame/AdvisorPopup')" in _POPUP_POLL_LUA


def test_dismiss_popup_closes_advisor_popup():
    class FakeConnection:
        lua_states = {}

        async def execute_write(self, lua, **kwargs):
            if 'LookUpControl("/InGame/AdvisorPopup")' in lua:
                return ["DISMISSED|AdvisorPopup", "---END---"]
            return ["---END---"]

    with patch(
        "civ_mcp.game_launcher.dismiss_crash_dialogs",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = asyncio.run(dismiss_popup(FakeConnection()))

    assert result == "Dismissed: AdvisorPopup"
