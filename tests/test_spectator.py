from types import SimpleNamespace
from unittest.mock import MagicMock

from civ_mcp.spectator import CameraController, PopupWatcher


def test_camera_controller_pauses_ingame_calls_during_pending_end_turn():
    camera = CameraController(
        MagicMock(), SimpleNamespace(_pending_end_turn=True)
    )

    assert camera._should_send_ingame() is False


def test_camera_controller_resumes_ingame_calls_after_turn_processing():
    camera = CameraController(
        MagicMock(), SimpleNamespace(_pending_end_turn=False)
    )

    assert camera._should_send_ingame() is True


def test_popup_watcher_pauses_ingame_polling_during_pending_end_turn():
    watcher = PopupWatcher(
        MagicMock(), SimpleNamespace(_pending_end_turn=True)
    )

    assert watcher._should_poll_ingame() is False


def test_popup_watcher_resumes_ingame_polling_after_turn_processing():
    watcher = PopupWatcher(
        MagicMock(), SimpleNamespace(_pending_end_turn=False)
    )

    assert watcher._should_poll_ingame() is True


def test_popup_watcher_without_game_state_keeps_existing_polling():
    watcher = PopupWatcher(MagicMock())

    assert watcher._should_poll_ingame() is True
