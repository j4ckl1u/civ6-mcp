from civ_mcp.game_launcher import _find_confirmation_button


def test_confirmation_button_matches_exact_english_or_localized_label():
    assert _find_confirmation_button(
        [
            ("OK", 10, 20, 30, 15),
            ("确定", 40, 30, 30, 15),
            ("Look okay", 70, 40, 60, 15),
        ]
    ) == ("确定", 40, 30, 30, 15)


def test_confirmation_button_prefers_lowest_matching_button():
    assert _find_confirmation_button(
        [
            ("OK", 10, 20, 30, 15),
            ("OK.", 40, 70, 30, 15),
        ]
    ) == ("OK.", 40, 70, 30, 15)


def test_confirmation_button_does_not_click_unrelated_text():
    assert _find_confirmation_button(
        [("Are you sure?", 10, 20, 100, 15), ("Yes", 20, 50, 30, 15)]
    ) is None
