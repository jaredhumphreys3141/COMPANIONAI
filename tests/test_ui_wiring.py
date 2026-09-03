"""Regression tests for interface wiring that unit tests would otherwise miss.

These introspect the built Gradio graph.  They are deliberately tolerant about
Gradio's internals: if a future version renames the attributes they use, the
test skips rather than failing for the wrong reason.
"""

import pytest

gr = pytest.importorskip("gradio")

from companionai.ui import app as ui  # noqa: E402
from companionai.ui.theme import CSS  # noqa: E402


def _talk_companion_dropdown(demo):
    matches = [
        block for block in demo.blocks.values()
        if isinstance(block, gr.Dropdown) and getattr(block, "label", "") == "Companion"
    ]
    assert len(matches) == 1, "expected exactly one Talk-tab companion picker"
    return matches[0]


def _writers_of(demo, component):
    if not hasattr(demo, "fns"):
        pytest.skip("this Gradio version does not expose the event graph")
    return [
        fn for fn in demo.fns.values()
        if any(getattr(out, "_id", None) == component._id for out in (fn.outputs or []))
    ]


def test_talk_companion_picker_is_refreshed_by_the_studio():
    """A companion created in the Studio must appear in the Talk tab.

    The picker's choices are baked into the page at build time, so unless
    something explicitly pushes new choices into it, companions created later
    are invisible until the whole app restarts.  That was the original bug.
    """
    demo = ui.build()
    picker = _talk_companion_dropdown(demo)
    writers = _writers_of(demo, picker)
    assert writers, "nothing updates the Talk tab's companion picker"
    # save, save-and-activate, create, duplicate, delete, and the tab-open refresh
    assert len(writers) >= 5, f"only {len(writers)} handler(s) refresh the picker"


def test_new_companions_reach_the_choice_list():
    from companionai import character as character_mod

    # An empty data directory yields a single placeholder entry, not an empty
    # list, so count growth only from the first real companion onwards.
    assert ui._character_choices() == [""]

    character_mod.Character(name="Test Subject").save()
    assert ui._character_choices() == ["test-subject"]

    character_mod.Character(name="Second One").save()
    after = ui._character_choices()
    assert set(after) == {"test-subject", "second-one"}


def test_net_badge_offers_a_short_label_for_small_screens():
    html = ui._net_badge()
    assert "net-badge-long" in html
    assert "net-badge-short" in html
    assert "OFFLINE" in html
    assert "title=" in html, "the full text should survive as a tooltip"


def test_narrow_screen_css_swaps_the_badge_and_reflows_the_header():
    # The 7" Raspberry Pi touchscreen is 800x480; both rules must be present or
    # the badge overlaps the status text.
    assert "@media (max-width: 820px)" in CSS
    assert ".net-badge-short { display:none; }" in CSS      # hidden by default
    assert ".net-badge-long { display:none; }" in CSS       # hidden when narrow
    assert "min-width: 0" in CSS, "flex children must be allowed to shrink"
    assert "flex-basis:100%" in CSS, "status line needs its own row when narrow"
