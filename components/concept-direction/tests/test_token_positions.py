"""The token-position strip: each labelled position marked once, text escaped, checked without a kernel."""

from __future__ import annotations

import pytest
from concept_displays import build_token_positions_html


def test_token_positions_mark_each_labelled_position_once_in_the_legend():
    tokens = ["<bos>", "\u2581apple", "\u2581:", "\n", "Fruit"]
    marks = {1: ("probe word", "#111111"), 3: ("context state", "#222222"), 4: ("answer state", "#333333")}
    html_out = build_token_positions_html(tokens, marks, title="Where <states> are read")
    assert "Where &lt;states&gt; are read" in html_out and "&lt;bos&gt;" in html_out
    assert " apple" in html_out and "\u23ce" in html_out  # the space marker and the newline as a reader sees them
    assert html_out.count('class="lab"') == 3 and 'title="position 4: answer state"' in html_out
    assert html_out.count('class="sw"') == 3
    with pytest.raises(ValueError, match=r"marked positions \[5\] are outside the prompt's 5 tokens"):
        build_token_positions_html(tokens, {5: ("x", "#000")}, title="t")
