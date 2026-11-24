"""The template's two displays, read from what they hand to the notebook."""

from __future__ import annotations

import concept_displays
import pytest
import torch


@pytest.fixture
def shown(monkeypatch) -> list[str]:
    out: list[str] = []
    monkeypatch.setattr(concept_displays, "display", lambda html: out.append(html.data))
    return out


def test_key_token_logits_are_sorted_by_change_and_report_both_ranks(shown):
    pre = torch.tensor([1.0, 3.0, 2.0, 0.0])
    post = torch.tensor([4.0, 1.0, 2.0, 0.0])
    concept_displays.display_key_token_logits(pre, post, [1, 0], ["<color>", "fruit"], title="Key & tokens")
    (markup,) = shown
    assert "Key &amp; tokens" in markup and "&lt;color&gt;" in markup and "<color>" not in markup
    body = markup.split("<tbody>")[1]
    assert body.index("fruit") < body.index("&lt;color&gt;")  # largest rise first, though listed second
    fruit, color = body.split("</tr>")[:2]
    assert "+3.0000" in fruit and "-2.0000" in color
    ranks = [cell.split("</td>")[0] for cell in fruit.split('<td style="text-align:center;">')[1:]]
    assert ranks == ["2", "0"]  # third before the intervention, first after


def test_the_ablation_chart_draws_every_bar_and_clips_what_is_not_a_probability(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    monkeypatch.setattr(plt, "close", lambda *_: None)
    groups = {"baseline": {"fruit": 0.25, "color": 0.5}, "top-10": {"fruit": float("nan"), "color": 1.5}}
    concept_displays.display_ablation_chart(groups, logit_diffs={"baseline": -2.0}, title="ablation")
    bars, line = figures[0].axes
    assert [round(p.get_height(), 2) for p in bars.patches] == [0.25, 0.0, 0.5, 1.0]
    assert list(line.lines[0].get_ydata()) == [-2.0, 0.0]  # a group with no recorded difference draws at zero
    assert bars.get_title() == "ablation"
