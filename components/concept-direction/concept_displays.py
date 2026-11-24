"""Displays the concept-direction notebooks use, beside the shared ones in ``interpretune.harness.notebook_ui``.

Token positions in a prompt (latent dynamics), the J-lens direction validation and the steering scale sweep (concept
steering), and the key-token logits and ablation chart (the experiment template). Each table is built by a pure ``build_*`` function, so a test reads the markup without a kernel.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
from IPython.display import HTML, display

from interpretune.harness.notebook_ui import display_plotly_figure


def _token_text(token: str) -> str:
    """A tokenizer piece as a reader sees it: the SentencePiece space marker as a space, a newline as a return sign."""
    return token.replace("\u2581", " ").replace("\n", "\u23ce")


def build_token_positions_html(
    tokens: Sequence[str],
    marks: Mapping[int, tuple[str, str]],
    *,
    title: str,
    caption: str = "",
) -> str:
    """One prompt as a strip of token chips, the marked positions coloured and labelled beneath.

    ``marks`` maps a position to ``(label, colour)``; positions sharing a label (a word's span) share its colour, and
    the legend lists each label once. Every chip carries its position on hover. Token text is escaped; nothing is HTML.
    """
    if bad := sorted(pos for pos in marks if not 0 <= pos < len(tokens)):
        raise ValueError(f"marked positions {bad} are outside the prompt's {len(tokens)} tokens")
    chips = []
    for pos, token in enumerate(tokens):
        label, colour = marks.get(pos, ("", ""))
        style = f"background:{html.escape(colour, quote=True)};color:white;font-weight:600" if label else ""
        under = f'<span class="lab">{html.escape(label)}</span>' if label else ""
        chips.append(
            f'<span class="tok" title="position {pos}{": " + html.escape(label) if label else ""}">'
            f'<span class="txt" style="{style}">{html.escape(_token_text(token)) or "&nbsp;"}</span>{under}</span>'
        )
    legend = dict.fromkeys(marks.values())
    swatches = "".join(
        f'<span class="sw"><span style="background:{html.escape(colour, quote=True)}"></span>'
        f"{html.escape(label)}</span>"
        for label, colour in legend
    )
    return f"""<style>
    .it-tokens {{font-family:system-ui,sans-serif; font-size:13px; margin:12px 0;}}
    .it-tokens .tok {{display:inline-flex; flex-direction:column; align-items:center; margin:1px; vertical-align:top;}}
    .it-tokens .txt {{font-family:monospace; padding:2px 4px; border-radius:3px; background:#78b4f018;
    white-space:pre;}}
    .it-tokens .lab {{font-size:10px; margin-top:1px;}}
    .it-tokens .sw {{margin-right:14px; white-space:nowrap; font-size:12px;}}
    .it-tokens .sw span {{display:inline-block; width:10px; height:10px; margin-right:4px; vertical-align:middle;}}
    </style><div class="it-tokens"><div><strong>{html.escape(title)}</strong></div>
    <div>{html.escape(caption)}</div><div style="line-height:2.2">{"".join(chips)}</div><div>{swatches}</div></div>"""


def display_token_positions(
    tokens: Sequence[str], marks: Mapping[int, tuple[str, str]], *, title: str, caption: str = ""
) -> None:
    """Display :func:`build_token_positions_html` in a notebook or its rendered documentation."""
    display(HTML(build_token_positions_html(tokens, marks, title=title, caption=caption)))


class JlensDirectionValidationSummary(NamedTuple):
    """What :func:`display_jlens_direction_validation` rendered, for assertions without re-deriving it."""

    direction_labels: tuple[str, ...]
    direction_shares: tuple[float, ...]
    direction_fractions: tuple[float, ...]
    attribution_total: float
    predicted_delta: float
    unexplained_remainder: float


def build_jlens_direction_validation_html(
    attribution: Mapping[str, Any],
    direction_labels: Sequence[str],
    *,
    finite_difference_slopes: Sequence[float] | None = None,
    measured_delta: float | None = None,
    title: str = "J-lens direction validation",
) -> tuple[str, JlensDirectionValidationSummary]:
    """Build the J-lens direction validation table as HTML; :func:`display_jlens_direction_validation` shows it.

    One row per J-lens direction from ``attribution`` (the dict
    :func:`~interpretune.analysis.ops.bundled.jlens.jlens_ops.subspace_attribution_scores` returns), with the
    coordinate change ``Δc_i``, the gradient readout ``w_i``, their product ``a_i``, the direction's fraction of
    the explained mass and, when given, an independent central-difference slope; a footer row each for the
    explained total, the remainder, the first-order prediction ``gᵀΔh`` and, when given, the measured logit change.

    This validates the J-lens directions against the edit they produced. It deliberately carries no feature
    column: SAE-feature influence and J-lens gap shares are different vocabularies, and setting them side by side
    implies a comparison this machinery cannot yet make. Pure (no display), so a test can check the markup and the
    summary without a kernel.
    """
    shares = [float(x) for x in attribution["attribution_shares"]]
    labels = [str(lbl) for lbl in direction_labels]
    if len(labels) != len(shares):
        raise ValueError(f"{len(labels)} direction labels for {len(shares)} attribution shares")
    delta_coords = [float(x) for x in attribution.get("delta_coords") or [float("nan")] * len(shares)]
    readouts = [float(x) for x in attribution.get("readouts") or [float("nan")] * len(shares)]
    slopes = [float(x) for x in finite_difference_slopes] if finite_difference_slopes is not None else None
    if slopes is not None and len(slopes) != len(shares):
        raise ValueError(f"{len(slopes)} finite-difference slopes for {len(shares)} attribution shares")
    total = float(attribution["attribution_total"])
    predicted = float(attribution["predicted_delta"])
    remainder = float(attribution["unexplained_remainder"])
    share_mass = sum(abs(a) for a in shares)
    fractions = [abs(a) / share_mass if share_mass else 0.0 for a in shares]

    # Sized to its content and scrolling inside the widget when the page is narrower, so the docs theme never
    # squeezes the table into its own box or scrolls the page sideways.
    style = """
    <style>
    .jl-val { font-family: system-ui, -apple-system, sans-serif; margin-bottom: 12px; font-size: 13px; }
    .jl-val .title { font-weight: bold; font-size: 14px; margin-bottom: 6px; padding: 4px 6px;
        border-radius: 3px; background: #2471A3; color: white; display: inline-block; }
    .jl-val .scroll { overflow-x: auto; padding-bottom: 6px; }
    .jl-val table { width: max-content; max-width: none; border-collapse: collapse; }
    .jl-val th, .jl-val td { padding: 3px 6px; border: 1px solid rgba(150,150,150,0.5); text-align: right; }
    .jl-val td { white-space: nowrap; }
    .jl-val th { white-space: normal; }
    .jl-val td.lbl, .jl-val th.lbl { text-align: left; }
    .jl-val tr.total td { background: rgba(120,180,240,0.15); font-weight: bold; }
    .jl-val .footnote { font-size: 12px; color: #666; margin-top: 6px; max-width: 900px; }
    </style>
    """

    def _signed(v: float, precision: int = 4) -> str:
        return "n/a" if v != v else f"{v:+.{precision}f}"  # NaN when a factor was not supplied

    # Headers are short because the table is sized to its content: a header long enough to want wrapping simply
    # widens the table instead. The symbols are spelled out in the legend.
    slope_header = "<th>dGap/ds</th>" if slopes is not None else ""
    table = (
        '<table><thead><tr><th class="lbl">Direction</th><th>&#916;c</th><th>Readout w</th>'
        f"<th>Share a</th><th>Fraction</th>{slope_header}</tr></thead><tbody>"
    )
    for i, lbl in enumerate(labels):
        slope_cell = f"<td>{_signed(slopes[i])}</td>" if slopes is not None else ""
        table += (
            f'<tr><td class="lbl">{html.escape(lbl)}</td><td>{_signed(delta_coords[i])}</td>'
            f"<td>{_signed(readouts[i])}</td><td>{shares[i]:+.4f}</td><td>{fractions[i]:.1%}</td>{slope_cell}</tr>"
        )
    # A footer value is a share-space quantity, so it must not land under the slope header, where it would read as
    # a slope: one left-aligned cell spanning every column bar the label keeps it beside its label and out of
    # every numeric column. Six columns with the slope column present, five without.
    span = 5 if slopes is not None else 4

    def _footer(label: str, value: float) -> str:
        return f'<tr class="total"><td class="lbl">{label}</td><td class="lbl" colspan="{span}">{value:+.4f}</td></tr>'

    table += _footer("Explained &#931;a", total)
    table += _footer("Remainder", remainder)
    table += _footer("First-order g&#7488;&#916;h", predicted)
    if measured_delta is not None:
        table += _footer("Measured &#916;gap", float(measured_delta))
    table += "</tbody></table>"

    # The legend defines only the columns actually rendered: naming a column the reader cannot see is the same
    # defect in prose that an empty column would be in the table.
    legend = (
        "&#916;c = V&#8314;&#916;h, the displacement's coordinates in the pair; w = V&#7488;g, the gradient's "
        "readout of each direction; a = w&#183;&#916;c, the direction's share of the first-order prediction "
        "g&#7488;&#916;h. Fraction is |a| over &#931;|a|."
    )
    if slopes is not None:
        legend += " dGap/ds is the central-difference slope of the gap along that direction, an independent check on w."
    footnote = (
        f"{legend} The first-order prediction is the gradient's linear estimate of the patch's effect; the measured "
        "change includes everything nonlinear downstream of the site."
    )
    markup = (
        f'{style}<div class="jl-val"><div class="title">{html.escape(title)}</div>'
        f'<div class="scroll">{table}</div><div class="footnote">{footnote}</div></div>'
    )
    summary = JlensDirectionValidationSummary(
        direction_labels=tuple(labels),
        direction_shares=tuple(shares),
        direction_fractions=tuple(fractions),
        attribution_total=total,
        predicted_delta=predicted,
        unexplained_remainder=remainder,
    )
    return markup, summary


def display_jlens_direction_validation(
    attribution: Mapping[str, Any], direction_labels: Sequence[str], **kwargs: Any
) -> JlensDirectionValidationSummary:
    """Render :func:`build_jlens_direction_validation_html` and return its summary (see that function for the
    columns)."""
    markup, summary = build_jlens_direction_validation_html(attribution, direction_labels, **kwargs)
    display(HTML(markup))
    return summary


def build_steering_scale_sweep_html(points: Sequence[Any], token_labels: Sequence[str]) -> str:
    """Build the per-arm, per-scale table of logit and probability shifts for the two target tokens.

    ``points`` are :class:`~steering_demo_helpers.SteeringScalePoint` rows. Each shift is signed
    and coloured by the repository convention (green positive, red negative), and the last column says whether the
    answer flipped, which is what the chart above it shows. Pure (no display), so a test can check the markup.
    """
    if len(token_labels) != 2:
        raise ValueError(f"expected two target token labels, got {len(token_labels)}")
    a, b = (html.escape(str(t)) for t in token_labels)

    def _signed(v: float, fmt: str) -> str:
        colour = "#1a7f37" if v > 0 else ("#d1242f" if v < 0 else "inherit")
        return f'<td style="color:{colour};font-weight:600">{v:{fmt}}</td>'

    rows = ""
    for p in points:
        # A strict sign change: a gap that lands exactly on zero is a tie, not a flip.
        flipped = p.pre_gap * p.post_gap < 0
        rows += (
            f'<tr><td class="lbl">{html.escape(str(p.arm))}</td><td>{p.scale:g}</td>'
            + _signed(p.post_logits[0] - p.pre_logits[0], "+.3f")
            + _signed(p.post_logits[1] - p.pre_logits[1], "+.3f")
            + _signed(p.post_gap - p.pre_gap, "+.3f")
            + _signed(100 * (p.post_probs[0] - p.pre_probs[0]), "+.2f")
            + _signed(100 * (p.post_probs[1] - p.pre_probs[1]), "+.2f")
            + f"<td>{'yes' if flipped else 'no'}</td></tr>"
        )
    style = """
    <style>
    .sweep { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; margin-bottom: 12px; }
    .sweep .scroll { overflow-x: auto; }
    .sweep table { width: max-content; border-collapse: collapse; }
    .sweep th, .sweep td { padding: 3px 8px; border: 1px solid rgba(150,150,150,0.5); text-align: right; }
    .sweep td.lbl, .sweep th.lbl { text-align: left; }
    .sweep .footnote { font-size: 12px; color: #666; margin-top: 6px; max-width: 900px; }
    </style>
    """
    header = (
        '<tr><th class="lbl">Arm</th><th>Scale</th>'
        f"<th>&#916; logit {a}</th><th>&#916; logit {b}</th><th>&#916; gap ({a} &#8722; {b})</th>"
        f"<th>&#916; prob {a} (pp)</th><th>&#916; prob {b} (pp)</th><th>Answer flipped</th></tr>"
    )
    footnote = (
        "Shifts are post minus pre at the answer position; probabilities are over the full vocabulary, in percentage "
        "points. Scales are each arm's own multiplier, not matched displacements across arms."
    )
    return (
        f'{style}<div class="sweep"><div class="scroll"><table><thead>{header}</thead><tbody>{rows}</tbody>'
        f'</table></div><div class="footnote">{footnote}</div></div>'
    )


def display_steering_scale_sweep(points: Sequence[Any], token_labels: Sequence[str], title: str) -> None:
    """Chart each arm's post-intervention gap against scale, with the answer-flip line, then show the shift table.

    The chart plots the gap after intervention rather than its change, so the flip threshold is one line at zero shared
    by every arm (all arms start from the same clean gap), and the smallest scale at which each arm crosses it is read
    off the chart rather than asserted in prose.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plotly unavailable] scale-sweep chart skipped; the table below carries the same numbers")
    else:
        fig = go.Figure()
        for arm in dict.fromkeys(p.arm for p in points):
            arm_points = sorted((p for p in points if p.arm == arm), key=lambda p: p.scale)
            fig.add_trace(
                go.Scatter(
                    x=[p.scale for p in arm_points], y=[p.post_gap for p in arm_points], mode="lines+markers", name=arm
                )
            )
        fig.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"answer flips above ({token_labels[0]} > {token_labels[1]})",
            annotation_position="top left",
        )
        fig.update_layout(
            title=title,
            autosize=True,
            height=420,
            template="simple_white",
            xaxis=dict(title="intervention scale (each arm's own multiplier)", type="log"),
            yaxis=dict(title=f"gap after intervention ({token_labels[0]} − {token_labels[1]}, logits)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        )
        display_plotly_figure(fig, "it-scale-sweep")
    display(HTML(build_steering_scale_sweep_html(points, token_labels)))


def display_ablation_chart(
    groups: dict[str, dict[str, float]],
    logit_diffs: dict[str, float] | None = None,
    title: str = "",
    colors: list[str] | None = None,
) -> None:
    """Display ablation results as a grouped bar chart with optional logit-difference line.

    Adapted from ``circuit_tracer.utils.demo_utils.display_ablation_chart``.

    Args:
        groups: Mapping from condition label (e.g. ``"baseline"``, ``"top-10"``)
            to a dict of ``{token_label: probability}``.
        logit_diffs: Optional mapping from condition label to a scalar logit
            difference value.  When provided, a dashed line overlay is drawn on
            a secondary y-axis.
        title: Chart title.
        colors: Optional list of bar colors (one per token label).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Close any leaked figures to avoid corrupted renderer state.
    plt.close("all")

    group_labels = list(groups.keys())
    token_labels = list(next(iter(groups.values())).keys())
    n_groups = len(group_labels)
    n_tokens = len(token_labels)

    if colors is None:
        colors = ["#2471A3", "#E67E22", "#27AE60", "#C0392B", "#8E44AD"][:n_tokens]

    def _safe_probability(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    x = np.arange(n_groups)
    width = 0.8 / n_tokens

    fig, ax1 = plt.subplots(figsize=(8, 5.0))

    for i, tok in enumerate(token_labels):
        vals = [_safe_probability(groups[g].get(tok, 0.0)) for g in group_labels]
        offset = (i - (n_tokens - 1) / 2) * width
        bars = ax1.bar(x + offset, vals, width * 0.9, label=tok, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    max_prob = max(max(_safe_probability(groups[g].get(t, 0.0)) for t in token_labels) for g in group_labels)
    ax1.set_ylim(0, max(0.05, max_prob * 1.4))

    if logit_diffs is not None:
        ax2 = ax1.twinx()
        diff_vals = [
            float(logit_diffs.get(g, 0.0)) if np.isfinite(logit_diffs.get(g, 0.0)) else 0.0 for g in group_labels
        ]
        ax2.plot(x, diff_vals, "k--o", label="Logit diff", linewidth=1.5, markersize=5)
        ax2.set_ylabel("Logit difference")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    if title:
        ax1.set_title(title, fontsize=13, fontweight="bold")
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.18)
    plt.show()
    plt.close(fig)


def display_key_token_logits(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    token_ids: list[int],
    token_labels: list[str],
    title: str = "Key-Token Logit Analysis",
) -> None:
    """Display a styled HTML table of per-token logit changes before/after intervention.

    Accepts 1-D logit vectors (vocab dimension).

    Args:
        pre_logits: Logits before intervention (1-D, vocab size).
        post_logits: Logits after intervention (1-D, vocab size).
        token_ids: Vocabulary indices to display.
        token_labels: Human-readable label for each token.
        title: Heading rendered above the table.
    """
    pre = pre_logits.float().cpu()
    post = post_logits.float().cpu()

    # Compute ranks
    _, pre_sorted = torch.sort(pre, descending=True)
    _, post_sorted = torch.sort(post, descending=True)
    pre_rank = {int(idx): r for r, idx in enumerate(pre_sorted.tolist())}
    post_rank = {int(idx): r for r, idx in enumerate(post_sorted.tolist())}

    # Sort by delta: most positive first, most negative last
    token_data = []
    for tid, label in zip(token_ids, token_labels):
        pre_v = pre[tid].item()
        post_v = post[tid].item()
        delta = post_v - pre_v
        pr = pre_rank.get(tid, -1)
        por = post_rank.get(tid, -1)
        token_data.append((tid, label, pre_v, post_v, delta, pr, por))
    token_data.sort(key=lambda x: x[4], reverse=True)

    rows = ""
    for i, (tid, label, pre_v, post_v, delta, pr, por) in enumerate(token_data):
        delta_color = "#27AE60" if delta > 0 else "#C0392B" if delta < 0 else "#888"
        row_bg = "rgba(240,240,240,0.1)" if i % 2 == 0 else "rgba(255,255,255,0.1)"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="font-family:monospace;">{html.escape(label)}</td>'
            f'<td style="text-align:right;">{pre_v:.4f}</td>'
            f'<td style="text-align:center;">{pr}</td>'
            f'<td style="text-align:right;">{post_v:.4f}</td>'
            f'<td style="text-align:center;">{por}</td>'
            f'<td style="text-align:right;color:{delta_color};font-weight:bold;">{delta:+.4f}</td>'
            f"</tr>\n"
        )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:650px;margin-bottom:12px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:6px;padding:4px 8px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(title)}</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Token</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Pre Logit</th>
                    <th style="text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Rank</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Post Logit</th>
                    <th style="text-align:center;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Rank</th>
                    <th style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Δ Logit</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))
