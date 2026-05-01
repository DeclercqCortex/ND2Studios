"""
Reusable scale bar drawing for matplotlib axes.
"""
from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt


# Valid values for ``text_position``
TEXT_POSITIONS = ("auto", "above", "below", "left", "right", "center", "none")


def draw_scale_bar(
    ax,
    pixel_size_um: float = 1.0,
    bar_length_um: float = 100.0,
    location: str = "bottom-right",
    bar_color: str = "white",
    text_color: str = "white",
    font_size: int = 10,
    bar_thickness: int = 5,
    padding: float = 0.03,
    show_text: bool = True,
    text_position: str = "auto",
    text_offset: float = 0.5,
    unit: str = "µm",
    bg_alpha: float = 0.5,
    bg_color: str = "black",
):
    """
    Draw a scale bar on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib Axes
    pixel_size_um : float, size of one pixel in microns (or other unit)
    bar_length_um : float, desired bar length in microns
    location : str, one of "bottom-right", "bottom-left", "top-right", "top-left"
    bar_color : str, color of the bar
    text_color : str, color of the label text
    font_size : int, label font size in points
    bar_thickness : int, bar height in pixels (in data coordinates)
    padding : float, fraction of axes size for padding from edge
    show_text : bool, whether to show the label (overrides ``text_position``
        if ``False``)
    text_position : str, one of "auto", "above", "below", "left", "right",
        "center", "none". "auto" places the label outside the bar on the
        interior side of the axes (above for image-convention y-inverted
        axes, below otherwise). "none" hides the label.
    text_offset : float, gap between bar and text in multiples of
        ``bar_thickness`` (or ``bar_length_px`` when the text is left/right
        of the bar); 0 = flush against bar.
    unit : str, unit label (e.g. "µm", "px")
    bg_alpha : float, background box opacity (0 = no background)
    bg_color : str, background box color
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])
    x_min, x_max = min(xlim), max(xlim)
    y_min, y_max = min(ylim), max(ylim)
    y_inverted = ylim[0] > ylim[1]

    # Bar length in data coordinates
    bar_length_px = bar_length_um / pixel_size_um

    # Position
    pad_x = x_range * padding
    pad_y = y_range * padding

    if "right" in location:
        x0 = x_max - pad_x - bar_length_px
    else:
        x0 = x_min + pad_x

    if "bottom" in location:
        y0 = (y_max - pad_y - bar_thickness) if y_inverted else (y_min + pad_y)
    else:
        y0 = (y_min + pad_y) if y_inverted else (y_max - pad_y - bar_thickness)

    # ── Resolve text position ──
    if not show_text:
        text_position = "none"
    if text_position not in TEXT_POSITIONS:
        text_position = "auto"
    if text_position == "auto":
        # Default: label on the interior side (above the bar for image axes).
        text_position = "above" if y_inverted else "below"

    gap_y = bar_thickness * max(0.0, text_offset)
    gap_x = bar_length_px * 0.05 + bar_thickness * max(0.0, text_offset)

    label = f"{bar_length_um:g} {unit}"
    text_h = font_size * 1.4  # rough text height in display pts — used for bg sizing

    # ── Background box ──
    if bg_alpha > 0:
        bg_pad = bar_thickness * 2
        bg_x = x0 - bg_pad
        bg_y = y0 - bg_pad
        bg_w = bar_length_px + bg_pad * 2
        bg_h = bar_thickness + bg_pad * 2

        if text_position in ("above", "below") and text_position != "none":
            bg_h += text_h + gap_y
            if text_position == "above":
                bg_y = y0 - gap_y - text_h - bg_pad
        elif text_position in ("left", "right") and text_position != "none":
            # Guess text width ≈ 0.6 * len(label) * font_size (in pts). This
            # is approximate but fine for a translucent background box.
            text_w = 0.6 * len(label) * font_size
            bg_w += text_w + gap_x
            if text_position == "left":
                bg_x = x0 - gap_x - text_w - bg_pad

        rect = plt.Rectangle(
            (bg_x, bg_y), bg_w, bg_h,
            facecolor=bg_color, alpha=bg_alpha,
            edgecolor="none", zorder=10,
        )
        ax.add_patch(rect)

    # ── Bar ──
    bar = plt.Rectangle(
        (x0, y0), bar_length_px, bar_thickness,
        facecolor=bar_color, edgecolor="none", zorder=11,
    )
    ax.add_patch(bar)

    # ── Text ──
    if text_position == "none":
        return

    if text_position == "above":
        # "above" = smaller y in image-convention, larger y otherwise.
        tx = x0 + bar_length_px / 2
        ty = (y0 - gap_y) if y_inverted else (y0 + bar_thickness + gap_y)
        ha, va = "center", ("bottom" if y_inverted else "bottom")
        # va interpretation: "above" means place text baseline above the bar.
        va = "bottom" if y_inverted else "bottom"
    elif text_position == "below":
        tx = x0 + bar_length_px / 2
        ty = (y0 + bar_thickness + gap_y) if y_inverted else (y0 - gap_y)
        ha, va = "center", ("top" if y_inverted else "top")
    elif text_position == "left":
        tx = x0 - gap_x
        ty = y0 + bar_thickness / 2
        ha, va = "right", "center"
    elif text_position == "right":
        tx = x0 + bar_length_px + gap_x
        ty = y0 + bar_thickness / 2
        ha, va = "left", "center"
    elif text_position == "center":
        tx = x0 + bar_length_px / 2
        ty = y0 + bar_thickness / 2
        ha, va = "center", "center"
    else:
        return

    ax.text(
        tx, ty, label,
        color=text_color, fontsize=font_size, fontweight="bold",
        ha=ha, va=va, zorder=12,
    )
