#!/usr/bin/env python3
"""
Plot OPT-6.7B VRAM usage from baseline_opt6.7b_max_length_data.csv.

Run from the FlexLLMGen directory:
    python3 my_script/plot_opt67b_vram_bar_chart.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any
from html import escape


SCRIPT_DIR = Path(__file__).resolve().parent
FLEXLLMGEN_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "my_script" else SCRIPT_DIR
DEFAULT_CSV = FLEXLLMGEN_ROOT / "output_data" / "baseline_opt6.7b_max_length_data.csv"
DEFAULT_OUTPUT = FLEXLLMGEN_ROOT / "output_data" / "baseline_opt6.7b_vram_bar_chart.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a stacked VRAM bar chart for the OPT-6.7B baseline sweep."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vram-limit", type=float, default=16.0)
    parser.add_argument("--title", type=str, default="OPT-6.7B Baseline VRAM Usage")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int:
    parsed = to_float(value)
    if parsed is None:
        raise ValueError(f"Expected an integer-like value, got {value!r}")
    return int(parsed)


def read_latest_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    latest_by_prompt: dict[int, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_len = to_int(row["prompt_len"])
            latest_by_prompt[prompt_len] = row

    rows = list(latest_by_prompt.values())
    rows.sort(key=lambda item: to_int(item["prompt_len"]))
    return rows


def plot(rows: list[dict[str, Any]], output_path: Path, vram_limit: float, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except ImportError as exc:
        svg_path = output_path.with_suffix(".svg")
        print(f"matplotlib is not available ({exc}); writing SVG instead.")
        plot_svg(rows, svg_path, vram_limit, title)
        return

    if not rows:
        raise SystemExit("CSV has no data rows.")

    prompt_lens = [to_int(row["prompt_len"]) for row in rows]
    statuses = [str(row["status"]).strip().upper() for row in rows]
    labels = [
        str(prompt_len) if status == "SUCCESS" else f"{prompt_len}\n{status}"
        for prompt_len, status in zip(prompt_lens, statuses)
    ]

    model_weight = np.array([to_float(row.get("model_weight_gb")) or 0.0 for row in rows])
    kv_cache = np.array([to_float(row.get("kv_cache_gb")) or 0.0 for row in rows])
    activation = np.array([to_float(row.get("activation_gb")) or 0.0 for row in rows])
    peak_gpu = np.array([to_float(row.get("peak_gpu_mem_gb")) or np.nan for row in rows])
    stacked_total = model_weight + kv_cache + activation
    success_mask = np.array([status == "SUCCESS" for status in statuses])
    oom_mask = ~success_mask

    x = np.arange(len(rows))
    fig_width = max(11.0, len(rows) * 0.9)
    fig, (ax, ax_act) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8.8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.25], "hspace": 0.08},
    )

    model_color = "#264653"
    kv_color = "#2a9d8f"
    activation_color = "#e9c46a"
    peak_color = "#1d1d1f"
    oom_color = "#c1121f"

    ax.bar(x, model_weight, label="Model Weight", color=model_color, edgecolor="white", linewidth=0.8)
    ax.bar(
        x,
        kv_cache,
        bottom=model_weight,
        label="KV Cache",
        color=kv_color,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        x,
        activation,
        bottom=model_weight + kv_cache,
        label="Activation",
        color=activation_color,
        edgecolor="white",
        linewidth=0.8,
    )

    if not np.isnan(peak_gpu[success_mask]).all():
        ax.plot(
            x[success_mask],
            peak_gpu[success_mask],
            color=peak_color,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label="Peak GPU Memory",
        )

    ax.axhline(
        vram_limit,
        color=oom_color,
        linestyle="--",
        linewidth=1.8,
        label=f"VRAM Limit ({vram_limit:.0f} GB)",
    )

    if oom_mask.any():
        oom_y = np.full(oom_mask.sum(), vram_limit)
        ax.scatter(
            x[oom_mask],
            oom_y,
            marker="X",
            s=130,
            color=oom_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
            label="OOM / Failed Run",
        )
        for idx in x[oom_mask]:
            ax.annotate(
                "OOM",
                xy=(idx, vram_limit),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=oom_color,
            )

    for idx, is_success in enumerate(success_mask):
        if not is_success:
            continue
        top = peak_gpu[idx] if not np.isnan(peak_gpu[idx]) else stacked_total[idx]
        ax.annotate(
            f"{top:.2f} GB",
            xy=(idx, top),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=peak_color,
        )

    y_candidates = [vram_limit, float(np.nanmax(stacked_total))]
    if not np.isnan(peak_gpu).all():
        y_candidates.append(float(np.nanmax(peak_gpu)))
    y_max = max(y_candidates) * 1.16

    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_ylabel("VRAM Usage (GB)", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.set_axisbelow(True)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if oom_mask.any():
        # Keep the OOM legend marker clear even if matplotlib changes scatter sizing.
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor=oom_color,
                markeredgecolor="white",
                markersize=10,
                label="OOM / Failed Run",
            )
        )
        legend_labels.append("OOM / Failed Run")

    # De-duplicate labels while preserving order.
    deduped: dict[str, Any] = {}
    for handle, label in zip(legend_handles, legend_labels):
        deduped.setdefault(label, handle)
    ax.legend(deduped.values(), deduped.keys(), loc="upper left", frameon=True, fontsize=10)

    # Activation is only tens of MB, so it is almost invisible on a 16GB scale.
    # Keep the main stacked bars quantitatively honest and add a zoom panel.
    activation_mb = activation * 1024
    ax_act.bar(
        x[success_mask],
        activation_mb[success_mask],
        color=activation_color,
        edgecolor="white",
        linewidth=0.8,
    )
    act_ymax = max(32.0, float(np.nanmax(activation_mb[success_mask])) * 1.35)
    ax_act.set_ylim(0, act_ymax)
    ax_act.set_ylabel("Activation\n(MB)", fontsize=10)
    ax_act.set_xlabel("Context Length / Prompt Length (tokens)", fontsize=12)
    ax_act.set_xticks(x)
    ax_act.set_xticklabels(labels, fontsize=10)
    ax_act.grid(axis="y", linestyle="--", alpha=0.32)
    ax_act.set_axisbelow(True)
    ax_act.set_title("Activation Zoom", fontsize=11, fontweight="bold", pad=4)

    for idx, is_success in enumerate(success_mask):
        if is_success:
            ax_act.annotate(
                f"{activation_mb[idx]:.1f}",
                xy=(idx, activation_mb[idx]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=peak_color,
            )
        else:
            ax_act.annotate(
                statuses[idx],
                xy=(idx, act_ymax * 0.55),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=oom_color,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = "#1d1d1f",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>{escape(text)}</text>'
    )


def plot_svg(rows: list[dict[str, Any]], output_path: Path, vram_limit: float, title: str) -> None:
    prompt_lens = [to_int(row["prompt_len"]) for row in rows]
    statuses = [str(row["status"]).strip().upper() for row in rows]

    model_weight = [to_float(row.get("model_weight_gb")) or 0.0 for row in rows]
    kv_cache = [to_float(row.get("kv_cache_gb")) or 0.0 for row in rows]
    activation = [to_float(row.get("activation_gb")) or 0.0 for row in rows]
    peak_gpu = [to_float(row.get("peak_gpu_mem_gb")) for row in rows]
    stacked_total = [m + k + a for m, k, a in zip(model_weight, kv_cache, activation)]

    n = len(rows)
    width = max(1050, 115 * n + 180)
    height = 820
    margin_left = 88
    margin_right = 34
    margin_top = 92
    plot_width = width - margin_left - margin_right
    plot_height = 440
    zoom_gap = 48
    zoom_height = 88
    zoom_top = margin_top + plot_height + zoom_gap
    zoom_axis_y = zoom_top + zoom_height

    max_peak = max([p for p in peak_gpu if p is not None] or [0.0])
    raw_y_max = max(max(stacked_total or [0.0]), max_peak, vram_limit) * 1.14
    tick_step = 2
    y_max = max(tick_step, math.ceil(raw_y_max / tick_step) * tick_step)

    def x_center(i: int) -> float:
        return margin_left + (i + 0.5) * plot_width / n

    def y_pos(value: float) -> float:
        return margin_top + plot_height - (value / y_max) * plot_height

    bar_width = min(62, plot_width / max(n, 1) * 0.58)
    model_color = "#264653"
    kv_color = "#2a9d8f"
    activation_color = "#e9c46a"
    peak_color = "#1d1d1f"
    oom_color = "#c1121f"
    grid_color = "#d7dde2"

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, title, size=21, weight="bold"),
        svg_text(width / 2, 58, "No offload: weights, KV cache, and activations are kept on GPU", size=12, fill="#4a5568"),
    ]

    # Grid and y-axis ticks.
    for tick in range(0, int(y_max) + 1, tick_step):
        y = y_pos(float(tick))
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{grid_color}" stroke-width="1" stroke-dasharray="4 4"/>'
        )
        parts.append(svg_text(margin_left - 12, y + 4, str(tick), size=11, anchor="end", fill="#4a5568"))

    # Axes.
    x_axis_y = margin_top + plot_height
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}" stroke="#1d1d1f" stroke-width="1.4"/>')
    parts.append(f'<line x1="{margin_left}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" stroke="#1d1d1f" stroke-width="1.4"/>')

    # Stacked bars.
    for i, status in enumerate(statuses):
        cx = x_center(i)
        x0 = cx - bar_width / 2
        base = 0.0
        for value, color in (
            (model_weight[i], model_color),
            (kv_cache[i], kv_color),
            (activation[i], activation_color),
        ):
            if value <= 0:
                continue
            y = y_pos(base + value)
            h = value / y_max * plot_height
            parts.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" '
                f'fill="{color}" stroke="white" stroke-width="0.8"/>'
            )
            base += value

        top = peak_gpu[i] if peak_gpu[i] is not None else stacked_total[i]
        if status == "SUCCESS" and top > 0:
            parts.append(svg_text(cx, y_pos(top) - 8, f"{top:.2f} GB", size=10, fill=peak_color))

    # Peak memory line.
    peak_points = [
        (x_center(i), y_pos(float(peak)))
        for i, peak in enumerate(peak_gpu)
        if statuses[i] == "SUCCESS" and peak is not None
    ]
    if peak_points:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in peak_points)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{peak_color}" stroke-width="2.4"/>')
        for x, y in peak_points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{peak_color}"/>')

    # VRAM limit and OOM marker.
    limit_y = y_pos(vram_limit)
    parts.append(
        f'<line x1="{margin_left}" y1="{limit_y:.1f}" x2="{width - margin_right}" y2="{limit_y:.1f}" '
        f'stroke="{oom_color}" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    parts.append(svg_text(width - margin_right - 6, limit_y - 8, f"{vram_limit:.0f} GB limit", size=11, anchor="end", fill=oom_color, weight="bold"))

    for i, status in enumerate(statuses):
        if status == "SUCCESS":
            continue
        cx = x_center(i)
        size = 9
        parts.append(f'<line x1="{cx - size}" y1="{limit_y - size}" x2="{cx + size}" y2="{limit_y + size}" stroke="{oom_color}" stroke-width="4"/>')
        parts.append(f'<line x1="{cx + size}" y1="{limit_y - size}" x2="{cx - size}" y2="{limit_y + size}" stroke="{oom_color}" stroke-width="4"/>')
        parts.append(svg_text(cx, limit_y - 18, status, size=11, weight="bold", fill=oom_color))

    # Activation zoom panel: values are only tens of MB and disappear on the main GB scale.
    activation_mb = [value * 1024 for value in activation]
    max_activation_mb = max(
        [value for value, status in zip(activation_mb, statuses) if status == "SUCCESS"] or [0.0]
    )
    activation_y_max_mb = max(32.0, math.ceil(max_activation_mb / 8.0) * 8.0)

    def zoom_y(value_mb: float) -> float:
        return zoom_top + zoom_height - (value_mb / activation_y_max_mb) * zoom_height

    parts.append(svg_text(width / 2, zoom_top - 18, "Activation Zoom", size=13, weight="bold"))
    for tick in (0.0, activation_y_max_mb / 2.0, activation_y_max_mb):
        y = zoom_y(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="{grid_color}" stroke-width="1" stroke-dasharray="4 4"/>'
        )
        parts.append(svg_text(margin_left - 12, y + 4, f"{tick:.0f}", size=10, anchor="end", fill="#4a5568"))

    parts.append(f'<line x1="{margin_left}" y1="{zoom_top}" x2="{margin_left}" y2="{zoom_axis_y}" stroke="#1d1d1f" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{zoom_axis_y}" x2="{width - margin_right}" y2="{zoom_axis_y}" stroke="#1d1d1f" stroke-width="1.2"/>')

    for i, status in enumerate(statuses):
        cx = x_center(i)
        x0 = cx - bar_width / 2
        if status == "SUCCESS":
            value = activation_mb[i]
            y = zoom_y(value)
            h = max(1.0, value / activation_y_max_mb * zoom_height)
            parts.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" '
                f'fill="{activation_color}" stroke="white" stroke-width="0.8"/>'
            )
            parts.append(svg_text(cx, y - 5, f"{value:.1f}", size=9, fill=peak_color))
            parts.append(svg_text(cx, zoom_axis_y + 24, str(prompt_lens[i]), size=11))
        else:
            parts.append(svg_text(cx, zoom_top + zoom_height * 0.52, status, size=10, weight="bold", fill=oom_color))
            parts.append(svg_text(cx, zoom_axis_y + 22, str(prompt_lens[i]), size=11, fill=oom_color))
            parts.append(svg_text(cx, zoom_axis_y + 39, status, size=10, weight="bold", fill=oom_color))

    # Axis labels.
    parts.append(svg_text(width / 2, height - 28, "Context Length / Prompt Length (tokens)", size=13, weight="bold"))
    parts.append(svg_text(22, margin_top + plot_height / 2, "VRAM Usage (GB)", size=13, weight="bold", rotate=-90))
    parts.append(svg_text(42, zoom_top + zoom_height / 2, "Activation (MB)", size=12, weight="bold", rotate=-90))

    # Legend.
    legend_x = margin_left + 10
    legend_y = margin_top - 24
    legend_items = [
        ("Model Weight", model_color, "rect"),
        ("KV Cache", kv_color, "rect"),
        ("Activation", activation_color, "rect"),
        ("Peak GPU Memory", peak_color, "line"),
        (f"VRAM Limit ({vram_limit:.0f} GB)", oom_color, "dash"),
        ("OOM / Failed Run", oom_color, "x"),
    ]
    cursor_x = legend_x
    for label, color, kind in legend_items:
        if kind == "rect":
            parts.append(f'<rect x="{cursor_x:.1f}" y="{legend_y - 11:.1f}" width="15" height="11" fill="{color}"/>')
        elif kind == "line":
            parts.append(f'<line x1="{cursor_x:.1f}" y1="{legend_y - 5:.1f}" x2="{cursor_x + 18:.1f}" y2="{legend_y - 5:.1f}" stroke="{color}" stroke-width="2.4"/>')
            parts.append(f'<circle cx="{cursor_x + 9:.1f}" cy="{legend_y - 5:.1f}" r="3.6" fill="{color}"/>')
        elif kind == "dash":
            parts.append(f'<line x1="{cursor_x:.1f}" y1="{legend_y - 5:.1f}" x2="{cursor_x + 18:.1f}" y2="{legend_y - 5:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
        else:
            parts.append(f'<line x1="{cursor_x:.1f}" y1="{legend_y - 12:.1f}" x2="{cursor_x + 16:.1f}" y2="{legend_y + 4:.1f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<line x1="{cursor_x + 16:.1f}" y1="{legend_y - 12:.1f}" x2="{cursor_x:.1f}" y2="{legend_y + 4:.1f}" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(cursor_x + 23, legend_y, label, size=11, anchor="start"))
        cursor_x += 23 + len(label) * 7.0 + 20

    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved chart: {output_path}")


def main() -> int:
    args = parse_args()
    rows = read_latest_rows(args.csv)
    plot(rows, args.output, args.vram_limit, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
