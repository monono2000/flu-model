from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import AGE_GROUPS, REGIME_COLORS, REGIONS
from .metrics import SimulationTables
from .simulation import SimulationResult


BACKGROUND = (255, 255, 255)
AXIS_COLOR = (40, 40, 40)
GRID_COLOR = (220, 220, 220)
TEXT_COLOR = (20, 20, 20)
SERIES_COLORS = ["#c0392b", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def create_all_plots(result: SimulationResult, tables: SimulationTables, run_dir: Path) -> None:
    plot_timeseries_overview(tables, run_dir)
    plot_region_comparison(tables, run_dir)
    plot_age_group_comparison(tables, run_dir)
    plot_regime_timeline(result, run_dir)
    create_network_snapshots(result, run_dir)


def save_line_chart_image(
    output_path: Path,
    title: str,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    x_label: str = "Day",
    size: tuple[int, int] = (960, 560),
) -> None:
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_line_chart(draw, (40, 30, size[0] - 40, size[1] - 30), title, x_values, series, y_label, x_label)
    image.save(output_path)


def save_grouped_bar_chart_image(
    output_path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    size: tuple[int, int] = (960, 560),
    highlight_last_category: bool = False,
) -> None:
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_grouped_bar_chart(
        draw,
        (40, 30, size[0] - 40, size[1] - 30),
        title,
        categories,
        series,
        y_label,
        highlight_last_category=highlight_last_category,
    )
    image.save(output_path)


def plot_timeseries_overview(tables: SimulationTables, run_dir: Path) -> None:
    overall = tables.overall_daily_metrics
    x_values = np.arange(1, len(overall) + 1)
    reinfection_share = np.divide(
        overall["total_new_reinfections"].to_numpy(dtype=float),
        overall["total_new_infections"].to_numpy(dtype=float),
        out=np.zeros(len(overall), dtype=float),
        where=overall["total_new_infections"].to_numpy(dtype=float) > 0.0,
    )

    image = Image.new("RGB", (1400, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    rects = _panel_layout((1400, 900), rows=2, cols=2, margin=40, gap=24)

    _draw_line_chart(
        draw,
        rects[0],
        "?꾩껜 ?좉퇋 媛먯뿼",
        x_values,
        [("new infections", overall["total_new_infections"].to_numpy(dtype=float), SERIES_COLORS[0])],
        "Count",
        "Day",
    )
    _draw_line_chart(
        draw,
        rects[1],
        "?꾩껜 ?쒖꽦 媛먯뿼",
        x_values,
        [("active infected", overall["total_active_infected"].to_numpy(dtype=float), SERIES_COLORS[1])],
        "Count",
        "Day",
    )
    _draw_line_chart(
        draw,
        rects[2],
        "?꾩껜 ?뚮났 ?곹깭??,
        x_values,
        [("recovered", overall["total_recovered"].to_numpy(dtype=float), SERIES_COLORS[2])],
        "Count",
        "Day",
    )
    _draw_line_chart(
        draw,
        rects[3],
        "?쇱씪 ?ш컧??鍮꾩쨷",
        x_values,
        [("reinfection share", reinfection_share, SERIES_COLORS[3])],
        "Share",
        "Day",
    )
    image.save(run_dir / "timeseries_overview.png")


def plot_region_comparison(tables: SimulationTables, run_dir: Path) -> None:
    region_df = tables.region_daily_metrics
    image = Image.new("RGB", (1360, 1080), BACKGROUND)
    draw = ImageDraw.Draw(image)
    rects = _panel_layout((1360, 1080), rows=3, cols=1, margin=40, gap=24)

    series_active = []
    series_ever = []
    series_reinf = []
    for color, region in zip(["#1f77b4", "#c0392b"], REGIONS):
        subset = region_df[region_df["region"] == region].reset_index(drop=True)
        x_values = np.arange(1, len(subset) + 1)
        series_active.append((region, subset["active_infected_per_100k"].to_numpy(dtype=float), color))
        series_ever.append((region, subset["ever_infected_rate"].to_numpy(dtype=float), color))
        series_reinf.append((region, subset["reinfection_share"].to_numpy(dtype=float), color))

    _draw_line_chart(draw, rects[0], "?쒖슱 vs ?먯＜ ?쒖꽦 媛먯뿼??(per 100k)", x_values, series_active, "per 100k", "Day")
    _draw_line_chart(draw, rects[1], "?쒖슱 vs ?먯＜ ever infected rate", x_values, series_ever, "Rate", "Day")
    _draw_line_chart(draw, rects[2], "?쒖슱 vs ?먯＜ ?쇱씪 ?ш컧??鍮꾩쨷", x_values, series_reinf, "Share", "Day")
    image.save(run_dir / "region_comparison.png")


def plot_age_group_comparison(tables: SimulationTables, run_dir: Path) -> None:
    age_summary = tables.age_group_summary
    categories = age_summary["age_group"].tolist()
    ever_values = age_summary["ever_infected_rate"].to_numpy(dtype=float)
    episode_values = age_summary["infection_episode_rate"].to_numpy(dtype=float)

    image = Image.new("RGB", (1300, 600), BACKGROUND)
    draw = ImageDraw.Draw(image)
    rects = _panel_layout((1300, 600), rows=1, cols=2, margin=40, gap=24)

    _draw_grouped_bar_chart(
        draw,
        rects[0],
        "?곕졊吏묐떒蹂??꾩쟻 ever infected rate",
        categories,
        [("ever", ever_values, "#3b82f6")],
        "Rate",
        highlight_last_category=True,
    )
    _draw_grouped_bar_chart(
        draw,
        rects[1],
        "?곕졊吏묐떒蹂??꾩쟻 infection episode rate",
        categories,
        [("episode", episode_values, "#ef4444")],
        "Rate",
        highlight_last_category=True,
    )
    image.save(run_dir / "age_group_comparison.png")


def plot_regime_timeline(result: SimulationResult, run_dir: Path) -> None:
    width, height = 1400, 220
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(24)
    label_font = _load_font(16)
    draw.text((40, 20), "Regime Timeline", fill=TEXT_COLOR, font=title_font)

    band_left, band_top, band_right, band_bottom = 40, 90, width - 40, 150
    band_width = band_right - band_left
    total_days = max(len(result.daily_regimes), 1)
    for idx, regime in enumerate(result.daily_regimes):
        x0 = band_left + int(idx / total_days * band_width)
        x1 = band_left + int((idx + 1) / total_days * band_width)
        draw.rectangle([x0, band_top, x1, band_bottom], fill=REGIME_COLORS[regime], outline=None)

    start_idx = 0
    while start_idx < len(result.daily_regimes):
        regime = result.daily_regimes[start_idx]
        end_idx = start_idx + 1
        while end_idx < len(result.daily_regimes) and result.daily_regimes[end_idx] == regime:
            end_idx += 1
        center_x = band_left + int(((start_idx + end_idx) / 2.0) / total_days * band_width)
        draw.text((center_x - 8, band_top - 24), regime, fill=TEXT_COLOR, font=label_font)
        start_idx = end_idx

    draw.rectangle([band_left, band_top, band_right, band_bottom], outline=AXIS_COLOR, width=1)
    draw.text((40, 170), "x異뺤? ?쒕??덉씠???쒖꽌 湲곗? day index", fill=TEXT_COLOR, font=label_font)
    image.save(run_dir / "regime_timeline.png")


def create_network_snapshots(result: SimulationResult, run_dir: Path) -> None:
    snapshot_dir = run_dir / "network_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    indices = select_snapshot_indices(result)
    if not indices:
        return

    active_history = result.state_history["I0"][1:] + result.state_history["I1"][1:]
    ever_history = result.population[np.newaxis, :, :] - result.state_history["S0"][1:]
    ever_rate = np.divide(
        ever_history,
        result.population[np.newaxis, :, :],
        out=np.zeros_like(ever_history),
        where=result.population[np.newaxis, :, :] > 0.0,
    )

    saved_paths: list[Path] = []
    for day_idx, tag in indices:
        output_path = snapshot_dir / f"{day_idx + 1:03d}_{tag}.png"
        draw_network_snapshot(
            output_path=output_path,
            label=result.daily_labels[day_idx],
            regime=result.daily_regimes[day_idx],
            active_per_100k=np.divide(
                active_history[day_idx],
                result.population,
                out=np.zeros_like(active_history[day_idx]),
                where=result.population > 0.0,
            )
            * 100000.0,
            burden=ever_rate[day_idx],
            flow_total=result.flow_total[day_idx],
        )
        saved_paths.append(output_path)

    if len(saved_paths) > 1:
        frames = [Image.open(path) for path in saved_paths]
        frames[0].save(
            run_dir / "network_animation.gif",
            save_all=True,
            append_images=frames[1:],
            duration=850,
            loop=0,
        )


def draw_network_snapshot(
    output_path: Path,
    label: str,
    regime: str,
    active_per_100k: np.ndarray,
    burden: np.ndarray,
    flow_total: np.ndarray,
) -> None:
    width, height = 1100, 760
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(24)
    label_font = _load_font(16)
    small_font = _load_font(14)

    draw.text((40, 20), f"Network Snapshot | {label} | Regime {regime}", fill=TEXT_COLOR, font=title_font)
    draw.text(
        (40, 52),
        "?몃뱶 ?ш린: active infected per 100k / ?몃뱶 梨꾩?: ever infected rate / ?몃뱶 ?뚮몢由? self-flow",
        fill=TEXT_COLOR,
        font=small_font,
    )

    positions = {
        ("Region_A", "0-18"): (260, 170),
        ("Region_A", "19-49"): (260, 300),
        ("Region_A", "50-64"): (260, 430),
        ("Region_A", "65+"): (260, 560),
        ("Region_B", "0-18"): (830, 170),
        ("Region_B", "19-49"): (830, 300),
        ("Region_B", "50-64"): (830, 430),
        ("Region_B", "65+"): (830, 560),
    }

    max_edge = float(flow_total.max()) if flow_total.size else 0.0
    edge_threshold = max_edge * 0.08
    max_self = max(float(flow_total[r, a, r, a]) for r in range(len(REGIONS)) for a in range(len(AGE_GROUPS)))
    node_radius = {}

    for region_idx, region in enumerate(REGIONS):
        for age_idx, age_group in enumerate(AGE_GROUPS):
            size_value = float(active_per_100k[region_idx, age_idx])
            radius = 26 + int(
                26 * (
                    size_value / max(float(active_per_100k.max()), 1.0e-9)
                    if active_per_100k.max() > 0
                    else 0.0
                )
            )
            node_radius[(region, age_group)] = radius

    for source_region_idx, source_region in enumerate(REGIONS):
        for source_age_idx, source_age_group in enumerate(AGE_GROUPS):
            for target_region_idx, target_region in enumerate(REGIONS):
                for target_age_idx, target_age_group in enumerate(AGE_GROUPS):
                    if source_region_idx == target_region_idx and source_age_idx == target_age_idx:
                        continue
                    flow_value = float(
                        flow_total[
                            source_region_idx,
                            source_age_idx,
                            target_region_idx,
                            target_age_idx,
                        ]
                    )
                    if flow_value <= edge_threshold:
                        continue

                    start = positions[(source_region, source_age_group)]
                    end = positions[(target_region, target_age_group)]
                    start_radius = node_radius[(source_region, source_age_group)]
                    end_radius = node_radius[(target_region, target_age_group)]
                    width_scale = 2 + int(8 * (flow_value / max_edge if max_edge > 0 else 0.0))
                    _draw_arrow(draw, start, end, start_radius, end_radius, (110, 110, 110), width_scale)

    for region_idx, region in enumerate(REGIONS):
        for age_idx, age_group in enumerate(AGE_GROUPS):
            center = positions[(region, age_group)]
            radius = node_radius[(region, age_group)]
            burden_value = float(burden[region_idx, age_idx])
            fill = _interpolate_color((255, 245, 230), (180, 35, 20), min(max(burden_value, 0.0), 1.0))
            self_flow = float(flow_total[region_idx, age_idx, region_idx, age_idx])
            border_width = 2 + int(7 * (self_flow / max_self if max_self > 0 else 0.0))
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, fill=fill, outline=(40, 40, 40), width=border_width)

            label_text = f"{region}\n{age_group}"
            text_width, text_height = _text_size(draw, label_text, label_font, multiline=True)
            draw.multiline_text(
                (center[0] - text_width / 2, center[1] - text_height / 2),
                label_text,
                fill=TEXT_COLOR,
                font=label_font,
                align="center",
                spacing=2,
            )

    _draw_color_legend(draw, (945, 110, 1010, 310), "Ever\ninfected\nrate")
    draw.text((945, 340), "x=Region_A / Region_B 怨좎젙", fill=TEXT_COLOR, font=small_font)
    draw.text((945, 362), "y=?곕졊 ?쒖꽌 怨좎젙", fill=TEXT_COLOR, font=small_font)
    image.save(output_path)


def select_snapshot_indices(result: SimulationResult) -> list[tuple[int, str]]:
    if not result.daily_labels:
        return []

    total_active = (result.state_history["I0"][1:] + result.state_history["I1"][1:]).sum(axis=(1, 2))
    indices = [(0, "start")]

    rise_candidates = np.where(total_active >= max(float(total_active.max()) * 0.25, 1.0e-9))[0]
    if rise_candidates.size:
        indices.append((int(rise_candidates[0]), "initial_rise"))

    for regime, tag in [("II", "vacation_start"), ("IV", "lunar_holiday")]:
        for idx, current_regime in enumerate(result.daily_regimes):
            if current_regime == regime:
                indices.append((idx, tag))
                break

    peak_idx = int(np.argmax(total_active))
    for delta, tag in [(-1, "pre_peak"), (0, "peak"), (1, "post_peak")]:
        candidate = peak_idx + delta
        if 0 <= candidate < len(result.daily_labels):
            indices.append((candidate, tag))

    seen = set()
    deduplicated = []
    for idx, tag in indices:
        if idx in seen:
            continue
        deduplicated.append((idx, tag))
        seen.add(idx)
    return deduplicated


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    x_label: str,
) -> None:
    left, top, right, bottom = rect
    title_font = _load_font(20)
    label_font = _load_font(15)
    small_font = _load_font(13)
    draw.rounded_rectangle(rect, radius=18, outline=(210, 210, 210), width=1)
    draw.text((left + 18, top + 12), title, fill=TEXT_COLOR, font=title_font)

    plot_left = left + 62
    plot_top = top + 52
    plot_right = right - 24
    plot_bottom = bottom - 42
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS_COLOR, width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS_COLOR, width=1)

    y_max = max(max(float(np.max(values)), 0.0) for _, values, _ in series)
    y_max = y_max * 1.08 if y_max > 0 else 1.0
    tick_count = 5
    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        y = plot_bottom - int((plot_bottom - plot_top) * fraction)
        value = y_max * fraction
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID_COLOR, width=1)
        text = _format_value(value)
        text_width, text_height = _text_size(draw, text, small_font)
        draw.text((plot_left - 10 - text_width, y - text_height / 2), text, fill=TEXT_COLOR, font=small_font)

    if len(x_values) <= 1:
        x_positions = np.array([plot_left + (plot_right - plot_left) / 2.0])
    else:
        x_positions = np.linspace(plot_left, plot_right, num=len(x_values))

    tick_indices = np.linspace(0, len(x_values) - 1, num=min(6, len(x_values)), dtype=int)
    for idx in tick_indices:
        x = int(x_positions[idx])
        draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=AXIS_COLOR, width=1)
        text = str(int(x_values[idx]))
        text_width, _ = _text_size(draw, text, small_font)
        draw.text((x - text_width / 2, plot_bottom + 8), text, fill=TEXT_COLOR, font=small_font)

    legend_x = plot_right - 180
    legend_y = top + 14
    for idx, (name, values, color) in enumerate(series):
        line_y = legend_y + idx * 18
        draw.line([(legend_x, line_y + 8), (legend_x + 18, line_y + 8)], fill=color, width=3)
        draw.text((legend_x + 24, line_y), name, fill=TEXT_COLOR, font=small_font)
        points = []
        for point_idx, value in enumerate(values):
            x = float(x_positions[point_idx])
            y = plot_bottom - (float(value) / y_max) * (plot_bottom - plot_top) if y_max > 0 else plot_bottom
            points.append((x, y))
        if len(points) == 1:
            draw.ellipse(
                [points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2],
                fill=color,
                outline=color,
            )
        else:
            draw.line(points, fill=color, width=3, joint="curve")

    draw.text((left + 14, plot_top - 4), y_label, fill=TEXT_COLOR, font=label_font)
    draw.text((plot_right - 25, plot_bottom + 24), x_label, fill=TEXT_COLOR, font=label_font)


def _draw_grouped_bar_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    categories: list[str],
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    highlight_last_category: bool = False,
) -> None:
    left, top, right, bottom = rect
    title_font = _load_font(20)
    label_font = _load_font(15)
    small_font = _load_font(13)
    draw.rounded_rectangle(rect, radius=18, outline=(210, 210, 210), width=1)
    draw.text((left + 18, top + 12), title, fill=TEXT_COLOR, font=title_font)

    plot_left = left + 62
    plot_top = top + 52
    plot_right = right - 24
    plot_bottom = bottom - 52
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS_COLOR, width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS_COLOR, width=1)

    all_values = np.concatenate([values for _, values, _ in series]) if series else np.array([0.0])
    y_max = max(float(np.max(all_values)), 0.0)
    y_max = y_max * 1.12 if y_max > 0 else 1.0
    for tick in range(6):
        fraction = tick / 5.0
        y = plot_bottom - int((plot_bottom - plot_top) * fraction)
        value = y_max * fraction
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID_COLOR, width=1)
        text = _format_value(value)
        text_width, text_height = _text_size(draw, text, small_font)
        draw.text((plot_left - 10 - text_width, y - text_height / 2), text, fill=TEXT_COLOR, font=small_font)

    group_width = (plot_right - plot_left) / max(len(categories), 1)
    bar_width = max(18, int(group_width / max(len(series) * 1.8, 2)))
    for cat_idx, category in enumerate(categories):
        group_center = plot_left + group_width * (cat_idx + 0.5)
        if highlight_last_category and cat_idx == len(categories) - 1:
            draw.rectangle(
                [
                    group_center - group_width / 2 + 4,
                    plot_top,
                    group_center + group_width / 2 - 4,
                    plot_bottom,
                ],
                fill=(245, 245, 225),
            )
        for series_idx, (_, values, color) in enumerate(series):
            value = float(values[cat_idx])
            x0 = group_center - (len(series) * bar_width) / 2 + series_idx * bar_width
            x1 = x0 + bar_width - 4
            y1 = plot_bottom
            y0 = plot_bottom - (value / y_max) * (plot_bottom - plot_top) if y_max > 0 else plot_bottom
            outline = (40, 40, 40) if highlight_last_category and cat_idx == len(categories) - 1 else color
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=2 if outline != color else 1)
        text_width, _ = _text_size(draw, category, small_font)
        draw.text((group_center - text_width / 2, plot_bottom + 8), category, fill=TEXT_COLOR, font=small_font)

    legend_x = plot_right - 180
    legend_y = top + 14
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 18
        draw.rectangle([legend_x, y + 2, legend_x + 14, y + 14], fill=color, outline=color)
        draw.text((legend_x + 20, y), name, fill=TEXT_COLOR, font=small_font)

    draw.text((left + 14, plot_top - 4), y_label, fill=TEXT_COLOR, font=label_font)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    start_radius: int,
    end_radius: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(math.hypot(dx, dy), 1.0)
    ux = dx / distance
    uy = dy / distance
    line_start = (start[0] + ux * start_radius, start[1] + uy * start_radius)
    line_end = (end[0] - ux * end_radius, end[1] - uy * end_radius)
    draw.line([line_start, line_end], fill=color, width=width)

    arrow_length = 12 + width
    arrow_width = 5 + width
    tip = line_end
    left = (
        tip[0] - arrow_length * ux + arrow_width * uy,
        tip[1] - arrow_length * uy - arrow_width * ux,
    )
    right = (
        tip[0] - arrow_length * ux - arrow_width * uy,
        tip[1] - arrow_length * uy + arrow_width * ux,
    )
    draw.polygon([tip, left, right], fill=color)


def _draw_color_legend(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str) -> None:
    left, top, right, bottom = rect
    font = _load_font(14)
    height = bottom - top
    for idx in range(height):
        fraction = idx / max(height - 1, 1)
        color = _interpolate_color((255, 245, 230), (180, 35, 20), fraction)
        draw.line([(left, bottom - idx), (right, bottom - idx)], fill=color, width=1)
    draw.rectangle(rect, outline=AXIS_COLOR, width=1)
    draw.multiline_text((left, top - 46), title, fill=TEXT_COLOR, font=font, spacing=2, align="left")
    draw.text((right + 8, top - 4), "1.0", fill=TEXT_COLOR, font=font)
    draw.text((right + 8, bottom - 12), "0.0", fill=TEXT_COLOR, font=font)


def _panel_layout(size: tuple[int, int], rows: int, cols: int, margin: int, gap: int) -> list[tuple[int, int, int, int]]:
    width, height = size
    panel_width = (width - margin * 2 - gap * (cols - 1)) // cols
    panel_height = (height - margin * 2 - gap * (rows - 1)) // rows
    rects = []
    for row in range(rows):
        for col in range(cols):
            left = margin + col * (panel_width + gap)
            top = margin + row * (panel_height + gap)
            rects.append((left, top, left + panel_width, top + panel_height))
    return rects


def _format_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    multiline: bool = False,
) -> tuple[int, int]:
    if multiline:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _interpolate_color(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    fraction: float,
) -> tuple[int, int, int]:
    fraction = min(max(fraction, 0.0), 1.0)
    return tuple(
        int(start_rgb[idx] + (end_rgb[idx] - start_rgb[idx]) * fraction)
        for idx in range(3)
    )

