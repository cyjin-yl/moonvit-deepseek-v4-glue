"""不依赖绘图库的轻量确定性 SVG 图表。"""

from __future__ import annotations

import html
from typing import Mapping, Sequence


PALETTE = (
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#4f46e5", "#be123c", "#65a30d", "#7c3aed",
)


def line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: Sequence[float | int],
    series: Mapping[str, Sequence[float | int]],
    y_bounds: tuple[float, float] | None = None,
    width: int = 900,
    height: int = 480,
) -> str:
    if not x_values or not series:
        raise ValueError("line chart needs x values and at least one series")
    if any(len(values) != len(x_values) for values in series.values()):
        raise ValueError("every series must align with x_values")
    left, right, top, bottom = 85, width - 190, 55, height - 70
    x_min, x_max = float(min(x_values)), float(max(x_values))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    all_y = [float(value) for values in series.values() for value in values]
    if y_bounds is None:
        y_min, y_max = min(all_y), max(all_y)
        if y_min == y_max:
            padding = max(0.05, abs(y_min) * 0.1)
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding
    else:
        y_min, y_max = (float(value) for value in y_bounds)
        if y_min >= y_max:
            raise ValueError("y_bounds must increase")

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}.axis{stroke:#4b5563;stroke-width:1.4}.grid{stroke:#d9dee7;stroke-width:1}.line{fill:none;stroke-width:2.5}</style>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y_value = y_min + (y_max - y_min) * fraction
        y = py(y_value)
        chunks.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        chunks.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{y_value:.3f}</text>')
    for tick in range(min(6, len(x_values))):
        index = round(tick * (len(x_values) - 1) / max(1, min(6, len(x_values)) - 1))
        x_value = float(x_values[index])
        x = px(x_value)
        chunks.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>')
        chunks.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-size="12">{x_value:g}</text>')
    chunks.extend([
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>',
        f'<text x="20" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 20 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="14">{html.escape(y_label)}</text>',
    ])
    for series_index, (name, values) in enumerate(series.items()):
        color = PALETTE[series_index % len(PALETTE)]
        points = " ".join(
            f"{px(float(x)):.2f},{py(float(y)):.2f}"
            for x, y in zip(x_values, values)
        )
        chunks.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
        for x, y in zip(x_values, values):
            chunks.append(f'<circle cx="{px(float(x)):.2f}" cy="{py(float(y)):.2f}" r="4" fill="{color}"/>')
        legend_y = top + series_index * 24
        chunks.append(f'<line x1="{right + 20}" y1="{legend_y}" x2="{right + 45}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<text x="{right + 52}" y="{legend_y + 4}" font-size="12">{html.escape(str(name))}</text>')
    chunks.append("</svg>\n")
    return "".join(chunks)


def heatmap_svg(
    *,
    title: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[float | int]],
    value_label: str,
    bounds: tuple[float, float],
    width: int = 900,
) -> str:
    """把带标签的紧凑数值矩阵渲染成确定性 SVG。"""

    if not row_labels or not column_labels or len(values) != len(row_labels):
        raise ValueError("heatmap labels and rows must be non-empty and aligned")
    if any(len(row) != len(column_labels) for row in values):
        raise ValueError("every heatmap row must align with column labels")
    low, high = (float(value) for value in bounds)
    if low >= high:
        raise ValueError("heatmap bounds must increase")
    left, top, right, cell_height = 170, 95, 35, 50
    cell_width = (width - left - right) / len(column_labels)
    height = top + cell_height * len(row_labels) + 70

    def color(value: float) -> str:
        clipped = min(high, max(low, value))
        fraction = (clipped - low) / (high - low)
        # 静态报告采用统一蓝色强度，数值文字保留精确方向。
        lightness = 96.0 - 56.0 * fraction
        return f"hsl(217 74% {lightness:.1f}%)"

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}.cell{stroke:#d9dee7;stroke-width:1}.value{font-size:12px;font-weight:700}</style>',
        f'<title>{html.escape(title)}</title>',
        f'<desc>{html.escape(value_label)} by row and checkpoint</desc>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for column_index, label in enumerate(column_labels):
        x = left + (column_index + 0.5) * cell_width
        chunks.append(
            f'<text x="{x:.2f}" y="72" text-anchor="middle" font-size="12">{html.escape(str(label))}</text>'
        )
    for row_index, (label, row) in enumerate(zip(row_labels, values)):
        y = top + row_index * cell_height
        chunks.append(
            f'<text x="{left - 12}" y="{y + cell_height / 2 + 4:.2f}" text-anchor="end" font-size="13">{html.escape(str(label))}</text>'
        )
        for column_index, raw_value in enumerate(row):
            value = float(raw_value)
            x = left + column_index * cell_width
            chunks.append(
                f'<rect class="cell" x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height}" fill="{color(value)}"/>'
            )
            chunks.append(
                f'<text class="value" x="{x + cell_width / 2:.2f}" y="{y + cell_height / 2 + 4:.2f}" text-anchor="middle">{value:.3f}</text>'
            )
    chunks.append(
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="12">{html.escape(value_label)}; fixed scale [{low:g}, {high:g}]</text>'
    )
    chunks.append("</svg>\n")
    return "".join(chunks)


def scatter_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: Sequence[tuple[str, float | int, float | int]],
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    width: int = 900,
    height: int = 520,
) -> str:
    """为 checkpoint 比较渲染直接标注的散点图。"""

    if not points:
        raise ValueError("scatter chart needs at least one point")
    left, right, top, bottom = 90, width - 55, 55, height - 75

    def limits(values: list[float], supplied: tuple[float, float] | None):
        if supplied is not None:
            low, high = (float(value) for value in supplied)
        else:
            low, high = min(values), max(values)
            padding = max(0.05, (high - low) * 0.1)
            low, high = low - padding, high + padding
        if low >= high:
            raise ValueError("scatter bounds must increase")
        return low, high

    xs = [float(point[1]) for point in points]
    ys = [float(point[2]) for point in points]
    x_low, x_high = limits(xs, x_bounds)
    y_low, y_high = limits(ys, y_bounds)

    def px(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}.axis{stroke:#4b5563;stroke-width:1.4}.grid{stroke:#d9dee7;stroke-width:1}</style>',
        f'<title>{html.escape(title)}</title>',
        f'<desc>{html.escape(x_label)} against {html.escape(y_label)}, labeled by checkpoint</desc>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_low + (x_high - x_low) * fraction
        y_value = y_low + (y_high - y_low) * fraction
        x, y = px(x_value), py(y_value)
        chunks.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>')
        chunks.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        chunks.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-size="12">{x_value:.3f}</text>')
        chunks.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{y_value:.3f}</text>')
    chunks.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<text x="{(left + right) / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 20 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    for index, (label, raw_x, raw_y) in enumerate(points):
        x, y = px(float(raw_x)), py(float(raw_y))
        color = PALETTE[index % len(PALETTE)]
        chunks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"/>')
        chunks.append(
            f'<text x="{x + 9:.2f}" y="{y - 8:.2f}" font-size="12">{html.escape(str(label))}</text>'
        )
    chunks.append("</svg>\n")
    return "".join(chunks)
