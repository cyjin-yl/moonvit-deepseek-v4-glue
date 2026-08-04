"""Trajectory charts are deterministic, dependency-free SVG assets."""

from moonvit_glue.svg_charts import heatmap_svg, line_chart_svg, scatter_chart_svg


def test_line_chart_svg_contains_series_and_handles_flat_values():
    svg = line_chart_svg(
        title="Synthetic accuracy",
        x_label="examples seen",
        y_label="accuracy",
        x_values=[0, 4000],
        series={"vision": [0.0, 0.0], "blind": [0.0, 0.0]},
        y_bounds=(0.0, 1.0),
    )
    assert svg.startswith("<svg")
    assert "Synthetic accuracy" in svg
    assert "vision" in svg and "blind" in svg
    assert svg.count("<circle") == 4


def test_heatmap_svg_labels_every_cell_and_escapes_labels():
    svg = heatmap_svg(
        title="Task < checkpoint",
        row_labels=["color", "OCR"],
        column_labels=["step-0", "step-500"],
        values=[[0.0, 0.5], [-0.2, 0.8]],
        value_label="accuracy",
        bounds=(-1.0, 1.0),
    )
    assert "Task &lt; checkpoint" in svg
    assert svg.count("<rect class=\"cell\"") == 4
    assert "color" in svg and "step-500" in svg


def test_scatter_chart_svg_has_one_mark_per_labeled_point():
    svg = scatter_chart_svg(
        title="Evidence vs generation",
        x_label="paired preference",
        y_label="generation accuracy",
        points=[("step-0", 0.2, 0.0), ("step-500", 0.6, 0.1)],
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    assert svg.count("<circle") == 2
    assert "step-0" in svg and "step-500" in svg
