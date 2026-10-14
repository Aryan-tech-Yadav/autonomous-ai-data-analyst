import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.tools.operations import (
    generate_bar_chart,
    generate_line_chart,
)


def test_bar_chart_aggregates_and_sorts_categories(tmp_path):
    df = pd.DataFrame({
        "Region": ["North", "South", "North", "West"],
        "Revenue": [100, 200, 300, 150],
    })

    result = generate_bar_chart(
        df,
        "Region",
        "Revenue",
        output_path=str(tmp_path / "bar.png"),
    )

    assert result["chart_type"] == "bar"
    assert result["data"] == [
        {"x": "North", "y": 400.0},
        {"x": "South", "y": 200.0},
        {"x": "West", "y": 150.0},
    ]
    assert (tmp_path / "bar.png").is_file()


def test_bar_chart_respects_top_n_and_ascending(tmp_path):
    df = pd.DataFrame({
        "Region": ["A", "B", "C"],
        "Revenue": [30, 10, 20],
    })

    result = generate_bar_chart(
        df,
        "Region",
        "Revenue",
        top_n=2,
        ascending=True,
        output_path=str(tmp_path / "top.png"),
    )

    assert result["data"] == [
        {"x": "B", "y": 10.0},
        {"x": "C", "y": 20.0},
    ]


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True])
def test_bar_chart_rejects_invalid_top_n(tmp_path, top_n):
    df = pd.DataFrame({
        "Region": ["A", "B"],
        "Revenue": [10, 20],
    })

    with pytest.raises(ValueError, match="top_n"):
        generate_bar_chart(
            df,
            "Region",
            "Revenue",
            top_n=top_n,
            output_path=str(tmp_path / "invalid.png"),
        )


@pytest.mark.parametrize(
    "chart_function",
    [generate_bar_chart, generate_line_chart],
)
def test_charts_reject_unsupported_aggregation(
    tmp_path, chart_function
):
    df = pd.DataFrame({
        "Category": ["A", "B"],
        "Value": [10, 20],
    })

    with pytest.raises(ValueError, match="Unsupported chart aggregation"):
        chart_function(
            df,
            "Category",
            "Value",
            aggregation="invalid",
            output_path=str(tmp_path / "invalid.png"),
        )


@pytest.mark.parametrize(
    "chart_function",
    [generate_bar_chart, generate_line_chart],
)
def test_charts_reject_data_without_valid_numeric_values(
    tmp_path, chart_function
):
    df = pd.DataFrame({
        "Category": ["A", "B"],
        "Value": ["invalid", None],
    })

    with pytest.raises(ValueError, match="No valid data"):
        chart_function(
            df,
            "Category",
            "Value",
            output_path=str(tmp_path / "empty.png"),
        )


def test_line_chart_aggregates_and_sorts_x_values(tmp_path):
    df = pd.DataFrame({
        "Month": ["March", "January", "January", "February"],
        "Revenue": [300, 100, 50, 200],
    })

    result = generate_line_chart(
        df,
        "Month",
        "Revenue",
        output_path=str(tmp_path / "line.png"),
    )

    assert result["chart_type"] == "line"
    assert result["data"] == [
        {"x": "February", "y": 200.0},
        {"x": "January", "y": 150.0},
        {"x": "March", "y": 300.0},
    ]
    assert (tmp_path / "line.png").is_file()


def test_chart_generation_closes_matplotlib_figures(tmp_path):
    df = pd.DataFrame({
        "Category": ["A", "B"],
        "Value": [10, 20],
    })

    existing_figures = set(plt.get_fignums())

    generate_bar_chart(
        df,
        "Category",
        "Value",
        output_path=str(tmp_path / "resource-check.png"),
    )

    assert set(plt.get_fignums()) == existing_figures


def test_automatic_chart_filename_does_not_overwrite_existing_files(
    tmp_path, monkeypatch
):
    import src.tools.operations as operations

    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()

    (reports / "chart_1.png").write_bytes(b"existing chart one")
    (reports / "chart_3.png").write_bytes(b"existing chart three")

    df = pd.DataFrame({
        "Category": ["A", "B"],
        "Value": [10, 20],
    })

    result = operations.generate_bar_chart(
        df,
        "Category",
        "Value",
    )

    assert result["output_path"] == str(reports / "chart_4.png")
    assert (reports / "chart_1.png").read_bytes() == b"existing chart one"
    assert (reports / "chart_3.png").read_bytes() == b"existing chart three"
    assert (reports / "chart_4.png").is_file()


def test_chart_figure_closes_when_saving_fails(
    tmp_path, monkeypatch
):
    import src.tools.operations as operations

    df = pd.DataFrame({
        "Category": ["A", "B"],
        "Value": [10, 20],
    })

    existing_figures = set(plt.get_fignums())

    def fail_save(*args, **kwargs):
        raise OSError("Simulated chart save failure")

    monkeypatch.setattr(operations, "_save_chart", fail_save)

    with pytest.raises(OSError, match="Simulated chart save failure"):
        operations.generate_bar_chart(
            df,
            "Category",
            "Value",
            output_path=str(tmp_path / "failure.png"),
        )

    assert set(plt.get_fignums()) == existing_figures
