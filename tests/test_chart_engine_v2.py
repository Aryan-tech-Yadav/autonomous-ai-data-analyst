from pathlib import Path

import pandas as pd
import pytest

from src.tools.chart_tool import (
    generate_bar_chart,
    generate_line_chart,
)


def test_line_chart_saves_png_and_sorts_months(tmp_path):
    df = pd.DataFrame({
        "Date": [
            "2026-03-10",
            "2026-01-10",
            "2026-02-10",
        ],
        "Revenue": [300, 100, 200],
    })

    output = tmp_path / "monthly_revenue.png"

    result = generate_line_chart(
        df=df,
        x_column="Date",
        y_column="Revenue",
        output_path=str(output),
        title="Monthly Revenue",
        frequency="monthly",
        aggregation="sum",
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert result["chart_type"] == "line"
    assert result["points"] == 3
    assert [point["date"] for point in result["data"]] == [
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
    ]
    assert [point["value"] for point in result["data"]] == [
        100.0,
        200.0,
        300.0,
    ]


def test_bar_chart_saves_png_and_returns_correct_values(tmp_path):
    df = pd.DataFrame({
        "Region": ["North", "South", "North", "West"],
        "Revenue": [100, 200, 300, 150],
    })

    output = tmp_path / "revenue_by_region.png"

    result = generate_bar_chart(
        df=df,
        category_column="Region",
        value_column="Revenue",
        output_path=str(output),
        title="Revenue by Region",
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert result["chart_type"] == "bar"
    assert result["categories"] == ["North", "South", "West"]
    assert result["values"] == [400.0, 200.0, 150.0]


def test_line_chart_rejects_missing_numeric_values(tmp_path):
    df = pd.DataFrame({
        "Date": ["2026-01-10", "2026-02-10"],
        "Revenue": ["invalid", None],
    })

    with pytest.raises(ValueError, match="No numeric metric column|No valid numeric values|No valid points|No time-series points"):
        generate_line_chart(
            df=df,
            x_column="Date",
            y_column="Revenue",
            output_path=str(tmp_path / "invalid.png"),
        )


def test_bar_chart_calculates_derived_revenue(tmp_path):
    df = pd.DataFrame({
        "Region": ["North", "South", "North"],
        "Units Sold": [2, 3, 4],
        "Unit Price": [10, 20, 5],
        "Revenue": [None, None, None],
    })

    output = tmp_path / "derived_revenue.png"

    result = generate_bar_chart(
        df=df,
        category_column="Region",
        value_column="Revenue",
        output_path=str(output),
        title="Derived Revenue by Region",
    )

    assert output.exists()
    assert result["value_column"] == "Revenue"
    assert result["categories"] == ["South", "North"]
    assert result["values"] == [60.0, 40.0]
