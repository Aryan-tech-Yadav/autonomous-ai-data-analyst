import pandas as pd

from src.tools.operations import (
    rank_by_value,
    calculate_percentage_change,
    compare_columns,
)


def test_rank_by_value():
    df = pd.DataFrame({
        "Region": ["North", "South", "North", "West", "South"],
        "Revenue": [100, 250, 150, 80, 300],
    })

    result = rank_by_value(
        df=df,
        group_column="Region",
        value_column="Revenue",
        aggregation="sum",
        ascending=False,
        top_n=3,
    )

    assert result["operation"] == "rank_by_value"
    assert result["results"][0]["group"] == "South"
    assert result["results"][0]["value"] == 550.0
    assert result["results"][0]["rank"] == 1


def test_calculate_percentage_change():
    df = pd.DataFrame({
        "Current": [120, 220, 180, 100, 330],
        "Previous": [100, 200, 150, 80, 300],
    })

    result = calculate_percentage_change(
        df=df,
        current_column="Current",
        previous_column="Previous",
        output_column="Percentage Change",
    )

    assert result["operation"] == "calculate_percentage_change"
    assert result["rows_calculated"] == 5
    assert result["mean_percentage_change"] == 17.0
    assert result["minimum_percentage_change"] == 10.0
    assert result["maximum_percentage_change"] == 25.0


def test_compare_columns():
    df = pd.DataFrame({
        "Current": [120, 220, 180, 100, 330],
        "Previous": [100, 200, 150, 80, 300],
    })

    result = compare_columns(
        df=df,
        left_column="Current",
        right_column="Previous",
    )

    assert result["operation"] == "compare_columns"
    assert result["valid_comparisons"] == 5
    assert result["left_greater_count"] == 5
    assert result["right_greater_count"] == 0
    assert result["equal_count"] == 0
