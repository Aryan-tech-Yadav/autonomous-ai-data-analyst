import pandas as pd

from src.data.profiler import profile_dataset


def test_profile_reports_missing_counts_and_percentages():
    df = pd.DataFrame({
        "Revenue": [100, None, 300, None],
        "Region": ["East", "West", None, "East"],
    })

    profile = profile_dataset(df)

    assert profile["rows"] == 4
    assert profile["columns"] == 2
    assert profile["missing_values"] == 3
    assert profile["missing_percentage"] == 37.5

    details = {item["name"]: item for item in profile["column_details"]}
    assert details["Revenue"]["missing"] == 2
    assert details["Revenue"]["missing_percentage"] == 50.0
    assert details["Region"]["missing"] == 1


def test_profile_identifies_columns_with_missing_values():
    df = pd.DataFrame({
        "Complete": [1, 2, 3, 4],
        "High Missing": [None, None, 3, 4],
    })

    profile = profile_dataset(df)

    assert len(profile["columns_with_missing_values"]) == 1
    assert profile["columns_with_missing_values"][0]["name"] == "High Missing"
    assert profile["columns_with_missing_values"][0]["priority"] == "high"
    assert profile["data_quality"]["status"] == "needs_attention"


def test_complete_dataset_has_good_quality_status():
    df = pd.DataFrame({
        "Region": ["East", "West"],
        "Revenue": [100, 200],
    })

    profile = profile_dataset(df)

    assert profile["missing_values"] == 0
    assert profile["missing_percentage"] == 0.0
    assert profile["columns_with_missing_values"] == []
    assert profile["data_quality"]["status"] == "good"


def test_empty_dataframe_is_supported():
    profile = profile_dataset(pd.DataFrame())

    assert profile["rows"] == 0
    assert profile["columns"] == 0
    assert profile["missing_percentage"] == 0.0
    assert profile["data_quality"]["status"] == "needs_attention"
