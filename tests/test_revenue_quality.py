import pandas as pd
import pytest

from src.execution.operation_executor import OperationExecutor


def test_calculates_revenue_when_column_is_missing():
    df = pd.DataFrame({
        "Units Sold": [2, 3],
        "Unit Price": [100, 50],
    })

    result = OperationExecutor()._ensure_revenue_column(df)

    assert result["Revenue"].tolist() == [200, 150]


def test_preserves_existing_revenue_and_fills_missing_values():
    df = pd.DataFrame({
        "Units Sold": [2, 3, 4],
        "Unit Price": [100, 50, 25],
        "Revenue": [None, 999, None],
    })

    result = OperationExecutor()._ensure_revenue_column(df)

    assert result["Revenue"].tolist() == [200, 999, 100]


def test_missing_inputs_remain_missing():
    df = pd.DataFrame({
        "Units Sold": [None, 2],
        "Unit Price": [100, None],
    })

    result = OperationExecutor()._ensure_revenue_column(df)

    assert result["Revenue"].isna().all()


def test_does_not_modify_original_dataframe():
    df = pd.DataFrame({
        "Units Sold": [2],
        "Unit Price": [100],
        "Revenue": [None],
    })
    original = df.copy(deep=True)

    OperationExecutor()._ensure_revenue_column(df)

    pd.testing.assert_frame_equal(df, original)


def test_missing_required_column_raises_value_error():
    df = pd.DataFrame({"Units Sold": [2]})

    with pytest.raises(ValueError, match="Price column"):
        OperationExecutor()._ensure_revenue_column(df)
