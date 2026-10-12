import pandas as pd

from src.execution.operation_executor import OperationExecutor


def test_find_max_uses_requested_metric_not_latest_trend():
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01",
            "2026-01-01",
            "2026-02-01",
        ]),
        "Revenue": [100, 200, 900, 300],
        "Unit Price": [10, 50, 20, 30],
    })

    executor = OperationExecutor()

    result = executor.execute(
        df=df,
        execution_steps=[
            {
                "step": 1,
                "operation": "trend_analysis",
                "parameters": {
                    "date_column": "Date",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                },
            },
            {
                "step": 2,
                "operation": "trend_analysis",
                "parameters": {
                    "date_column": "Date",
                    "value_column": "Unit Price",
                    "aggregation": "mean",
                },
            },
            {
                "step": 3,
                "operation": "find_max",
                "parameters": {
                    "column": "Revenue",
                },
            },
        ],
    )

    assert result["status"] == "success"
    assert result["successful_steps"] == 3

    max_result = result["results"][2]["result"]

    assert max_result["value_column"] == "Revenue"
    assert max_result["group"] == "2026-01"
    assert max_result["value"] == 1000
