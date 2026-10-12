import pandas as pd

from src.execution.operation_executor import OperationExecutor


def test_line_charts_use_the_matching_trend_metric():
    executor = OperationExecutor()

    trend_results = [
        {
            "operation": "trend_analysis",
            "status": "success",
            "result": {
                "date_column": "Date",
                "value_column": "Revenue",
                "results": [
                    {"period": "2026-01", "value": 1000},
                    {"period": "2026-02", "value": 800},
                ],
            },
        },
        {
            "operation": "trend_analysis",
            "status": "success",
            "result": {
                "date_column": "Date",
                "value_column": "Units Sold",
                "results": [
                    {"period": "2026-01", "value": 100},
                    {"period": "2026-02", "value": 80},
                ],
            },
        },
        {
            "operation": "trend_analysis",
            "status": "success",
            "result": {
                "date_column": "Date",
                "value_column": "Unit Price",
                "results": [
                    {"period": "2026-01", "value": 10},
                    {"period": "2026-02", "value": 8},
                ],
            },
        },
    ]

    executor.results = trend_results

    cases = [
        ("Revenue", [1000, 800]),
        ("Units Sold", [100, 80]),
        ("Unit Price", [10, 8]),
    ]

    for metric, expected_values in cases:
        chart_df, chart_result = executor._execute_trend_line_chart(
            {
                "y_column": metric,
                "title": f"{metric} Trend",
            }
        )

        assert chart_df["Value"].tolist() == expected_values
        assert chart_result["title"] == f"{metric} Trend"
        assert chart_result["data"][0]["y"] == expected_values[0]
        assert chart_result["data"][1]["y"] == expected_values[1]
