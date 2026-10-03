from typing import Callable, Dict

from src.tools.operations import (
    calculate_revenue,
    groupby_aggregate,
    find_max,
    calculate_statistics,
    categorical_analysis,
)

from src.tools.pandas_tool import pandas_analysis
from src.tools.time_tool import time_analysis
from src.tools.chart_tool import (
    generate_bar_chart,
    generate_line_chart,
)


TOOL_REGISTRY: Dict[str, Callable] = {
    "pandas_analysis": pandas_analysis,
    "time_analysis": time_analysis,
    "chart_generator": generate_bar_chart,
    "bar_chart": generate_bar_chart,
    "line_chart": generate_line_chart,
}


OPERATION_REGISTRY: Dict[str, Callable] = {
    "revenue_calculation": calculate_revenue,
    "revenue_calculations": calculate_revenue,
    "groupby_aggregation": groupby_aggregate,
    "group_by": groupby_aggregate,
    "find_max": find_max,
    "statistics": calculate_statistics,
    "calculate_statistics": calculate_statistics,
    "categorical_analysis": categorical_analysis,
    "generate_bar_chart": generate_bar_chart,
    "generate_line_chart": generate_line_chart,
}


def get_tool(
    tool_name: str,
) -> Callable:

    tool = TOOL_REGISTRY.get(
        tool_name
    )

    if tool is None:

        available_tools = ", ".join(
            TOOL_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown tool: {tool_name}. "
            f"Available tools: {available_tools}"
        )

    return tool


def get_operation(
    operation_name: str,
) -> Callable:

    operation = OPERATION_REGISTRY.get(
        operation_name
    )

    if operation is None:

        available_operations = ", ".join(
            OPERATION_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown operation: "
            f"{operation_name}. "
            f"Available operations: "
            f"{available_operations}"
        )

    return operation


def list_tools() -> list[str]:
    return list(
        TOOL_REGISTRY.keys()
    )


def list_operations() -> list[str]:
    return list(
        OPERATION_REGISTRY.keys()
    )
