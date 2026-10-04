from __future__ import annotations

from typing import Callable, Dict, Any

from src.tools.operations import (
    calculate_revenue,
    create_shifted_column,
    groupby_aggregate,
    find_max,
    calculate_statistics,
    categorical_analysis,
    rank_by_value,
    calculate_percentage_change,
    compare_columns,
    generate_bar_chart,
    generate_line_chart,
)

from src.tools.pandas_tool import pandas_analysis
from src.tools.time_tool import time_analysis


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY: Dict[str, Callable] = {
    "pandas_analysis": pandas_analysis,
    "time_analysis": time_analysis,
    "chart_generator": generate_bar_chart,
    "bar_chart": generate_bar_chart,
    "line_chart": generate_line_chart,
}


# ============================================================
# OPERATION REGISTRY
# ============================================================

OPERATION_REGISTRY: Dict[str, Callable] = {
    "calculate_revenue": calculate_revenue,
    "revenue_calculation": calculate_revenue,
    "revenue_calculations": calculate_revenue,

    "create_shifted_column": create_shifted_column,
    "shift_column": create_shifted_column,

    "groupby_aggregation": groupby_aggregate,
    "group_by": groupby_aggregate,

    "find_max": find_max,

    "statistics": calculate_statistics,
    "calculate_statistics": calculate_statistics,

    "categorical_analysis": categorical_analysis,

    "generate_bar_chart": generate_bar_chart,
    "generate_line_chart": generate_line_chart,

    "rank_by_value": rank_by_value,
    "ranking": rank_by_value,

    "calculate_percentage_change": calculate_percentage_change,
    "percentage_change": calculate_percentage_change,

    "compare_columns": compare_columns,
    "comparison": compare_columns,
}


# ============================================================
# TOOL CAPABILITIES
# ============================================================

TOOL_CAPABILITIES: Dict[str, Dict[str, Any]] = {

    "pandas_analysis": {
        "description": (
            "General-purpose dataframe analysis for "
            "inspection, calculations, transformations, "
            "aggregation, ranking, comparison, and "
            "derived metrics."
        ),
        "use_when": [
            "general dataset analysis",
            "custom dataframe calculations",
            "data inspection",
            "filtering",
            "business metric calculations",
        ],
        "operations": [
            "revenue_calculations",
            "create_shifted_column",
            "groupby_aggregation",
            "find_max",
            "statistics",
            "categorical_analysis",
            "rank_by_value",
            "calculate_percentage_change",
            "compare_columns",
        ],
    },

    "time_analysis": {
        "description": (
            "Analyzes date and time columns and can prepare "
            "daily, weekly, monthly, quarterly, or yearly "
            "time-series data."
        ),
        "use_when": [
            "trend analysis",
            "revenue over time",
            "sales over time",
            "monthly analysis",
            "weekly analysis",
            "date-based analysis",
        ],
        "operations": [
            "time_analysis",
        ],
    },

    "revenue_calculations": {
        "description": (
            "Calculates revenue, including derived revenue "
            "from Units Sold multiplied by Unit Price."
        ),
        "use_when": [
            "revenue",
            "sales value",
            "total sales",
            "business revenue",
        ],
        "operations": [
            "revenue_calculations",
        ],
    },

    "groupby_aggregation": {
        "description": (
            "Groups data by a categorical column and "
            "calculates an aggregate metric."
        ),
        "use_when": [
            "revenue by region",
            "sales by product",
            "sales by representative",
            "category comparison",
            "regional comparison",
            "group-level metrics",
        ],
        "operations": [
            "groupby_aggregation",
        ],
    },

    "find_max": {
        "description": (
            "Finds the maximum value or highest-performing "
            "category in a metric."
        ),
        "use_when": [
            "highest",
            "maximum",
            "best performing",
            "top performer",
        ],
        "operations": [
            "find_max",
        ],
    },

    "statistics": {
        "description": (
            "Calculates descriptive statistics for numeric columns."
        ),
        "use_when": [
            "average",
            "mean",
            "median",
            "minimum",
            "maximum",
            "standard statistics",
            "numeric summary",
        ],
        "operations": [
            "statistics",
        ],
    },

    "categorical_analysis": {
        "description": (
            "Analyzes categorical columns, including "
            "value counts and category distributions."
        ),
        "use_when": [
            "category distribution",
            "status distribution",
            "product distribution",
            "region distribution",
            "count by category",
        ],
        "operations": [
            "categorical_analysis",
        ],
    },

    "shift_column": {
        "description": (
            "Creates a previous or next value column by "
            "shifting a source column. Can sort by a date "
            "or ordering column before shifting."
        ),
        "use_when": [
            "previous value",
            "next value",
            "previous period",
            "previous order",
            "consecutive values",
            "period-over-period",
            "prior value",
        ],
        "operations": [
            "create_shifted_column",
            "shift_column",
        ],
    },

    "rank_by_value": {
        "description": (
            "Ranks categories according to an aggregated "
            "numeric metric."
        ),
        "use_when": [
            "rank regions by revenue",
            "top products",
            "top sales representatives",
            "highest revenue categories",
            "lowest revenue categories",
            "top performers",
            "bottom performers",
            "ranking",
        ],
        "operations": [
            "rank_by_value",
            "ranking",
        ],
    },

    "percentage_change": {
        "description": (
            "Calculates percentage change between current "
            "and previous numeric values."
        ),
        "use_when": [
            "percentage change",
            "percent change",
            "growth percentage",
            "growth rate",
            "increase percentage",
            "decrease percentage",
            "change from previous period",
            "period-over-period change",
        ],
        "operations": [
            "calculate_percentage_change",
            "percentage_change",
        ],
    },

    "comparison": {
        "description": (
            "Compares two numeric columns row by row."
        ),
        "use_when": [
            "compare two metrics",
            "compare columns",
            "which is greater",
            "which metric is higher",
            "column comparison",
            "metric comparison",
        ],
        "operations": [
            "compare_columns",
            "comparison",
        ],
    },

    "generate_bar_chart": {
        "description": (
            "Generates a bar chart for categorical "
            "comparisons and ranked business metrics."
        ),
        "use_when": [
            "bar chart",
            "category comparison chart",
            "regional comparison chart",
            "top performers chart",
            "sales by category chart",
        ],
        "operations": [
            "generate_bar_chart",
        ],
    },

    "generate_line_chart": {
        "description": (
            "Generates an aggregated line chart for "
            "time-series trends."
        ),
        "use_when": [
            "line chart",
            "trend chart",
            "revenue over time",
            "sales over time",
            "monthly trend",
            "time-series visualization",
        ],
        "operations": [
            "generate_line_chart",
        ],
    },
}


# ============================================================
# REGISTRY ACCESS FUNCTIONS
# ============================================================

def get_tool(tool_name: str) -> Callable:

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        available_tools = ", ".join(
            TOOL_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown tool: {tool_name}. "
            f"Available tools: {available_tools}"
        )

    return tool


def get_operation(operation_name: str) -> Callable:

    operation = OPERATION_REGISTRY.get(
        operation_name
    )

    if operation is None:
        available_operations = ", ".join(
            OPERATION_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown operation: {operation_name}. "
            f"Available operations: {available_operations}"
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


def list_capabilities() -> dict:
    return TOOL_CAPABILITIES.copy()


def get_capability(
    capability_name: str,
) -> dict:

    capability = TOOL_CAPABILITIES.get(
        capability_name
    )

    if capability is None:
        available_capabilities = ", ".join(
            TOOL_CAPABILITIES.keys()
        )

        raise ValueError(
            f"Unknown capability: {capability_name}. "
            f"Available capabilities: "
            f"{available_capabilities}"
        )

    return capability
