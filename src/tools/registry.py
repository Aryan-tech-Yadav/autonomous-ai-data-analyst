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
    trend_analysis,
    generate_bar_chart,
    generate_line_chart,
)


OPERATION_REGISTRY = {
    "calculate_revenue": calculate_revenue,
    "revenue_calculation": calculate_revenue,
    "revenue_calculations": calculate_revenue,

    "create_shifted_column": create_shifted_column,
    "shift_column": create_shifted_column,

    "groupby_aggregation": groupby_aggregate,
    "group_by": groupby_aggregate,
    "groupby": groupby_aggregate,
    "groupby_aggregate": groupby_aggregate,

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

    "trend_analysis": trend_analysis,
}


def get_operation(operation_name: str):
    """
    Return the registered operation function.

    Raises:
        ValueError: If the requested operation does not exist.
    """
    if operation_name not in OPERATION_REGISTRY:
        available = ", ".join(OPERATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown operation: {operation_name}. "
            f"Available operations: {available}"
        )

    return OPERATION_REGISTRY[operation_name]


def get_tool(tool_name: str):
    """
    Backward-compatible alias for get_operation().

    Older agent modules use get_tool(), while newer
    execution code uses get_operation().
    """
    return get_operation(tool_name)
