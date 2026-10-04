from typing import Any


DERIVED_REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "total_revenue",
    "calculated revenue",
    "calculated_revenue",
}

PANDAS_ANALYSIS_OPERATIONS = {
    "revenue_calculations",
    "revenue_calculation",
    "groupby_aggregation",
    "group_by",
    "find_max",
    "statistics",
    "calculate_statistics",
    "categorical_analysis",
    "rank_by_value",
    "ranking",
    "calculate_percentage_change",
    "percentage_change",
    "compare_columns",
    "comparison",
    "create_shifted_column",
    "shift_column",
    "shift",
    "lag",
}


OPERATION_ALIASES = {
    "revenue_calculation": "revenue_calculations",
    "group_by": "groupby_aggregation",
    "calculate_statistics": "statistics",
    "ranking": "rank_by_value",
    "percentage_change": "calculate_percentage_change",
    "comparison": "compare_columns",
    "shift_column": "create_shifted_column",
    "shift": "create_shifted_column",
    "lag": "create_shifted_column",
}


def _normalize(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _canonical_operation(operation: Any) -> str:
    normalized = _normalize(operation)
    return OPERATION_ALIASES.get(normalized, normalized)


def _schema_columns(context: dict[str, Any] | None) -> list[str]:
    if not context:
        return []

    schema = context.get("schema", {})

    if not isinstance(schema, dict):
        return []

    columns = schema.get("columns", [])

    if not isinstance(columns, list):
        return []

    result = []

    for column in columns:
        if isinstance(column, dict) and column.get("name"):
            result.append(str(column["name"]))

    return result


def _find_column(
    requested: Any,
    available_columns: list[str],
    derived_columns: set[str] | None = None,
) -> str:
    if requested is None:
        raise ValueError("Column name is required.")

    requested_text = str(requested).strip()

    if not requested_text:
        raise ValueError("Column name cannot be empty.")

    derived_columns = derived_columns or set()

    all_columns = list(available_columns) + list(derived_columns)

    # Exact match first.
    for column in all_columns:
        if column == requested_text:
            return column

    normalized_requested = _normalize(requested_text)

    # Normalized match.
    for column in all_columns:
        if _normalize(column) == normalized_requested:
            return column

    # Revenue aliases.
    if normalized_requested in {
        _normalize(alias)
        for alias in DERIVED_REVENUE_ALIASES
    }:
        for column in all_columns:
            if _normalize(column) in {
                _normalize(alias)
                for alias in DERIVED_REVENUE_ALIASES
            }:
                return column

        if "Revenue" in derived_columns:
            return "Revenue"

    raise ValueError(
        f"Unable to resolve column '{requested_text}'. "
        f"Available columns: {all_columns}"
    )


def _tool_for_operation(operation: str) -> str:
    canonical = _canonical_operation(operation)

    if canonical in PANDAS_ANALYSIS_OPERATIONS:
        return "pandas_analysis"

    if canonical in {
        "generate_bar_chart",
        "generate_line_chart",
    }:
        return "chart_generator"

    if canonical in {
        "time_analysis",
    }:
        return "time_analysis"

    return "pandas_analysis"


def _adapt_revenue_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
) -> dict[str, Any]:
    units_column = parameters.get(
        "units_column",
        parameters.get("quantity_column"),
    )

    price_column = parameters.get(
        "price_column",
        parameters.get("unit_price_column"),
    )

    output_column = parameters.get(
        "output_column",
        parameters.get("target_column", "Revenue"),
    )

    units_column = _find_column(
        units_column,
        available_columns,
    )

    price_column = _find_column(
        price_column,
        available_columns,
    )

    return {
        "units_column": units_column,
        "price_column": price_column,
        "output_column": output_column,
    }


def _adapt_groupby_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    group_column = parameters.get(
        "group_column",
        parameters.get("group_by"),
    )

    value_column = parameters.get(
        "value_column",
        parameters.get("metric_column"),
    )

    aggregation = parameters.get(
        "aggregation",
        parameters.get("agg", "sum"),
    )

    group_column = _find_column(
        group_column,
        available_columns,
        derived_columns,
    )

    value_column = _find_column(
        value_column,
        available_columns,
        derived_columns,
    )

    return {
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": aggregation,
    }


def _adapt_find_max_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    group_column = parameters.get(
        "group_column",
        parameters.get("group_by"),
    )

    value_column = parameters.get(
        "value_column",
        parameters.get("metric_column"),
    )

    return {
        "group_column": _find_column(
            group_column,
            available_columns,
            derived_columns,
        ),
        "value_column": _find_column(
            value_column,
            available_columns,
            derived_columns,
        ),
    }


def _adapt_rank_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    group_column = parameters.get(
        "group_column",
        parameters.get("group_by"),
    )

    value_column = parameters.get(
        "value_column",
        parameters.get("metric_column"),
    )

    aggregation = parameters.get(
        "aggregation",
        parameters.get("agg", "sum"),
    )

    ascending = parameters.get("ascending", False)

    top_n = parameters.get(
        "top_n",
        parameters.get("n"),
    )

    adapted = {
        "group_column": _find_column(
            group_column,
            available_columns,
            derived_columns,
        ),
        "value_column": _find_column(
            value_column,
            available_columns,
            derived_columns,
        ),
        "aggregation": aggregation,
        "ascending": bool(ascending),
    }

    if top_n is not None:
        adapted["top_n"] = int(top_n)

    return adapted


def _adapt_percentage_change_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    current_column = parameters.get(
        "current_column",
        parameters.get(
            "new_column",
            parameters.get("current"),
        ),
    )

    previous_column = parameters.get(
        "previous_column",
        parameters.get(
            "old_column",
            parameters.get(
                "previous",
                parameters.get("prior_column"),
            ),
        ),
    )

    output_column = parameters.get(
        "output_column",
        parameters.get(
            "target_column",
            "Percentage Change",
        ),
    )

    return {
        "current_column": _find_column(
            current_column,
            available_columns,
            derived_columns,
        ),
        "previous_column": _find_column(
            previous_column,
            available_columns,
            derived_columns,
        ),
        "output_column": output_column,
    }


def _adapt_comparison_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    left_column = parameters.get(
        "left_column",
        parameters.get("first_column"),
    )

    right_column = parameters.get(
        "right_column",
        parameters.get("second_column"),
    )

    return {
        "left_column": _find_column(
            left_column,
            available_columns,
            derived_columns,
        ),
        "right_column": _find_column(
            right_column,
            available_columns,
            derived_columns,
        ),
    }


def _adapt_shifted_column_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    source_column = parameters.get(
        "source_column",
        parameters.get(
            "column",
            parameters.get(
                "value_column",
                parameters.get("current_column"),
            ),
        ),
    )

    output_column = parameters.get(
        "output_column",
        parameters.get(
            "new_column",
            parameters.get(
                "target_column",
                parameters.get(
                    "name",
                    None,
                ),
            ),
        ),
    )

    if output_column is None:
        resolved_source = _find_column(
            source_column,
            available_columns,
            derived_columns,
        )
        output_column = f"Previous {resolved_source}"

    sort_column = parameters.get(
        "sort_column",
        parameters.get(
            "order_by",
            parameters.get("sort_by"),
        ),
    )

    periods = parameters.get(
        "periods",
        parameters.get(
            "period",
            parameters.get(
                "shift",
                1,
            ),
        ),
    )

    ascending = parameters.get(
        "ascending",
        True,
    )

    adapted = {
        "source_column": _find_column(
            source_column,
            available_columns,
            derived_columns,
        ),
        "output_column": str(output_column),
        "periods": int(periods),
        "ascending": bool(ascending),
    }

    if sort_column is not None:
        adapted["sort_column"] = _find_column(
            sort_column,
            available_columns,
            derived_columns,
        )
    else:
        adapted["sort_column"] = None

    return adapted


def _adapt_statistics_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    column = parameters.get(
        "column",
        parameters.get("value_column"),
    )

    return {
        "column": _find_column(
            column,
            available_columns,
            derived_columns,
        )
    }


def _adapt_categorical_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    column = parameters.get(
        "column",
        parameters.get("category_column"),
    )

    return {
        "column": _find_column(
            column,
            available_columns,
            derived_columns,
        )
    }


def _adapt_chart_parameters(
    parameters: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    adapted = dict(parameters)

    for key in (
        "x_column",
        "y_column",
        "group_column",
        "value_column",
        "column",
    ):
        if key in adapted and adapted[key] is not None:
            adapted[key] = _find_column(
                adapted[key],
                available_columns,
                derived_columns,
            )

    return adapted


def _register_derived_column(
    operation: str,
    parameters: dict[str, Any],
    derived_columns: set[str],
) -> None:
    canonical = _canonical_operation(operation)

    if canonical == "revenue_calculations":
        output_column = parameters.get(
            "output_column",
            "Revenue",
        )
        derived_columns.add(str(output_column))

    elif canonical == "create_shifted_column":
        output_column = parameters.get("output_column")

        if output_column:
            derived_columns.add(str(output_column))

    elif canonical == "calculate_percentage_change":
        output_column = parameters.get(
            "output_column",
            "Percentage Change",
        )
        derived_columns.add(str(output_column))


def _adapt_step(
    step: dict[str, Any],
    available_columns: list[str],
    derived_columns: set[str],
) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError("Each analysis step must be a dictionary.")

    operation = _canonical_operation(
        step.get("operation")
    )

    if not operation:
        raise ValueError("Analysis step is missing operation.")

    raw_parameters = step.get(
        "parameters",
        {},
    )

    if not isinstance(raw_parameters, dict):
        raise ValueError(
            f"Parameters for operation '{operation}' must be an object."
        )

    if operation == "revenue_calculations":
        parameters = _adapt_revenue_parameters(
            raw_parameters,
            available_columns,
        )

    elif operation == "groupby_aggregation":
        parameters = _adapt_groupby_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "find_max":
        parameters = _adapt_find_max_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "rank_by_value":
        parameters = _adapt_rank_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "calculate_percentage_change":
        parameters = _adapt_percentage_change_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "compare_columns":
        parameters = _adapt_comparison_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "create_shifted_column":
        parameters = _adapt_shifted_column_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "statistics":
        parameters = _adapt_statistics_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation == "categorical_analysis":
        parameters = _adapt_categorical_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    elif operation in {
        "generate_bar_chart",
        "generate_line_chart",
    }:
        parameters = _adapt_chart_parameters(
            raw_parameters,
            available_columns,
            derived_columns,
        )

    else:
        parameters = dict(raw_parameters)

    adapted_step = {
        "step": step.get(
            "step",
            1,
        ),
        "operation": operation,
        "tool": step.get(
            "tool",
            _tool_for_operation(operation),
        ),
        "description": step.get(
            "description",
            "",
        ),
        "parameters": parameters,
    }

    _register_derived_column(
        operation,
        parameters,
        derived_columns,
    )

    return adapted_step


def adapt_plan(
    plan: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert a validated LLM analysis plan into the internal
    execution-plan format.

    `context` is accepted by design so the adapter can resolve
    columns using the current dataset schema.
    """

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dictionary.")

    analysis_plan = plan.get(
        "analysis_plan",
        [],
    )

    if not isinstance(analysis_plan, list):
        raise ValueError(
            "'analysis_plan' must be a list."
        )

    available_columns = _schema_columns(context)

    # Fallback for callers that do not provide context.
    if not available_columns:
        for step in analysis_plan:
            parameters = step.get(
                "parameters",
                {},
            )

            if not isinstance(parameters, dict):
                continue

            for key in (
                "group_column",
                "value_column",
                "units_column",
                "price_column",
                "column",
                "source_column",
                "current_column",
                "previous_column",
                "left_column",
                "right_column",
                "sort_column",
            ):
                value = parameters.get(key)

                if isinstance(value, str):
                    if value not in available_columns:
                        available_columns.append(value)

    derived_columns: set[str] = set()

    execution_steps = []

    for index, step in enumerate(
        analysis_plan,
        start=1,
    ):
        normalized_step = dict(step)

        if "step" not in normalized_step:
            normalized_step["step"] = index

        adapted_step = _adapt_step(
            normalized_step,
            available_columns,
            derived_columns,
        )

        execution_steps.append(
            adapted_step
        )

    return {
        "analysis_plan": analysis_plan,
        "execution_steps": execution_steps,
        "derived_columns": sorted(
            derived_columns
        ),
    }
