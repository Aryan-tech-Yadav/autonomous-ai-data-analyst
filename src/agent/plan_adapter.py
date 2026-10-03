from typing import Any, Dict, List, Optional


DERIVED_REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "total_revenue",
    "calculated revenue",
    "calculated_revenue",
}


def _normalize(value: Any) -> str:
    return str(value).strip().lower()


def _find_actual_column(
    requested: str,
    actual_columns: List[str],
) -> Optional[str]:

    if not requested:
        return None

    requested_normalized = _normalize(
        requested
    )

    # Exact match first.
    for column in actual_columns:

        if _normalize(column) == requested_normalized:
            return column

    return None


def _find_column_by_keywords(
    keywords: List[str],
    actual_columns: List[str],
) -> Optional[str]:

    normalized_columns = [
        (
            column,
            _normalize(column),
        )
        for column in actual_columns
    ]

    # First try exact keyword matches.
    for keyword in keywords:

        keyword_normalized = _normalize(
            keyword
        )

        for column, normalized in normalized_columns:

            if normalized == keyword_normalized:
                return column

    # Then try substring matches.
    for keyword in keywords:

        keyword_normalized = _normalize(
            keyword
        )

        for column, normalized in normalized_columns:

            if keyword_normalized in normalized:
                return column

    return None


def _is_derived_revenue(
    value: Any,
) -> bool:

    return _normalize(
        value
    ) in DERIVED_REVENUE_ALIASES


def _resolve_column(
    requested: Any,
    actual_columns: List[str],
    keywords: Optional[List[str]] = None,
) -> Optional[str]:

    if isinstance(
        requested,
        str,
    ):

        actual = _find_actual_column(
            requested,
            actual_columns,
        )

        if actual:
            return actual

        if _is_derived_revenue(
            requested
        ):

            return requested

    if keywords:

        return _find_column_by_keywords(
            keywords,
            actual_columns,
        )

    return None


def _normalize_operation(
    operation: Any,
) -> str:

    if not operation:
        return ""

    operation = _normalize(
        operation
    )

    aliases = {
        "revenue_calculation":
            "revenue_calculations",

        "revenue_calculations":
            "revenue_calculations",

        "group_by":
            "groupby_aggregation",

        "groupby":
            "groupby_aggregation",

        "groupby_aggregate":
            "groupby_aggregation",

        "calculate_statistics":
            "statistics",

        "categorical":
            "categorical_analysis",

        "bar_chart":
            "generate_bar_chart",

        "chart_generator":
            "generate_bar_chart",

        "line_chart":
            "generate_line_chart",
    }

    return aliases.get(
        operation,
        operation,
    )


def _extract_operation(
    step: Dict[str, Any],
) -> str:

    candidates = [
        step.get("operation"),
        step.get("type"),
        step.get("sub_tool"),
        step.get("subtype"),
    ]

    for candidate in candidates:

        if candidate:
            return str(
                candidate
            ).strip()

    return ""


def _adapt_revenue_calculation(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    units_column = _resolve_column(
        parameters.get(
            "units_column"
        )
        or parameters.get(
            "quantity_column"
        ),
        actual_columns,
        keywords=[
            "Units Sold",
            "Units",
            "Quantity",
        ],
    )

    price_column = _resolve_column(
        parameters.get(
            "price_column"
        )
        or parameters.get(
            "unit_column"
        ),
        actual_columns,
        keywords=[
            "Unit Price",
            "Price",
        ],
    )

    output_column = parameters.get(
        "output_column",
        "Revenue",
    )

    if not units_column:
        raise ValueError(
            "Could not identify the units "
            "column for revenue calculation."
        )

    if not price_column:
        raise ValueError(
            "Could not identify the price "
            "column for revenue calculation."
        )

    return {
        "units_column": units_column,
        "price_column": price_column,
        "output_column": output_column,
    }


def _adapt_groupby(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    group_column = _resolve_column(
        parameters.get(
            "group_column"
        )
        or parameters.get(
            "group_by"
        )
        or parameters.get(
            "group"
        ),
        actual_columns,
        keywords=[
            "Region",
            "Category",
            "Product Category",
            "Sales Rep",
        ],
    )

    value_column = (
        parameters.get(
            "value_column"
        )
        or parameters.get(
            "metric_column"
        )
        or parameters.get(
            "value"
        )
    )

    if _is_derived_revenue(
        value_column
    ):

        value_column = "Revenue"

    else:

        value_column = _resolve_column(
            value_column,
            actual_columns,
            keywords=[
                "Revenue",
                "Total Revenue",
                "Sales",
                "Amount",
            ],
        )

    aggregation = parameters.get(
        "aggregation",
        "sum",
    )

    if not group_column:
        raise ValueError(
            "Could not identify the "
            "group column."
        )

    if not value_column:
        raise ValueError(
            "Could not identify the "
            "value column."
        )

    return {
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": str(
            aggregation
        ).lower().strip(),
    }


def _adapt_find_max(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    group_column = _resolve_column(
        parameters.get(
            "group_column"
        )
        or parameters.get(
            "group_by"
        ),
        actual_columns,
        keywords=[
            "Region",
            "Category",
            "Product Category",
            "Sales Rep",
        ],
    )

    value_column = (
        parameters.get(
            "value_column"
        )
        or parameters.get(
            "metric_column"
        )
        or parameters.get(
            "value"
        )
    )

    if _is_derived_revenue(
        value_column
    ):

        value_column = "Revenue"

    else:

        value_column = _resolve_column(
            value_column,
            actual_columns,
            keywords=[
                "Revenue",
                "Total Revenue",
                "Sales",
                "Amount",
            ],
        )

    if not group_column:
        raise ValueError(
            "Could not identify the "
            "group column."
        )

    if not value_column:
        raise ValueError(
            "Could not identify the "
            "value column."
        )

    return {
        "group_column": group_column,
        "value_column": value_column,
    }


def _adapt_statistics(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    requested = (
        parameters.get(
            "column"
        )
        or parameters.get(
            "value_column"
        )
        or parameters.get(
            "metric_column"
        )
    )

    column = _resolve_column(
        requested,
        actual_columns,
        keywords=[
            "Revenue",
            "Total Revenue",
            "Units Sold",
            "Unit Price",
        ],
    )

    if not column:
        raise ValueError(
            "Could not identify the "
            "statistics column."
        )

    return {
        "column": column,
    }


def _adapt_categorical(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    requested = (
        parameters.get(
            "column"
        )
        or parameters.get(
            "category_column"
        )
    )

    column = _resolve_column(
        requested,
        actual_columns,
        keywords=[
            "Region",
            "Product Category",
            "Sales Rep",
            "Status",
        ],
    )

    if not column:
        raise ValueError(
            "Could not identify the "
            "categorical column."
        )

    return {
        "column": column,
    }


def _adapt_bar_chart(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    category_column = _resolve_column(
        parameters.get(
            "category_column"
        )
        or parameters.get(
            "group_column"
        )
        or parameters.get(
            "x_column"
        ),
        actual_columns,
        keywords=[
            "Region",
            "Product Category",
            "Category",
            "Sales Rep",
            "Status",
        ],
    )

    value_column = (
        parameters.get(
            "value_column"
        )
        or parameters.get(
            "metric_column"
        )
        or parameters.get(
            "y_column"
        )
    )

    if _is_derived_revenue(
        value_column
    ):

        value_column = "Revenue"

    else:

        value_column = _resolve_column(
            value_column,
            actual_columns,
            keywords=[
                "Revenue",
                "Total Revenue",
                "Units Sold",
                "Unit Price",
                "Sales",
                "Amount",
            ],
        )

    if not category_column:
        raise ValueError(
            "Could not identify the "
            "category column for bar chart."
        )

    if not value_column:
        raise ValueError(
            "Could not identify the "
            "value column for bar chart."
        )

    return {
        "category_column": category_column,
        "value_column": value_column,
        "output_path": parameters.get(
            "output_path",
            "reports/chart.png",
        ),
        "title": parameters.get(
            "title",
            "",
        ),
        "ascending": bool(
            parameters.get(
                "ascending",
                False,
            )
        ),
    }


def _adapt_line_chart(
    parameters: Dict[str, Any],
    actual_columns: List[str],
) -> Dict[str, Any]:

    x_column = _resolve_column(
        parameters.get(
            "x_column"
        )
        or parameters.get(
            "category_column"
        ),
        actual_columns,
        keywords=[
            "Date",
            "Time",
            "Month",
            "Year",
        ],
    )

    y_column = (
        parameters.get(
            "y_column"
        )
        or parameters.get(
            "value_column"
        )
        or parameters.get(
            "metric_column"
        )
    )

    if _is_derived_revenue(
        y_column
    ):

        y_column = "Revenue"

    else:

        y_column = _resolve_column(
            y_column,
            actual_columns,
            keywords=[
                "Revenue",
                "Total Revenue",
                "Units Sold",
                "Unit Price",
                "Sales",
            ],
        )

    if not x_column:
        raise ValueError(
            "Could not identify the "
            "X column for line chart."
        )

    if not y_column:
        raise ValueError(
            "Could not identify the "
            "Y column for line chart."
        )

    return {
        "x_column": x_column,
        "y_column": y_column,
        "output_path": parameters.get(
            "output_path",
            "reports/chart.png",
        ),
        "title": parameters.get(
            "title",
            "",
        ),
    }


def adapt_plan(
    plan: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:

    schema = context.get(
        "schema",
        {},
    )

    schema_columns = schema.get(
        "columns",
        [],
    )

    actual_columns = []

    for item in schema_columns:

        if isinstance(
            item,
            dict,
        ):

            name = item.get(
                "name"
            )

            if name:
                actual_columns.append(
                    str(name)
                )

    analysis_plan = plan.get(
        "analysis_plan",
        [],
    )

    execution_steps = []

    for index, step in enumerate(
        analysis_plan,
        start=1,
    ):

        operation = _extract_operation(
            step
        )

        normalized_operation = (
            _normalize_operation(
                operation
            )
        )

        parameters = step.get(
            "parameters",
            {},
        )

        if not isinstance(
            parameters,
            dict,
        ):

            parameters = {}

        if normalized_operation == (
            "revenue_calculations"
        ):

            adapted_parameters = (
                _adapt_revenue_calculation(
                    parameters,
                    actual_columns,
                )
            )

        elif normalized_operation == (
            "groupby_aggregation"
        ):

            adapted_parameters = (
                _adapt_groupby(
                    parameters,
                    actual_columns,
                )
            )

        elif normalized_operation == (
            "find_max"
        ):

            adapted_parameters = (
                _adapt_find_max(
                    parameters,
                    actual_columns,
                )
            )

        elif normalized_operation in {
            "statistics",
            "calculate_statistics",
        }:

            adapted_parameters = (
                _adapt_statistics(
                    parameters,
                    actual_columns,
                )
            )

            normalized_operation = "statistics"

        elif normalized_operation == (
            "categorical_analysis"
        ):

            adapted_parameters = (
                _adapt_categorical(
                    parameters,
                    actual_columns,
                )
            )

        elif normalized_operation == (
            "generate_bar_chart"
        ):

            adapted_parameters = (
                _adapt_bar_chart(
                    parameters,
                    actual_columns,
                )
            )

        elif normalized_operation == (
            "generate_line_chart"
        ):

            adapted_parameters = (
                _adapt_line_chart(
                    parameters,
                    actual_columns,
                )
            )

        else:

            adapted_parameters = parameters

        execution_steps.append(
            {
                "step": index,
                "operation": normalized_operation,
                "parameters": adapted_parameters,
            }
        )

    return {
        "execution_steps": execution_steps
    }
