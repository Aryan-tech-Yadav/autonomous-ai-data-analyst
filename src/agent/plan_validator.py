from typing import Any, Dict, List


ALLOWED_TOOLS = {
    "pandas_analysis",
    "time_analysis",
    "chart_generator",
    "bar_chart",
    "line_chart",
}


ALLOWED_OPERATIONS = {
    "revenue_calculation",
    "revenue_calculations",
    "groupby_aggregation",
    "group_by",
    "find_max",
    "statistics",
    "calculate_statistics",
    "categorical_analysis",
    "generate_bar_chart",
    "generate_line_chart",
}


DERIVED_REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "total_revenue",
    "calculated revenue",
    "calculated_revenue",
}


def _normalize(value: Any) -> str:
    return str(value).strip().lower()


def _column_exists(
    column: str,
    actual_columns: List[str],
) -> bool:

    normalized = _normalize(column)

    return any(
        _normalize(actual) == normalized
        for actual in actual_columns
    )


def _is_derived_revenue(
    column: str,
) -> bool:

    normalized = _normalize(column)

    return normalized in DERIVED_REVENUE_ALIASES


def _validate_parameter_columns(
    parameters: Dict[str, Any],
    actual_columns: List[str],
    warnings: List[str],
) -> List[str]:

    errors = []

    column_parameters = {
        "column",
        "group_column",
        "value_column",
        "units_column",
        "price_column",
        "category_column",
        "x_column",
        "y_column",
    }

    for parameter_name in column_parameters:

        if parameter_name not in parameters:
            continue

        value = parameters[
            parameter_name
        ]

        if not isinstance(value, str):
            errors.append(
                f"Parameter '{parameter_name}' "
                "must be a string."
            )
            continue

        if _is_derived_revenue(value):

            warnings.append(
                f"Parameter '{parameter_name}' "
                f"references derived revenue "
                f"column '{value}'."
            )

            continue

        if not _column_exists(
            value,
            actual_columns,
        ):

            errors.append(
                f"Column '{value}' referenced by "
                f"parameter '{parameter_name}' "
                "does not exist in the dataset."
            )

    return errors


def _validate_chart_step(
    step: Dict[str, Any],
    actual_columns: List[str],
    warnings: List[str],
) -> List[str]:

    errors = []

    operation = _normalize(
        step.get("operation", "")
    )

    parameters = step.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):

        return [
            f"Chart operation '{operation}' "
            "must contain a parameters object."
        ]

    if operation in {
        "generate_bar_chart",
        "bar_chart",
        "chart_generator",
    }:

        category_column = parameters.get(
            "category_column"
        )

        value_column = parameters.get(
            "value_column"
        )

        if not category_column:
            errors.append(
                "Bar chart requires "
                "'category_column'."
            )

        if not value_column:
            errors.append(
                "Bar chart requires "
                "'value_column'."
            )

        if category_column:

            if not isinstance(
                category_column,
                str,
            ):

                errors.append(
                    "'category_column' must be "
                    "a string."
                )

            elif not _column_exists(
                category_column,
                actual_columns,
            ):

                errors.append(
                    f"Chart category column "
                    f"'{category_column}' does not "
                    "exist in the dataset."
                )

        if value_column:

            if not isinstance(
                value_column,
                str,
            ):

                errors.append(
                    "'value_column' must be "
                    "a string."
                )

            elif _is_derived_revenue(
                value_column
            ):

                warnings.append(
                    f"Chart value column "
                    f"'{value_column}' is a "
                    "derived revenue column."
                )

            elif not _column_exists(
                value_column,
                actual_columns,
            ):

                errors.append(
                    f"Chart value column "
                    f"'{value_column}' does not "
                    "exist in the dataset."
                )

    elif operation in {
        "generate_line_chart",
        "line_chart",
    }:

        x_column = parameters.get(
            "x_column"
        )

        y_column = parameters.get(
            "y_column"
        )

        if not x_column:
            errors.append(
                "Line chart requires "
                "'x_column'."
            )

        if not y_column:
            errors.append(
                "Line chart requires "
                "'y_column'."
            )

        if x_column:

            if not isinstance(
                x_column,
                str,
            ):

                errors.append(
                    "'x_column' must be "
                    "a string."
                )

            elif not _column_exists(
                x_column,
                actual_columns,
            ):

                errors.append(
                    f"Chart X column "
                    f"'{x_column}' does not "
                    "exist in the dataset."
                )

        if y_column:

            if not isinstance(
                y_column,
                str,
            ):

                errors.append(
                    "'y_column' must be "
                    "a string."
                )

            elif _is_derived_revenue(
                y_column
            ):

                warnings.append(
                    f"Chart Y column "
                    f"'{y_column}' is a "
                    "derived revenue column."
                )

            elif not _column_exists(
                y_column,
                actual_columns,
            ):

                errors.append(
                    f"Chart Y column "
                    f"'{y_column}' does not "
                    "exist in the dataset."
                )

    return errors


def validate_plan(
    plan: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:

    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(
        plan,
        dict,
    ):

        return {
            "valid": False,
            "errors": [
                "Plan must be a dictionary."
            ],
            "warnings": [],
            "validated_plan": None,
        }

    analysis_plan = plan.get(
        "analysis_plan"
    )

    if not isinstance(
        analysis_plan,
        list,
    ):

        return {
            "valid": False,
            "errors": [
                "'analysis_plan' must be a list."
            ],
            "warnings": [],
            "validated_plan": None,
        }

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

    validated_steps = []

    for index, step in enumerate(
        analysis_plan,
        start=1,
    ):

        if not isinstance(
            step,
            dict,
        ):

            errors.append(
                f"Step {index} must be "
                "a dictionary."
            )

            continue

        operation = step.get(
            "operation"
        )

        if not operation:

            operation = step.get(
                "type"
            )

        if not operation:

            operation = step.get(
                "sub_tool"
            )

        if not operation:

            operation = step.get(
                "subtype"
            )

        if not operation:

            errors.append(
                f"Step {index} has no "
                "operation."
            )

            continue

        operation = str(
            operation
        ).strip()

        normalized_operation = _normalize(
            operation
        )

        if normalized_operation not in {
            _normalize(item)
            for item in ALLOWED_OPERATIONS
        }:

            errors.append(
                f"Step {index} uses unsupported "
                f"operation '{operation}'."
            )

            continue

        parameters = step.get(
            "parameters",
            {},
        )

        if not isinstance(
            parameters,
            dict,
        ):

            errors.append(
                f"Step {index} parameters "
                "must be a dictionary."
            )

            continue

        step_errors = (
            _validate_parameter_columns(
                parameters=parameters,
                actual_columns=actual_columns,
                warnings=warnings,
            )
        )

        errors.extend(
            [
                f"Step {index}: {error}"
                for error in step_errors
            ]
        )

        if normalized_operation in {
            "generate_bar_chart",
            "bar_chart",
            "chart_generator",
            "generate_line_chart",
            "line_chart",
        }:

            chart_errors = (
                _validate_chart_step(
                    step=step,
                    actual_columns=actual_columns,
                    warnings=warnings,
                )
            )

            errors.extend(
                [
                    f"Step {index}: {error}"
                    for error in chart_errors
                ]
            )

        validated_step = dict(
            step
        )

        validated_step[
            "operation"
        ] = operation

        validated_step[
            "parameters"
        ] = parameters

        validated_steps.append(
            validated_step
        )

    validated_plan = dict(
        plan
    )

    validated_plan[
        "analysis_plan"
    ] = validated_steps

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "validated_plan": (
            validated_plan
            if len(errors) == 0
            else None
        ),
    }
