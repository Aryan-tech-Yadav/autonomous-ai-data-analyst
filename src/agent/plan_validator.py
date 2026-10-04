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

    # Advanced analytics
    "rank_by_value",
    "ranking",
    "calculate_percentage_change",
    "percentage_change",
    "compare_columns",
    "comparison",

    # Shift / lag
    "create_shifted_column",
    "shift_column",
    "shift",
    "lag",

    # Charts
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


SHIFTED_COLUMN_OPERATIONS = {
    "create_shifted_column",
    "shift_column",
    "shift",
    "lag",
}


def _normalize(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


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

    normalized = str(column).strip().lower()

    return normalized in DERIVED_REVENUE_ALIASES


def _validate_parameter_columns(
    parameters: Dict[str, Any],
    actual_columns: List[str],
    warnings: List[str],
    derived_columns: set[str] | None = None,
) -> List[str]:

    errors = []

    if derived_columns is None:
        derived_columns = set()

    column_parameters = {
        "column",
        "group_column",
        "value_column",
        "units_column",
        "price_column",
        "category_column",
        "x_column",
        "y_column",
        "current_column",
        "previous_column",
        "left_column",
        "right_column",
        "source_column",
        "sort_column",
    }

    for parameter_name in column_parameters:

        if parameter_name not in parameters:
            continue

        value = parameters[parameter_name]

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

        if _column_exists(
            value,
            actual_columns,
        ):

            continue

        if _column_exists(
            value,
            list(derived_columns),
        ):

            warnings.append(
                f"Parameter '{parameter_name}' "
                f"references a column '{value}' "
                "created by an earlier plan step."
            )

            continue

        errors.append(
            f"Column '{value}' referenced by "
            f"parameter '{parameter_name}' "
            "does not exist in the dataset "
            "or as a previously derived column."
        )

    return errors


def _validate_shift_step(
    parameters: Dict[str, Any],
    actual_columns: List[str],
    derived_columns: set[str],
    warnings: List[str],
) -> List[str]:

    errors = []

    source_column = parameters.get(
        "source_column"
    )

    if not source_column:
        source_column = parameters.get(
            "column"
        )

    if not source_column:

        errors.append(
            "Shift operation requires "
            "'source_column'."
        )

    elif not isinstance(
        source_column,
        str,
    ):

        errors.append(
            "'source_column' must be a string."
        )

    elif not (
        _column_exists(
            source_column,
            actual_columns,
        )
        or _column_exists(
            source_column,
            list(derived_columns),
        )
    ):

        errors.append(
            f"Shift source column "
            f"'{source_column}' does not exist."
        )

    output_column = parameters.get(
        "output_column"
    )

    if not output_column:
        output_column = parameters.get(
            "new_column"
        )

    if not output_column:

        errors.append(
            "Shift operation requires "
            "'output_column'."
        )

    elif not isinstance(
        output_column,
        str,
    ):

        errors.append(
            "'output_column' must be a string."
        )

    else:

        if (
            _column_exists(
                output_column,
                actual_columns,
            )
        ):

            warnings.append(
                f"Shift output column "
                f"'{output_column}' already exists "
                "and may be overwritten."
            )

        derived_columns.add(
            output_column
        )

    sort_column = parameters.get(
        "sort_column"
    )

    if sort_column is not None:

        if not isinstance(
            sort_column,
            str,
        ):

            errors.append(
                "'sort_column' must be a string."
            )

        elif not _column_exists(
            sort_column,
            actual_columns,
        ):

            errors.append(
                f"Sort column "
                f"'{sort_column}' does not exist."
            )

    periods = parameters.get(
        "periods",
        1,
    )

    if not isinstance(
        periods,
        int,
    ):

        errors.append(
            "'periods' must be an integer."
        )

    elif periods == 0:

        errors.append(
            "'periods' cannot be zero."
        )

    return errors


def _validate_percentage_change_step(
    parameters: Dict[str, Any],
    actual_columns: List[str],
    derived_columns: set[str],
) -> List[str]:

    errors = []

    current_column = parameters.get(
        "current_column"
    )

    previous_column = parameters.get(
        "previous_column"
    )

    if not current_column:

        errors.append(
            "Percentage change requires "
            "'current_column'."
        )

    elif not isinstance(
        current_column,
        str,
    ):

        errors.append(
            "'current_column' must be a string."
        )

    elif not (
        _column_exists(
            current_column,
            actual_columns,
        )
        or _column_exists(
            current_column,
            list(derived_columns),
        )
        or _is_derived_revenue(
            current_column
        )
    ):

        errors.append(
            f"Current column "
            f"'{current_column}' does not exist."
        )

    if not previous_column:

        errors.append(
            "Percentage change requires "
            "'previous_column'."
        )

    elif not isinstance(
        previous_column,
        str,
    ):

        errors.append(
            "'previous_column' must be a string."
        )

    elif not (
        _column_exists(
            previous_column,
            actual_columns,
        )
        or _column_exists(
            previous_column,
            list(derived_columns),
        )
        or _is_derived_revenue(
            previous_column
        )
    ):

        errors.append(
            f"Previous column "
            f"'{previous_column}' does not exist "
            "or has not been created by an "
            "earlier plan step."
        )

    return errors


def _validate_chart_step(
    step: Dict[str, Any],
    actual_columns: List[str],
    warnings: List[str],
    derived_columns: set[str] | None = None,
) -> List[str]:

    errors = []

    if derived_columns is None:
        derived_columns = set()

    operation = _normalize(
        step.get("operation", "")
    )

    parameters = step.get(
        "parameters",
        {},
    )

    if not isinstance(
        parameters,
        dict,
    ):

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

            elif not (
                _column_exists(
                    category_column,
                    actual_columns,
                )
                or _column_exists(
                    category_column,
                    list(derived_columns),
                )
            ):

                errors.append(
                    f"Chart category column "
                    f"'{category_column}' does not "
                    "exist."
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

            elif not (
                _column_exists(
                    value_column,
                    actual_columns,
                )
                or _column_exists(
                    value_column,
                    list(derived_columns),
                )
            ):

                errors.append(
                    f"Chart value column "
                    f"'{value_column}' does not "
                    "exist."
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

            elif not (
                _column_exists(
                    x_column,
                    actual_columns,
                )
                or _column_exists(
                    x_column,
                    list(derived_columns),
                )
            ):

                errors.append(
                    f"Chart X column "
                    f"'{x_column}' does not "
                    "exist."
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

            elif not (
                _column_exists(
                    y_column,
                    actual_columns,
                )
                or _column_exists(
                    y_column,
                    list(derived_columns),
                )
            ):

                errors.append(
                    f"Chart Y column "
                    f"'{y_column}' does not "
                    "exist."
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

    actual_columns: List[str] = []

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

    # Columns created by earlier execution
    # steps in this same plan.
    derived_columns: set[str] = set()

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

        allowed_normalized_operations = {
            _normalize(item)
            for item in ALLOWED_OPERATIONS
        }

        if (
            normalized_operation
            not in allowed_normalized_operations
        ):

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

        # --------------------------------------------------
        # Shift / lag
        # --------------------------------------------------

        if normalized_operation in {
            _normalize(item)
            for item in SHIFTED_COLUMN_OPERATIONS
        }:

            step_errors = _validate_shift_step(
                parameters=parameters,
                actual_columns=actual_columns,
                derived_columns=derived_columns,
                warnings=warnings,
            )

        # --------------------------------------------------
        # Percentage change
        # --------------------------------------------------

        elif normalized_operation in {
            "calculate_percentage_change",
            "percentage_change",
        }:

            step_errors = (
                _validate_percentage_change_step(
                    parameters=parameters,
                    actual_columns=actual_columns,
                    derived_columns=derived_columns,
                )
            )

        else:

            step_errors = (
                _validate_parameter_columns(
                    parameters=parameters,
                    actual_columns=actual_columns,
                    warnings=warnings,
                    derived_columns=derived_columns,
                )
            )

        errors.extend(
            [
                f"Step {index}: {error}"
                for error in step_errors
            ]
        )

        # --------------------------------------------------
        # Charts
        # --------------------------------------------------

        if normalized_operation in {
            "generate_bar_chart",
            "bar_chart",
            "chart_generator",
            "generate_line_chart",
            "line_chart",
        }:

            chart_errors = _validate_chart_step(
                step=step,
                actual_columns=actual_columns,
                warnings=warnings,
                derived_columns=derived_columns,
            )

            errors.extend(
                [
                    f"Step {index}: {error}"
                    for error in chart_errors
                ]
            )

        # --------------------------------------------------
        # Preserve validated step
        # --------------------------------------------------

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
