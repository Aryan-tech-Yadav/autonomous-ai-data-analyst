from typing import Any


class ContinuationAdapter:
    """
    Converts ResultInterpreter continuation operations into the
    analysis-plan structure expected by the existing pipeline.

    Also repairs missing parameters deterministically using:
    1. Previous execution results
    2. Dataset schema
    3. Known business semantics
    """

    def adapt(
        self,
        next_operations: list[dict[str, Any]],
        start_step: int = 1,
        execution_results: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not isinstance(next_operations, list):
            raise ValueError("next_operations must be a list.")

        execution_results = execution_results or {}
        context = context or {}

        analysis_plan = []
        step_number = start_step

        for operation in next_operations:

            if not isinstance(operation, dict):
                continue

            operation_name = operation.get("operation")

            if not operation_name:
                continue

            operation_name = str(operation_name).strip().lower()

            parameters = operation.get("parameters", {})

            if not isinstance(parameters, dict):
                parameters = {}

            parameters = dict(parameters)

            parameters = self._repair_parameters(
                operation_name=operation_name,
                parameters=parameters,
                execution_results=execution_results,
                context=context,
            )

            step = {
                "step": step_number,
                "tool": self._infer_tool(operation_name),
                "operation": operation_name,
                "parameters": parameters,
                "description": (
                    operation.get("description")
                    or f"Autonomous continuation: {operation_name}"
                ),
            }

            analysis_plan.append(step)
            step_number += 1

        if not analysis_plan:
            raise ValueError("No valid continuation operations were provided.")

        return {
            "analysis_plan": analysis_plan
        }

    # ============================================================
    # PARAMETER REPAIR
    # ============================================================

    def _repair_parameters(
        self,
        operation_name: str,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        normalized = self._normalize_operation(operation_name)

        if normalized == "generate_bar_chart":
            return self._repair_bar_chart_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "generate_line_chart":
            return self._repair_line_chart_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized in {
            "groupby_aggregation",
            "groupby_aggregate",
        }:
            return self._repair_groupby_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "trend_analysis":
            return self._repair_trend_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "find_max":
            return self._repair_value_operation_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "calculate_statistics":
            return self._repair_value_operation_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "rank_by_value":
            return self._repair_value_operation_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "categorical_analysis":
            return self._repair_categorical_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "calculate_percentage_change":
            return self._repair_percentage_change_parameters(
                parameters,
                execution_results,
                context,
            )

        if normalized == "compare_columns":
            return self._repair_compare_parameters(
                parameters,
                execution_results,
                context,
            )

        return parameters

    # ============================================================
    # GROUPBY
    # ============================================================

    def _repair_groupby_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        group_column = (
            repaired.get("group_column")
            or repaired.get("group_by")
            or repaired.get("category_column")
        )

        value_column = (
            repaired.get("value_column")
            or repaired.get("metric_column")
            or repaired.get("column")
        )

        aggregation = (
            repaired.get("aggregation")
            or repaired.get("agg")
            or "sum"
        )

        # First use previous groupby metadata
        if not group_column or not value_column:

            previous = self._latest_result(
                execution_results,
                {
                    "groupby_aggregate",
                    "groupby_aggregation",
                    "group_by",
                },
            )

            metadata = self._result_metadata(previous)

            if not group_column:
                group_column = (
                    metadata.get("group_column")
                    or metadata.get("group")
                    or metadata.get("x_column")
                )

            if not value_column:
                value_column = (
                    metadata.get("value_column")
                    or metadata.get("value")
                    or metadata.get("y_column")
                )

        columns = self._schema_columns(context)

        # Business-safe defaults for this project
        if not group_column:
            for candidate in [
                "Region",
                "Product Category",
                "Sales Rep",
                "Status",
            ]:
                if candidate in columns:
                    group_column = candidate
                    break

        if not value_column:
            value_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if group_column:
            repaired["group_column"] = group_column

        if value_column:
            repaired["value_column"] = value_column

        repaired["aggregation"] = aggregation

        return repaired

    # ============================================================
    # TREND
    # ============================================================

    def _repair_trend_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        date_column = (
            repaired.get("date_column")
            or repaired.get("x_column")
            or repaired.get("column")
        )

        value_column = (
            repaired.get("value_column")
            or repaired.get("metric_column")
            or repaired.get("y_column")
        )

        period = repaired.get("period") or "monthly"

        columns = self._schema_columns(context)

        previous = self._latest_result(
            execution_results,
            {
                "trend_analysis",
                "grouped_trend_analysis",
            },
        )

        metadata = self._result_metadata(previous)

        if not date_column:
            date_column = (
                metadata.get("date_column")
                or metadata.get("x_column")
                or metadata.get("time_column")
            )

        if not value_column:
            value_column = (
                metadata.get("value_column")
                or metadata.get("y_column")
                or metadata.get("metric_column")
            )

        if not date_column and "Date" in columns:
            date_column = "Date"

        if not value_column:
            value_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if date_column:
            repaired["date_column"] = date_column

        if value_column:
            repaired["value_column"] = value_column

        repaired["period"] = period

        return repaired

    # ============================================================
    # VALUE OPERATIONS
    # ============================================================

    def _repair_value_operation_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        value_column = (
            repaired.get("value_column")
            or repaired.get("column")
            or repaired.get("metric_column")
        )

        columns = self._schema_columns(context)

        if not value_column:
            previous = self._latest_result(
                execution_results,
                {
                    "find_max",
                    "calculate_statistics",
                    "rank_by_value",
                },
            )

            metadata = self._result_metadata(previous)

            value_column = (
                metadata.get("value_column")
                or metadata.get("column")
                or metadata.get("metric_column")
            )

        if not value_column:
            value_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if value_column:
            repaired["value_column"] = value_column

        return repaired

    # ============================================================
    # CATEGORICAL
    # ============================================================

    def _repair_categorical_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        column = (
            repaired.get("column")
            or repaired.get("category_column")
            or repaired.get("group_column")
        )

        columns = self._schema_columns(context)

        if not column:
            for candidate in [
                "Region",
                "Product Category",
                "Sales Rep",
                "Status",
            ]:
                if candidate in columns:
                    column = candidate
                    break

        if column:
            repaired["column"] = column

        return repaired

    # ============================================================
    # PERCENTAGE CHANGE
    # ============================================================

    def _repair_percentage_change_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        value_column = (
            repaired.get("value_column")
            or repaired.get("column")
            or repaired.get("metric_column")
        )

        columns = self._schema_columns(context)

        if not value_column:
            value_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if value_column:
            repaired["value_column"] = value_column

        return repaired

    # ============================================================
    # COMPARE COLUMNS
    # ============================================================

    def _repair_compare_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        column_a = (
            repaired.get("column_a")
            or repaired.get("first_column")
            or repaired.get("left_column")
        )

        column_b = (
            repaired.get("column_b")
            or repaired.get("second_column")
            or repaired.get("right_column")
        )

        columns = self._schema_columns(context)

        numeric_columns = [
            c for c in columns
            if c not in {
                "Order ID",
                "Date",
                "Sales Rep",
                "Region",
                "Product Category",
                "Status",
            }
        ]

        if not column_a and numeric_columns:
            column_a = numeric_columns[0]

        if not column_b and len(numeric_columns) > 1:
            column_b = numeric_columns[1]

        if column_a:
            repaired["column_a"] = column_a

        if column_b:
            repaired["column_b"] = column_b

        return repaired

    # ============================================================
    # BAR CHART
    # ============================================================

    def _repair_bar_chart_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        previous = self._latest_result(
            execution_results,
            {
                "groupby_aggregate",
                "groupby_aggregation",
                "group_by",
            },
        )

        metadata = self._result_metadata(previous)

        x_column = (
            repaired.get("x_column")
            or metadata.get("group_column")
            or metadata.get("group")
            or metadata.get("x_column")
        )

        y_column = (
            repaired.get("y_column")
            or metadata.get("value_column")
            or metadata.get("value")
            or metadata.get("y_column")
        )

        columns = self._schema_columns(context)

        if not x_column and "Region" in columns:
            x_column = "Region"

        if not y_column:
            y_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if x_column:
            repaired["x_column"] = x_column

        if y_column:
            repaired["y_column"] = y_column

        if (
            not repaired.get("title")
            and x_column
            and y_column
        ):
            repaired["title"] = f"{y_column} by {x_column}"

        return repaired

    # ============================================================
    # LINE CHART
    # ============================================================

    def _repair_line_chart_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        previous = self._latest_result(
            execution_results,
            {
                "trend_analysis",
                "grouped_trend_analysis",
            },
        )

        metadata = self._result_metadata(previous)

        x_column = (
            repaired.get("x_column")
            or metadata.get("date_column")
            or metadata.get("x_column")
        )

        y_column = (
            repaired.get("y_column")
            or metadata.get("value_column")
            or metadata.get("y_column")
        )

        columns = self._schema_columns(context)

        if not x_column and "Date" in columns:
            x_column = "Date"

        if not y_column:
            y_column = self._best_numeric_column(
                columns,
                prefer_revenue=True,
            )

        if x_column:
            repaired["x_column"] = x_column

        if y_column:
            repaired["y_column"] = y_column

        if (
            not repaired.get("title")
            and x_column
            and y_column
        ):
            repaired["title"] = f"{y_column} over {x_column}"

        return repaired

    # ============================================================
    # HELPERS
    # ============================================================

    def _schema_columns(
        self,
        context: dict[str, Any],
    ) -> list[str]:

        schema = context.get("schema", {})

        if not isinstance(schema, dict):
            return []

        raw_columns = schema.get("columns", [])

        if not isinstance(raw_columns, list):
            return []

        columns = []

        for column in raw_columns:

            if isinstance(column, dict):
                name = column.get("name")
            else:
                name = column

            if name:
                columns.append(str(name))

        return columns

    def _best_numeric_column(
        self,
        columns: list[str],
        prefer_revenue: bool = False,
    ) -> str | None:

        if prefer_revenue:

            for candidate in [
                "Revenue",
                "Total Revenue",
            ]:
                if candidate in columns:
                    return candidate

        for candidate in [
            "Units Sold",
            "Unit Price",
            "Customer Satisfaction",
        ]:
            if candidate in columns:
                return candidate

        # Avoid selecting obvious categorical/date fields.
        excluded = {
            "Order ID",
            "Date",
            "Sales Rep",
            "Region",
            "Product Category",
            "Status",
        }

        for column in columns:
            if column not in excluded:
                return column

        return None

    def _latest_result(
        self,
        execution_results: dict[str, Any],
        operations: set[str],
    ) -> dict[str, Any] | None:

        results = execution_results.get("results", [])

        if not isinstance(results, list):
            return None

        for result in reversed(results):

            if not isinstance(result, dict):
                continue

            operation = str(
                result.get("operation", "")
            ).strip().lower()

            if operation in operations:
                return result

        return None

    def _result_metadata(
        self,
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:

        if not isinstance(result, dict):
            return {}

        result_data = result.get("result", {})

        if not isinstance(result_data, dict):
            return {}

        return result_data

    def _normalize_operation(
        self,
        operation: str,
    ) -> str:

        normalized = str(operation).strip().lower()

        aliases = {
            "group_by": "groupby_aggregation",
            "groupby": "groupby_aggregation",
            "groupby_aggregate": "groupby_aggregation",
            "groupby_aggregation": "groupby_aggregation",
            "line_chart": "generate_line_chart",
            "bar_chart": "generate_bar_chart",
            "statistics": "calculate_statistics",
            "statistical_analysis": "calculate_statistics",
            "percentage_change": "calculate_percentage_change",
            "percentage": "calculate_percentage_change",
            "comparison": "compare_columns",
        }

        return aliases.get(normalized, normalized)

    def _infer_tool(
        self,
        operation: str,
    ) -> str:

        normalized = self._normalize_operation(operation)

        if normalized in {
            "generate_bar_chart",
            "generate_line_chart",
        }:
            return "chart_generator"

        if normalized in {
            "trend_analysis",
            "grouped_trend_analysis",
        }:
            return "time_analysis"

        return "pandas_analysis"
