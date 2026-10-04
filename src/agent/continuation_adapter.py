from typing import Any


class ContinuationAdapter:
    """
    Converts ResultInterpreter next_operations into the same
    analysis-plan structure expected by the existing pipeline.

    Also repairs missing parameters for common autonomous
    continuation operations using the previous execution results.
    """

    def adapt(
        self,
        next_operations: list[dict[str, Any]],
        start_step: int = 1,
        execution_results: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not isinstance(next_operations, list):
            raise ValueError(
                "next_operations must be a list."
            )

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

            operation_name = str(
                operation_name
            ).strip().lower()

            parameters = operation.get(
                "parameters",
                {},
            )

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
                "tool": self._infer_tool(
                    operation_name
                ),
                "operation": operation_name,
                "parameters": parameters,
                "description": (
                    operation.get(
                        "description"
                    )
                    or (
                        "Autonomous continuation: "
                        f"{operation_name}"
                    )
                ),
            }

            analysis_plan.append(step)

            step_number += 1

        if not analysis_plan:
            raise ValueError(
                "No valid continuation operations were provided."
            )

        return {
            "analysis_plan": analysis_plan
        }

    def _repair_parameters(
        self,
        operation_name: str,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Deterministically repair missing parameters.

        We only infer parameters when they can be derived from
        actual execution results or dataset schema.
        """

        if operation_name == "generate_bar_chart":
            return self._repair_bar_chart_parameters(
                parameters=parameters,
                execution_results=execution_results,
                context=context,
            )

        if operation_name == "generate_line_chart":
            return self._repair_line_chart_parameters(
                parameters=parameters,
                execution_results=execution_results,
                context=context,
            )

        return parameters

    def _repair_bar_chart_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        results = execution_results.get(
            "results",
            [],
        )

        if not isinstance(results, list):
            results = []

        # --------------------------------------------------
        # 1. Find the latest groupby result
        # --------------------------------------------------

        groupby_result = None

        for result in reversed(results):

            if not isinstance(result, dict):
                continue

            operation = str(
                result.get("operation", "")
            ).strip().lower()

            if operation in {
                "groupby_aggregate",
                "groupby_aggregation",
                "group_by",
            }:
                groupby_result = result
                break

        if groupby_result:

            result_data = groupby_result.get(
                "result",
                {},
            )

            if isinstance(result_data, dict):

                group_column = (
                    result_data.get("group_column")
                    or result_data.get("group")
                    or result_data.get("x_column")
                )

                value_column = (
                    result_data.get("value_column")
                    or result_data.get("value")
                    or result_data.get("y_column")
                )

                if (
                    not repaired.get("x_column")
                    and group_column
                ):
                    repaired["x_column"] = str(
                        group_column
                    )

                if (
                    not repaired.get("y_column")
                    and value_column
                ):
                    repaired["y_column"] = str(
                        value_column
                    )

                if (
                    not repaired.get("title")
                    and group_column
                    and value_column
                ):
                    repaired["title"] = (
                        f"{value_column} by "
                        f"{group_column}"
                    )

        # --------------------------------------------------
        # 2. Inspect nested result data when metadata
        #    is not available
        # --------------------------------------------------

        if (
            not repaired.get("x_column")
            or not repaired.get("y_column")
        ):

            for result in reversed(results):

                if not isinstance(result, dict):
                    continue

                result_data = result.get(
                    "result",
                    {},
                )

                if not isinstance(
                    result_data,
                    dict,
                ):
                    continue

                data = result_data.get(
                    "data"
                )

                if not isinstance(
                    data,
                    list,
                ) or not data:
                    continue

                first_row = data[0]

                if not isinstance(
                    first_row,
                    dict,
                ):
                    continue

                if (
                    "x" in first_row
                    and "y" in first_row
                ):

                    if not repaired.get(
                        "x_column"
                    ):
                        repaired["x_column"] = (
                            result_data.get(
                                "x_column"
                            )
                            or "Region"
                        )

                    if not repaired.get(
                        "y_column"
                    ):
                        repaired["y_column"] = (
                            result_data.get(
                                "y_column"
                            )
                            or "Total Revenue"
                        )

                    break

        # --------------------------------------------------
        # 3. Safe fallback from schema
        # --------------------------------------------------

        schema = context.get(
            "schema",
            {},
        )

        if isinstance(schema, dict):

            columns = schema.get(
                "columns",
                [],
            )

            if isinstance(
                columns,
                list,
            ):

                column_names = []

                for column in columns:

                    if isinstance(
                        column,
                        dict,
                    ):
                        name = column.get(
                            "name"
                        )
                    else:
                        name = column

                    if name:
                        column_names.append(
                            str(name)
                        )

                if (
                    not repaired.get(
                        "x_column"
                    )
                    and "Region"
                    in column_names
                ):
                    repaired["x_column"] = (
                        "Region"
                    )

                if (
                    not repaired.get(
                        "y_column"
                    )
                    and "Total Revenue"
                    in column_names
                ):
                    repaired["y_column"] = (
                        "Total Revenue"
                    )

        # --------------------------------------------------
        # 4. Known semantic fallback:
        #    Revenue is commonly represented as
        #    Total Revenue in our execution layer.
        # --------------------------------------------------

        if (
            not repaired.get("y_column")
            and self._has_revenue_signal(
                execution_results
            )
        ):
            repaired["y_column"] = (
                "Total Revenue"
            )

        if (
            not repaired.get("x_column")
            and self._has_region_signal(
                execution_results,
                context,
            )
        ):
            repaired["x_column"] = "Region"

        # --------------------------------------------------
        # 5. Final chart defaults
        # --------------------------------------------------

        if (
            not repaired.get("title")
            and repaired.get("x_column")
            and repaired.get("y_column")
        ):
            repaired["title"] = (
                f"{repaired['y_column']} by "
                f"{repaired['x_column']}"
            )

        return repaired

    def _repair_line_chart_parameters(
        self,
        parameters: dict[str, Any],
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        repaired = dict(parameters)

        schema = context.get(
            "schema",
            {},
        )

        columns = []

        if isinstance(schema, dict):
            schema_columns = schema.get(
                "columns",
                [],
            )

            if isinstance(
                schema_columns,
                list,
            ):
                for column in schema_columns:

                    if isinstance(
                        column,
                        dict,
                    ):
                        name = column.get(
                            "name"
                        )
                    else:
                        name = column

                    if name:
                        columns.append(
                            str(name)
                        )

        if (
            not repaired.get("x_column")
            and "Date" in columns
        ):
            repaired["x_column"] = "Date"

        if (
            not repaired.get("y_column")
            and "Total Revenue" in columns
        ):
            repaired["y_column"] = (
                "Total Revenue"
            )

        if (
            not repaired.get("title")
            and repaired.get("x_column")
            and repaired.get("y_column")
        ):
            repaired["title"] = (
                f"{repaired['y_column']} over "
                f"{repaired['x_column']}"
            )

        return repaired

    def _has_revenue_signal(
        self,
        execution_results: dict[str, Any],
    ) -> bool:

        serialized = str(
            execution_results
        ).lower()

        return (
            "revenue" in serialized
            or "total_revenue" in serialized
        )

    def _has_region_signal(
        self,
        execution_results: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:

        if "Region" in str(
            context.get("schema", {})
        ):
            return True

        serialized = str(
            execution_results
        )

        return "Region" in serialized

    def _infer_tool(
        self,
        operation: str,
    ) -> str:

        normalized = str(
            operation
        ).strip().lower()

        if normalized in {
            "generate_bar_chart",
            "generate_line_chart",
        }:
            return "chart_generator"

        if normalized == "time_analysis":
            return "time_analysis"

        return "pandas_analysis"
