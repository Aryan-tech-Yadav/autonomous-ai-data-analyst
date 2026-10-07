from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.tools.registry import get_operation
from src.tools.operations import create_trend_chart_data


DERIVED_REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "sales",
    "sales revenue",
}


def _normalize_column_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )


def _is_derived_revenue_column(value: Any) -> bool:
    return (
        _normalize_column_name(value)
        in DERIVED_REVENUE_ALIASES
    )


class OperationExecutor:

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    # ========================================================
    # REVENUE
    # ========================================================

    def _ensure_revenue_column(
        self,
        df: pd.DataFrame,
        units_column: str = "Units Sold",
        price_column: str = "Unit Price",
        output_column: str = "Revenue",
    ) -> pd.DataFrame:

        working_df = df.copy()

        if units_column not in working_df.columns:
            raise ValueError(
                f"Units column '{units_column}' does not exist."
            )

        if price_column not in working_df.columns:
            raise ValueError(
                f"Price column '{price_column}' does not exist."
            )

        if output_column in working_df.columns:

            existing_revenue = pd.to_numeric(
                working_df[output_column],
                errors="coerce",
            )

            if existing_revenue.notna().sum() > 0:
                working_df[output_column] = existing_revenue
                return working_df

        units = pd.to_numeric(
            working_df[units_column],
            errors="coerce",
        )

        price = pd.to_numeric(
            working_df[price_column],
            errors="coerce",
        )

        working_df[output_column] = units * price

        return working_df

    # ========================================================
    # REVENUE ALIAS NORMALIZATION
    # ========================================================

    def _resolve_revenue_column(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> tuple[pd.DataFrame, str]:

        if not _is_derived_revenue_column(column):
            return df, column

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if values.notna().sum() > 0:
                return df, column

        working_df = self._ensure_revenue_column(
            df=df,
            output_column="Revenue",
        )

        return working_df, "Revenue"

    # ========================================================
    # TREND RESULT HELPERS
    # ========================================================

    def _get_latest_trend_result(
        self,
    ) -> Dict[str, Any] | None:
        """
        Return the latest successful trend_analysis result.
        """

        for item in reversed(self.results):

            if (
                item.get("operation")
                == "trend_analysis"
                and item.get("status")
                == "success"
            ):

                result = item.get("result")

                if isinstance(result, dict):
                    return result

        return None

    def _trend_result_to_chart_dataframe(
        self,
        trend_result: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert a verified trend_analysis result into
        a chart-ready DataFrame.

        The trend result is the source of truth.
        No trend recalculation happens here.
        """

        return create_trend_chart_data(
            trend_result
        )

    def _execute_trend_line_chart(
        self,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate a line chart from the latest successful
        trend_analysis result.
        """

        trend_result = self._get_latest_trend_result()

        if trend_result is None:
            raise ValueError(
                "generate_line_chart requires a successful "
                "trend_analysis result before chart creation."
            )

        chart_df = self._trend_result_to_chart_dataframe(
            trend_result
        )

        chart_parameters = {
            "x_column": "Period",
            "y_column": "Value",
            "title": parameters.get(
                "title",
                "Trend",
            ),
            "aggregation": "sum",
            "sort_x": True,
            "output_path": parameters.get(
                "output_path"
            ),
        }

        tool = get_operation(
            "generate_line_chart"
        )

        result = tool(
            df=chart_df,
            **chart_parameters,
        )

        return chart_df, result

    # ========================================================
    # CHART EXECUTION
    # ========================================================

    def _execute_chart(
        self,
        operation: str,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        working_df = df.copy()
        params = dict(parameters)

        x_column = (
            params.get("x_column")
            or params.get("category_column")
            or params.get("group_column")
        )

        y_column = (
            params.get("y_column")
            or params.get("value_column")
            or params.get("metric_column")
        )

        if not x_column:
            raise ValueError(
                f"{operation} requires an x/category column."
            )

        if not y_column:
            raise ValueError(
                f"{operation} requires a y/value column."
            )

        if _is_derived_revenue_column(y_column):

            working_df, resolved_y = (
                self._resolve_revenue_column(
                    working_df,
                    y_column,
                )
            )

            y_column = resolved_y

        if operation == "generate_bar_chart":

            chart_parameters = {
                "x_column": x_column,
                "y_column": y_column,
                "title": params.get("title"),
                "aggregation": params.get(
                    "aggregation",
                    "sum",
                ),
                "top_n": params.get("top_n"),
                "ascending": params.get(
                    "ascending",
                    False,
                ),
                "output_path": params.get(
                    "output_path"
                ),
            }

        elif operation == "generate_line_chart":

            chart_parameters = {
                "x_column": x_column,
                "y_column": y_column,
                "title": params.get("title"),
                "aggregation": params.get(
                    "aggregation",
                    "sum",
                ),
                "sort_x": params.get(
                    "sort_x",
                    True,
                ),
                "output_path": params.get(
                    "output_path"
                ),
            }

        else:
            raise ValueError(
                f"Unsupported chart operation: {operation}"
            )

        tool = get_operation(operation)

        result = tool(
            df=working_df,
            **chart_parameters,
        )

        return working_df, result

    # ========================================================
    # PERSIST SHIFTED COLUMN
    # ========================================================

    def _persist_shifted_column(
        self,
        working_df: pd.DataFrame,
        result: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        output_column = (
            result.get("output_column")
            or parameters.get("output_column")
        )

        if (
            output_column
            and output_column in result
        ):
            working_df[
                output_column
            ] = result[output_column]

        return working_df, result

    # ========================================================
    # EXECUTE
    # ========================================================

    def execute(
        self,
        df: pd.DataFrame,
        execution_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        working_df = df.copy()

        self.results = []

        total_steps = len(execution_steps)

        successful_steps = 0
        failed_steps = 0

        for index, step in enumerate(
            execution_steps,
            start=1,
        ):

            if not isinstance(step, dict):

                self.results.append(
                    {
                        "step": index,
                        "operation": None,
                        "status": "error",
                        "error": (
                            "Execution step must "
                            "be a dictionary."
                        ),
                    }
                )

                failed_steps += 1
                continue

            operation = step.get(
                "operation"
            )

            parameters = step.get(
                "parameters",
                {},
            )

            if not operation:

                self.results.append(
                    {
                        "step": index,
                        "operation": None,
                        "status": "error",
                        "error": (
                            "Missing operation."
                        ),
                    }
                )

                failed_steps += 1
                continue

            if not isinstance(
                parameters,
                dict,
            ):

                self.results.append(
                    {
                        "step": index,
                        "operation": operation,
                        "status": "error",
                        "error": (
                            "Parameters must "
                            "be a dictionary."
                        ),
                    }
                )

                failed_steps += 1
                continue

            normalized_operation = (
                str(operation)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            operation_aliases = {

                "revenue_calculation":
                    "revenue_calculations",

                "revenue_calculate":
                    "revenue_calculations",

                "group_by":
                    "groupby_aggregation",

                "groupby":
                    "groupby_aggregation",

                "statistics":
                    "calculate_statistics",

                "statistical_analysis":
                    "calculate_statistics",

                "percentage_change":
                    "calculate_percentage_change",

                "percentage":
                    "calculate_percentage_change",

                "comparison":
                    "compare_columns",

                "line_chart":
                    "generate_line_chart",

                "bar_chart":
                    "generate_bar_chart",
            }

            normalized_operation = (
                operation_aliases.get(
                    normalized_operation,
                    normalized_operation,
                )
            )

            allowed_operations = {

                "revenue_calculations",
                "calculate_revenue",
                "groupby_aggregation",
                "groupby_aggregate",
                "find_max",
                "calculate_statistics",
                "categorical_analysis",
                "rank_by_value",
                "create_shifted_column",
                "calculate_percentage_change",
                "compare_columns",
                "trend_analysis",
                "generate_bar_chart",
                "generate_line_chart",
            }

            if normalized_operation not in allowed_operations:

                self.results.append(
                    {
                        "step": index,
                        "operation":
                            normalized_operation,
                        "status": "error",
                        "error": (
                            "Operation is not allowed: "
                            f"{normalized_operation}"
                        ),
                    }
                )

                failed_steps += 1
                continue

            try:

                # ==================================================
                # CHARTS
                # ==================================================

                if normalized_operation == "generate_line_chart":

                    latest_trend = (
                        self._get_latest_trend_result()
                    )

                    if latest_trend is not None:

                        chart_df, result = (
                            self._execute_trend_line_chart(
                                parameters,
                            )
                        )

                        # IMPORTANT:
                        # Chart data must NEVER replace the
                        # original working dataframe.
                        # Keep business data available for
                        # all subsequent operations.
                        _ = chart_df

                    else:

                        (
                            _chart_df,
                            result,
                        ) = self._execute_chart(
                            normalized_operation,
                            working_df,
                            parameters,
                        )

                elif normalized_operation == "generate_bar_chart":

                    (
                        working_df,
                        result,
                    ) = self._execute_chart(
                        normalized_operation,
                        working_df,
                        parameters,
                    )

                # ==================================================
                # REVENUE
                # ==================================================

                elif normalized_operation in {
                    "revenue_calculations",
                }:

                    working_df = (
                        self._ensure_revenue_column(
                            df=working_df,
                        )
                    )

                    result = {
                        "operation":
                            "revenue_calculations",
                        "status":
                            "success",
                        "column":
                            "Revenue",
                        "non_null_values":
                            int(
                                working_df[
                                    "Revenue"
                                ].notna().sum()
                            ),
                        "total_revenue":
                            float(
                                working_df[
                                    "Revenue"
                                ].sum()
                            ),
                    }

                # ==================================================
                # TREND ANALYSIS
                # ==================================================

                elif normalized_operation == "trend_analysis":

                    trend_parameters = dict(
                        parameters
                    )

                    value_column = (
                        trend_parameters.get(
                            "value_column"
                        )
                    )

                    date_column = (
                        trend_parameters.get(
                            "date_column"
                        )
                    )

                    if _is_derived_revenue_column(
                        value_column
                    ):

                        (
                            working_df,
                            resolved_value,
                        ) = (
                            self._resolve_revenue_column(
                                working_df,
                                value_column,
                            )
                        )

                        trend_parameters[
                            "value_column"
                        ] = resolved_value

                    if (
                        date_column
                        not in working_df.columns
                    ):
                        raise ValueError(
                            f"Date column "
                            f"'{date_column}' "
                            "does not exist."
                        )

                    if (
                        trend_parameters.get(
                            "value_column"
                        )
                        not in working_df.columns
                    ):
                        raise ValueError(
                            "Value column "
                            f"'{trend_parameters.get('value_column')}' "
                            "does not exist."
                        )

                    tool = get_operation(
                        "trend_analysis"
                    )

                    result = tool(
                        df=working_df,
                        **trend_parameters,
                    )

                # ==================================================
                # SHIFTED COLUMN
                # ==================================================

                elif normalized_operation == (
                    "create_shifted_column"
                ):

                    tool = get_operation(
                        normalized_operation
                    )

                    result = tool(
                        df=working_df,
                        **parameters,
                    )

                    (
                        working_df,
                        result,
                    ) = self._persist_shifted_column(
                        working_df,
                        result,
                        parameters,
                    )

                # ==================================================
                # STANDARD OPERATIONS
                # ==================================================

                else:

                    operation_parameters = dict(parameters)

                    value_column = operation_parameters.get(
                        "value_column"
                    )

                    if _is_derived_revenue_column(
                        value_column
                    ):
                        (
                            working_df,
                            resolved_value_column,
                        ) = self._resolve_revenue_column(
                            working_df,
                            value_column,
                        )

                        operation_parameters[
                            "value_column"
                        ] = resolved_value_column

                    tool = get_operation(
                        normalized_operation
                    )

                    result = tool(
                        df=working_df,
                        **operation_parameters,
                    )

                # ==================================================
                # STORE SUCCESS
                # ==================================================

                self.results.append(
                    {
                        "step": index,
                        "operation":
                            normalized_operation,
                        "status": "success",
                        "result": result,
                    }
                )

                successful_steps += 1

            except Exception as exc:

                self.results.append(
                    {
                        "step": index,
                        "operation":
                            normalized_operation,
                        "status": "error",
                        "error": str(exc),
                    }
                )

                failed_steps += 1

        if failed_steps == 0:

            status = "success"

        elif successful_steps > 0:

            status = "partial_success"

        else:

            status = "error"

        return {
            "status": status,
            "total_steps": total_steps,
            "successful_steps":
                successful_steps,
            "failed_steps":
                failed_steps,
            "results":
                self.results,
        }
