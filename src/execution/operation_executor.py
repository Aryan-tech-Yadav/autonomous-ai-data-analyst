from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.tools.registry import get_operation


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
    """
    Execute validated analysis operations sequentially.

    A working DataFrame is maintained across all steps so
    derived columns created by earlier operations can be
    consumed by later operations.
    """

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
        """
        Ensure a usable revenue column exists.

        Important:
        If the requested revenue column already exists but
        contains only missing values, it MUST be overwritten
        with calculated revenue.
        """

        working_df = df.copy()

        if units_column not in working_df.columns:
            raise ValueError(
                f"Units column '{units_column}' does not exist."
            )

        if price_column not in working_df.columns:
            raise ValueError(
                f"Price column '{price_column}' does not exist."
            )

        # If an existing revenue column contains usable data,
        # preserve it.
        if output_column in working_df.columns:

            existing_revenue = pd.to_numeric(
                working_df[output_column],
                errors="coerce",
            )

            if existing_revenue.notna().sum() > 0:
                return working_df

        units = pd.to_numeric(
            working_df[units_column],
            errors="coerce",
        )

        price = pd.to_numeric(
            working_df[price_column],
            errors="coerce",
        )

        # IMPORTANT:
        # Always assign the calculated values when the existing
        # column is empty.
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

        # Prefer the exact requested column if it contains
        # usable values.
        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if values.notna().sum() > 0:
                return df, column

        # Otherwise create/use Revenue.
        working_df = self._ensure_revenue_column(
            df=df,
            output_column="Revenue",
        )

        return working_df, "Revenue"

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

        # ----------------------------------------------------
        # COLUMN ALIASES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DERIVED REVENUE
        # ----------------------------------------------------

        if _is_derived_revenue_column(y_column):

            working_df, resolved_y = (
                self._resolve_revenue_column(
                    working_df,
                    y_column,
                )
            )

            y_column = resolved_y

        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LINE CHART
        # ----------------------------------------------------

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
            or "Shifted Value"
        )

        shifted_values = result.get(
            "shifted_values"
        )

        if shifted_values is None:
            raise ValueError(
                "Shift operation did not return shifted values."
            )

        if not isinstance(
            shifted_values,
            pd.Series,
        ):
            shifted_values = pd.Series(
                shifted_values,
                index=working_df.index,
            )

        working_df = working_df.copy()

        working_df[output_column] = (
            shifted_values.reindex(
                working_df.index
            )
        )

        safe_result = dict(result)

        safe_result.pop(
            "shifted_values",
            None,
        )

        safe_result["output_column"] = (
            output_column
        )

        return working_df, safe_result

    # ========================================================
    # PERSIST PERCENTAGE CHANGE
    # ========================================================

    def _persist_percentage_change(
        self,
        working_df: pd.DataFrame,
        result: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        output_column = (
            result.get("output_column")
            or parameters.get("output_column")
            or "Percentage Change"
        )

        percentage_change = result.get(
            "percentage_change"
        )

        if percentage_change is None:
            percentage_change = result.get(
                "values"
            )

        if percentage_change is None:
            raise ValueError(
                "Percentage change operation did not return values."
            )

        if not isinstance(
            percentage_change,
            pd.Series,
        ):
            percentage_change = pd.Series(
                percentage_change,
                index=working_df.index,
            )

        working_df = working_df.copy()

        working_df[output_column] = (
            percentage_change.reindex(
                working_df.index
            )
        )

        safe_result = dict(result)

        safe_result.pop(
            "percentage_change",
            None,
        )

        safe_result.pop(
            "values",
            None,
        )

        safe_result["output_column"] = (
            output_column
        )

        return working_df, safe_result

    # ========================================================
    # REVENUE EXECUTION
    # ========================================================

    def _execute_revenue(
        self,
        working_df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        units_column = parameters.get(
            "units_column",
            parameters.get(
                "quantity_column",
                "Units Sold",
            ),
        )

        price_column = parameters.get(
            "price_column",
            "Unit Price",
        )

        output_column = parameters.get(
            "output_column",
            "Revenue",
        )

        working_df = self._ensure_revenue_column(
            df=working_df,
            units_column=units_column,
            price_column=price_column,
            output_column=output_column,
        )

        revenue_values = pd.to_numeric(
            working_df[output_column],
            errors="coerce",
        )

        valid_values = revenue_values.dropna()

        result = {
            "operation": "calculate_revenue",
            "output_column": output_column,
            "rows_calculated": int(
                revenue_values.notna().sum()
            ),
            "total_revenue": float(
                valid_values.sum()
            ),
        }

        return working_df, result

    # ========================================================
    # MAIN EXECUTOR
    # ========================================================

    def execute(
        self,
        df: pd.DataFrame,
        execution_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        working_df = df.copy()

        self.results = []

        for step in execution_steps:

            step_number = step.get(
                "step"
            )

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
                        "step": step_number,
                        "status": "error",
                        "error": "Missing operation.",
                    }
                )

                continue

            if not isinstance(
                parameters,
                dict,
            ):

                self.results.append(
                    {
                        "step": step_number,
                        "operation": operation,
                        "status": "error",
                        "error": (
                            "Parameters must "
                            "be a dictionary."
                        ),
                    }
                )

                continue

            normalized_operation = str(
                operation
            ).strip().lower().replace(
                "-",
                "_",
            ).replace(
                " ",
                "_",
            )

            operation_aliases = {
                "revenue_calculation":
                    "revenue_calculations",

                "group_by":
                    "groupby_aggregation",

                "groupby":
                    "groupby_aggregation",

                "groupby_aggregate":
                    "groupby_aggregation",

                "calculate_statistics":
                    "statistics",

                "bar_chart":
                    "generate_bar_chart",

                "chart_generator":
                    "generate_bar_chart",

                "line_chart":
                    "generate_line_chart",

                "shift":
                    "create_shifted_column",

                "ranking":
                    "rank_by_value",

                "percentage":
                    "calculate_percentage_change",

                "comparison":
                    "compare_columns",
            }

            normalized_operation = (
                operation_aliases.get(
                    normalized_operation,
                    normalized_operation,
                )
            )

            try:

                # ==========================================
                # CHARTS
                # ==========================================

                if normalized_operation in {
                    "generate_bar_chart",
                    "generate_line_chart",
                }:

                    (
                        working_df,
                        result,
                    ) = self._execute_chart(
                        normalized_operation,
                        working_df,
                        parameters,
                    )

                # ==========================================
                # REVENUE
                # ==========================================

                elif normalized_operation in {
                    "calculate_revenue",
                    "revenue_calculation",
                    "revenue_calculations",
                }:

                    (
                        working_df,
                        result,
                    ) = self._execute_revenue(
                        working_df,
                        parameters,
                    )

                # ==========================================
                # SHIFT
                # ==========================================

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

                # ==========================================
                # PERCENTAGE CHANGE
                # ==========================================

                elif normalized_operation == (
                    "calculate_percentage_change"
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
                    ) = self._persist_percentage_change(
                        working_df,
                        result,
                        parameters,
                    )

                # ==========================================
                # STANDARD OPERATIONS
                # ==========================================

                else:

                    tool = get_operation(
                        normalized_operation
                    )

                    result = tool(
                        df=working_df,
                        **parameters,
                    )

                self.results.append(
                    {
                        "step": step_number,
                        "operation":
                            normalized_operation,
                        "status": "success",
                        "result": result,
                    }
                )

            except Exception as exc:

                self.results.append(
                    {
                        "step": step_number,
                        "operation":
                            normalized_operation,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        successful_steps = sum(
            1
            for item in self.results
            if item.get("status")
            == "success"
        )

        failed_steps = sum(
            1
            for item in self.results
            if item.get("status")
            == "error"
        )

        total_steps = len(
            execution_steps
        )

        if failed_steps == 0:
            status = "success"

        elif successful_steps == 0:
            status = "error"

        else:
            status = "partial"

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
