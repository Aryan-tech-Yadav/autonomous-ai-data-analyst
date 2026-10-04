from typing import Any, Dict, List

import pandas as pd

from src.tools.registry import get_operation


DERIVED_REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "total_revenue",
    "calculated revenue",
    "calculated_revenue",
}


def _normalize_column_name(value: Any) -> str:
    return str(value).strip().lower()


def _is_derived_revenue_column(value: Any) -> bool:
    return _normalize_column_name(value) in DERIVED_REVENUE_ALIASES


class OperationExecutor:
    """
    Execute validated analysis operations sequentially.

    The executor keeps a working DataFrame so that derived columns
    created by one operation can be reused by later operations.

    Examples:
        Revenue
        Previous Unit Price
        Percentage Change
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def _ensure_revenue_column(
        self,
        df: pd.DataFrame,
        units_column: str = "Units Sold",
        price_column: str = "Unit Price",
    ) -> pd.DataFrame:

        working_df = df.copy()

        if "Revenue" in working_df.columns:
            revenue = pd.to_numeric(
                working_df["Revenue"],
                errors="coerce",
            )

            if revenue.notna().any():
                working_df["Revenue"] = revenue
                return working_df

        if units_column not in working_df.columns:
            raise ValueError(
                f"Cannot calculate Revenue: "
                f"units column '{units_column}' "
                "does not exist."
            )

        if price_column not in working_df.columns:
            raise ValueError(
                f"Cannot calculate Revenue: "
                f"price column '{price_column}' "
                "does not exist."
            )

        units = pd.to_numeric(
            working_df[units_column],
            errors="coerce",
        )

        prices = pd.to_numeric(
            working_df[price_column],
            errors="coerce",
        )

        working_df["Revenue"] = units * prices

        return working_df

    def _resolve_revenue_reference(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        working_df = df

        updated_parameters = dict(parameters)

        value_column = updated_parameters.get("value_column")

        if _is_derived_revenue_column(value_column):
            working_df = self._ensure_revenue_column(
                working_df
            )

            updated_parameters["value_column"] = "Revenue"

        return working_df, updated_parameters

    def _execute_chart(
        self,
        operation: str,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        working_df = df

        updated_parameters = dict(parameters)

        if operation == "generate_bar_chart":

            value_column = updated_parameters.get(
                "value_column"
            )

            if _is_derived_revenue_column(value_column):
                working_df = self._ensure_revenue_column(
                    working_df
                )

                updated_parameters["value_column"] = "Revenue"

        elif operation == "generate_line_chart":

            y_column = updated_parameters.get(
                "y_column"
            )

            if _is_derived_revenue_column(y_column):
                working_df = self._ensure_revenue_column(
                    working_df
                )

                updated_parameters["y_column"] = "Revenue"

        tool = get_operation(operation)

        result = tool(
            df=working_df,
            **updated_parameters,
        )

        return working_df, result

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

        shifted_values = result.get("shifted_values")

        if shifted_values is None:
            raise ValueError(
                "Shift operation did not return shifted values."
            )

        if not isinstance(shifted_values, pd.Series):
            shifted_values = pd.Series(
                shifted_values,
                index=working_df.index,
            )

        working_df = working_df.copy()

        working_df[output_column] = shifted_values

        safe_result = dict(result)

        safe_result.pop(
            "shifted_values",
            None,
        )

        safe_result["output_column"] = output_column

        return working_df, safe_result

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

        percentage_values = result.get(
            "percentage_change"
        )

        if percentage_values is None:
            return working_df, result

        if not isinstance(
            percentage_values,
            pd.Series,
        ):
            percentage_values = pd.Series(
                percentage_values,
                index=working_df.index,
            )

        working_df = working_df.copy()

        working_df[output_column] = percentage_values

        safe_result = dict(result)

        safe_result.pop(
            "percentage_change",
            None,
        )

        safe_result["output_column"] = output_column

        return working_df, safe_result

    def execute(
        self,
        df: pd.DataFrame,
        execution_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        working_df = df.copy()

        self.results = []

        for step in execution_steps:

            step_number = step.get("step")

            operation = step.get("operation")

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

            if not isinstance(parameters, dict):
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

            try:

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

                    "lag":
                        "create_shifted_column",

                    "shift_column":
                        "create_shifted_column",

                    "percentage_change":
                        "calculate_percentage_change",

                    "percent_change":
                        "calculate_percentage_change",
                }

                normalized_operation = operation_aliases.get(
                    normalized_operation,
                    normalized_operation,
                )

                updated_parameters = dict(parameters)

                # --------------------------------------------------
                # Revenue calculation
                # --------------------------------------------------

                if normalized_operation == "revenue_calculations":

                    tool = get_operation(
                        normalized_operation
                    )

                    result = tool(
                        df=working_df,
                        **updated_parameters,
                    )

                    revenue = result.get("revenue")

                    output_column = result.get(
                        "output_column",
                        "Revenue",
                    )

                    if revenue is not None:

                        working_df[
                            output_column
                        ] = revenue

                        if output_column != "Revenue":
                            working_df[
                                "Revenue"
                            ] = revenue

                    safe_result = dict(result)

                    safe_result.pop(
                        "revenue",
                        None,
                    )

                    self.results.append(
                        {
                            "step": step_number,
                            "operation": normalized_operation,
                            "status": "success",
                            "result": safe_result,
                        }
                    )

                    continue

                # --------------------------------------------------
                # Shift / lag operation
                # --------------------------------------------------

                if normalized_operation == (
                    "create_shifted_column"
                ):

                    sort_column = updated_parameters.get(
                        "sort_column"
                    )

                    ascending = updated_parameters.get(
                        "ascending",
                        True,
                    )

                    # If a sort column is requested, the working
                    # dataframe itself must follow that order so
                    # subsequent operations use the same sequence.
                    if sort_column is not None:

                        if sort_column not in working_df.columns:
                            raise ValueError(
                                f"Sort column "
                                f"'{sort_column}' "
                                "does not exist."
                            )

                        working_df = (
                            working_df
                            .sort_values(
                                by=sort_column,
                                ascending=ascending,
                                kind="stable",
                            )
                            .copy()
                        )

                    tool = get_operation(
                        normalized_operation
                    )

                    tool_parameters = dict(
                        updated_parameters
                    )

                    # The dataframe has already been sorted above.
                    # Avoid sorting twice inside the operation.
                    tool_parameters["sort_column"] = None

                    result = tool(
                        df=working_df,
                        **tool_parameters,
                    )

                    (
                        working_df,
                        safe_result,
                    ) = self._persist_shifted_column(
                        working_df=working_df,
                        result=result,
                        parameters=updated_parameters,
                    )

                    self.results.append(
                        {
                            "step": step_number,
                            "operation": normalized_operation,
                            "status": "success",
                            "result": safe_result,
                        }
                    )

                    continue

                # --------------------------------------------------
                # Group-by and max operations
                # --------------------------------------------------

                if normalized_operation in {
                    "groupby_aggregation",
                    "find_max",
                }:

                    (
                        working_df,
                        updated_parameters,
                    ) = self._resolve_revenue_reference(
                        working_df,
                        updated_parameters,
                    )

                # --------------------------------------------------
                # Chart operations
                # --------------------------------------------------

                if normalized_operation in {
                    "generate_bar_chart",
                    "generate_line_chart",
                }:

                    (
                        working_df,
                        result,
                    ) = self._execute_chart(
                        operation=normalized_operation,
                        df=working_df,
                        parameters=updated_parameters,
                    )

                    self.results.append(
                        {
                            "step": step_number,
                            "operation": normalized_operation,
                            "status": "success",
                            "result": result,
                        }
                    )

                    continue

                # --------------------------------------------------
                # Standard operation
                # --------------------------------------------------

                tool = get_operation(
                    normalized_operation
                )

                result = tool(
                    df=working_df,
                    **updated_parameters,
                )

                # --------------------------------------------------
                # Percentage change persistence
                # --------------------------------------------------

                if normalized_operation == (
                    "calculate_percentage_change"
                ):

                    (
                        working_df,
                        safe_result,
                    ) = self._persist_percentage_change(
                        working_df=working_df,
                        result=result,
                        parameters=updated_parameters,
                    )

                else:

                    if isinstance(result, dict):

                        safe_result = dict(result)

                        safe_result.pop(
                            "revenue",
                            None,
                        )

                        safe_result.pop(
                            "shifted_values",
                            None,
                        )

                    else:
                        safe_result = result

                self.results.append(
                    {
                        "step": step_number,
                        "operation": normalized_operation,
                        "status": "success",
                        "result": safe_result,
                    }
                )

            except Exception as exc:

                self.results.append(
                    {
                        "step": step_number,
                        "operation": operation,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        successful_steps = sum(
            1
            for item in self.results
            if item.get("status") == "success"
        )

        failed_steps = sum(
            1
            for item in self.results
            if item.get("status") == "error"
        )

        return {
            "status": (
                "success"
                if failed_steps == 0
                else "partial"
                if successful_steps > 0
                else "failed"
            ),
            "total_steps": len(
                execution_steps
            ),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "results": self.results,
        }
