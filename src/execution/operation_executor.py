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


def _normalize_column_name(
    value: Any,
) -> str:
    return str(value).strip().lower()


def _is_derived_revenue_column(
    value: Any,
) -> bool:

    return (
        _normalize_column_name(value)
        in DERIVED_REVENUE_ALIASES
    )


class OperationExecutor:
    """
    Execute validated analysis operations sequentially.

    The executor keeps a working DataFrame so that
    derived columns such as Revenue can be created
    and reused by later operations.
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

        working_df["Revenue"] = (
            units * prices
        )

        return working_df

    def _resolve_revenue_reference(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        working_df = df

        updated_parameters = dict(
            parameters
        )

        value_column = (
            updated_parameters.get(
                "value_column"
            )
        )

        if _is_derived_revenue_column(
            value_column
        ):

            working_df = (
                self._ensure_revenue_column(
                    working_df
                )
            )

            updated_parameters[
                "value_column"
            ] = "Revenue"

        return (
            working_df,
            updated_parameters,
        )

    def _execute_chart(
        self,
        operation: str,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:

        working_df = df

        updated_parameters = dict(
            parameters
        )

        if operation == "generate_bar_chart":

            value_column = (
                updated_parameters.get(
                    "value_column"
                )
            )

            if _is_derived_revenue_column(
                value_column
            ):

                working_df = (
                    self._ensure_revenue_column(
                        working_df
                    )
                )

                updated_parameters[
                    "value_column"
                ] = "Revenue"

        elif operation == "generate_line_chart":

            y_column = (
                updated_parameters.get(
                    "y_column"
                )
            )

            if _is_derived_revenue_column(
                y_column
            ):

                working_df = (
                    self._ensure_revenue_column(
                        working_df
                    )
                )

                updated_parameters[
                    "y_column"
                ] = "Revenue"

        tool = get_operation(
            operation
        )

        result = tool(
            df=working_df,
            **updated_parameters,
        )

        return (
            working_df,
            result,
        )

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
                        "error": (
                            "Missing operation."
                        ),
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

            try:

                normalized_operation = (
                    str(operation)
                    .strip()
                    .lower()
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
                }

                normalized_operation = (
                    operation_aliases.get(
                        normalized_operation,
                        normalized_operation,
                    )
                )

                updated_parameters = dict(
                    parameters
                )

                # --------------------------------------------------
                # Revenue calculation
                # --------------------------------------------------

                if normalized_operation == (
                    "revenue_calculations"
                ):

                    tool = get_operation(
                        normalized_operation
                    )

                    result = tool(
                        df=working_df,
                        **updated_parameters,
                    )

                    revenue = result.get(
                        "revenue"
                    )

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

                    safe_result = dict(
                        result
                    )

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
                # Group-by and max operations
                # --------------------------------------------------

                if normalized_operation in {
                    "groupby_aggregation",
                    "find_max",
                }:

                    (
                        working_df,
                        updated_parameters,
                    ) = (
                        self._resolve_revenue_reference(
                            working_df,
                            updated_parameters,
                        )
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
                # Standard operations
                # --------------------------------------------------

                tool = get_operation(
                    normalized_operation
                )

                result = tool(
                    df=working_df,
                    **updated_parameters,
                )

                safe_result = result

                if isinstance(
                    result,
                    dict,
                ):

                    safe_result = dict(
                        result
                    )

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
