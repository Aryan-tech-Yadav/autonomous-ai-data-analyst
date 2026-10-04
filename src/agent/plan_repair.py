from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from typing import Any

from src.agent.plan_validator import (
    ALLOWED_OPERATIONS,
    ALLOWED_TOOLS,
)
from src.tools.registry import get_operation, get_tool


SHIFT_OPERATIONS = {
    "create_shifted_column",
    "shift_column",
    "shift",
    "lag",
}


class PlanRepair:
    """
    Deterministic repair layer for invalid LLM-generated analysis plans.

    The repair process tries safe, schema-aware corrections before
    asking the LLM to generate another plan.

    The repair layer is plan-aware:
    columns created by an earlier step can be referenced by
    later steps.
    """

    def __init__(self, context: dict[str, Any]):
        self.context = context
        self.schema = context.get("schema", {})
        self.columns = self._get_columns()

    def repair(
        self,
        plan: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:

        repaired_plan = deepcopy(plan)
        changes: list[str] = []

        analysis_plan = repaired_plan.get(
            "analysis_plan"
        )

        if not isinstance(
            analysis_plan,
            list,
        ):

            return {
                "repaired": False,
                "plan": repaired_plan,
                "changes": [],
                "remaining_errors": validation_result.get(
                    "errors",
                    [],
                ),
            }

        # Columns created by earlier steps.
        derived_columns: set[str] = set()

        for step in analysis_plan:

            if not isinstance(
                step,
                dict,
            ):
                continue

            self._repair_tool(
                step,
                changes,
            )

            self._repair_operation(
                step,
                changes,
            )

            self._repair_parameters(
                step,
                changes,
                derived_columns,
            )

            self._register_derived_column(
                step,
                derived_columns,
            )

        return {
            "repaired": bool(changes),
            "plan": repaired_plan,
            "changes": changes,
            "remaining_errors": validation_result.get(
                "errors",
                [],
            ),
        }

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_columns(self) -> list[str]:

        columns = self.schema.get(
            "columns",
            [],
        )

        result = []

        for column in columns:

            if isinstance(
                column,
                dict,
            ):

                name = column.get(
                    "name"
                )

                if name:
                    result.append(
                        str(name)
                    )

        return result

    def _find_column(
        self,
        requested: Any,
        extra_columns: set[str] | None = None,
    ) -> str | None:

        if requested is None:
            return None

        requested = str(
            requested
        ).strip()

        if not requested:
            return None

        available_columns = list(
            self.columns
        )

        if extra_columns:
            available_columns.extend(
                extra_columns
            )

        # ----------------------------------------------------------
        # Exact match
        # ----------------------------------------------------------

        for column in available_columns:

            if column == requested:
                return column

        # ----------------------------------------------------------
        # Case-insensitive match
        # ----------------------------------------------------------

        normalized_requested = self._normalize(
            requested
        )

        for column in available_columns:

            if (
                self._normalize(column)
                == normalized_requested
            ):

                return column

        # ----------------------------------------------------------
        # Revenue references
        # ----------------------------------------------------------

        derived_revenue_aliases = {
            "revenue",
            "total revenue",
            "total_revenue",
            "calculated revenue",
            "calculated_revenue",
        }

        if normalized_requested in derived_revenue_aliases:

            # Prefer an existing Total Revenue column.
            for column in self.columns:

                if (
                    self._normalize(column)
                    == "total revenue"
                ):

                    return column

            # Prefer an existing Revenue column.
            for column in self.columns:

                if (
                    self._normalize(column)
                    == "revenue"
                ):

                    return column

            # Check derived columns.
            if extra_columns:

                for column in extra_columns:

                    if (
                        self._normalize(column)
                        == "revenue"
                    ):

                        return column

            # Revenue can be created by a previous step.
            return "Revenue"

        # ----------------------------------------------------------
        # Common semantic aliases
        # ----------------------------------------------------------

        aliases = {
            "sales": [
                "Total Revenue",
                "Revenue",
                "Sales",
            ],
            "region": [
                "Region",
            ],
            "date": [
                "Date",
            ],
            "time": [
                "Date",
            ],
            "units": [
                "Units Sold",
            ],
            "units sold": [
                "Units Sold",
            ],
            "price": [
                "Unit Price",
            ],
            "unit price": [
                "Unit Price",
            ],
            "sales rep": [
                "Sales Rep",
            ],
            "product category": [
                "Product Category",
            ],
        }

        for candidate in aliases.get(
            normalized_requested,
            [],
        ):

            for column in available_columns:

                if (
                    self._normalize(column)
                    == self._normalize(candidate)
                ):

                    return column

        # ----------------------------------------------------------
        # Fuzzy match
        # ----------------------------------------------------------

        matches = get_close_matches(
            requested,
            available_columns,
            n=1,
            cutoff=0.75,
        )

        if matches:
            return matches[0]

        return None

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:

        return (
            value.lower()
            .strip()
            .replace("_", " ")
            .replace("-", " ")
        )

    # ------------------------------------------------------------------
    # Tool repair
    # ------------------------------------------------------------------

    def _repair_tool(
        self,
        step: dict[str, Any],
        changes: list[str],
    ) -> None:

        tool = step.get(
            "tool"
        )

        if not tool:

            operation = step.get(
                "operation"
            )

            inferred_tool = self._tool_for_operation(
                operation
            )

            if inferred_tool:

                step["tool"] = inferred_tool

                changes.append(
                    f"Added missing tool "
                    f"'{inferred_tool}' "
                    f"for operation "
                    f"'{operation}'."
                )

            return

        if tool in ALLOWED_TOOLS:
            return

        normalized_tool = str(
            tool
        ).strip().lower()

        aliases = {
            "pandas": "pandas_analysis",
            "pandas analysis": "pandas_analysis",
            "dataframe": "pandas_analysis",
            "data analysis": "pandas_analysis",
            "time": "time_analysis",
            "time series": "time_analysis",
            "charts": "chart_generator",
            "chart": "chart_generator",
            "bar": "bar_chart",
            "bar chart": "bar_chart",
            "line": "line_chart",
            "line chart": "line_chart",
        }

        replacement = aliases.get(
            normalized_tool
        )

        if replacement:

            step["tool"] = replacement

            changes.append(
                f"Repaired tool '{tool}' "
                f"→ '{replacement}'."
            )

    def _tool_for_operation(
        self,
        operation: Any,
    ) -> str | None:

        if not operation:
            return None

        operation = str(
            operation
        ).strip()

        mapping = {
            "revenue_calculation":
                "pandas_analysis",

            "revenue_calculations":
                "pandas_analysis",

            "groupby_aggregation":
                "pandas_analysis",

            "group_by":
                "pandas_analysis",

            "find_max":
                "pandas_analysis",

            "statistics":
                "pandas_analysis",

            "calculate_statistics":
                "pandas_analysis",

            "categorical_analysis":
                "pandas_analysis",

            "rank_by_value":
                "pandas_analysis",

            "ranking":
                "pandas_analysis",

            "calculate_percentage_change":
                "pandas_analysis",

            "percentage_change":
                "pandas_analysis",

            "compare_columns":
                "pandas_analysis",

            "comparison":
                "pandas_analysis",

            "create_shifted_column":
                "pandas_analysis",

            "shift_column":
                "pandas_analysis",

            "shift":
                "pandas_analysis",

            "lag":
                "pandas_analysis",

            "generate_bar_chart":
                "chart_generator",

            "generate_line_chart":
                "chart_generator",
        }

        return mapping.get(
            operation
        )

    # ------------------------------------------------------------------
    # Operation repair
    # ------------------------------------------------------------------

    def _repair_operation(
        self,
        step: dict[str, Any],
        changes: list[str],
    ) -> None:

        operation = step.get(
            "operation"
        )

        if not operation:
            return

        operation_string = str(
            operation
        ).strip()

        normalized = (
            operation_string
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        allowed_normalized = {
            str(item)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            for item in ALLOWED_OPERATIONS
        }

        if normalized in allowed_normalized:

            # Canonicalize aliases.
            canonical_aliases = {
                "shift": "create_shifted_column",
                "lag": "create_shifted_column",
                "shift_column": "create_shifted_column",
                "ranking": "rank_by_value",
                "percentage_change":
                    "calculate_percentage_change",
                "comparison": "compare_columns",
                "revenue_calculation":
                    "revenue_calculations",
                "group_by":
                    "groupby_aggregation",
            }

            canonical = canonical_aliases.get(
                normalized
            )

            if canonical and canonical != operation_string:

                step["operation"] = canonical

                changes.append(
                    f"Normalized operation "
                    f"'{operation_string}' "
                    f"→ '{canonical}'."
                )

            return

        aliases = {
            "revenue":
                "revenue_calculations",

            "revenue_calculation":
                "revenue_calculations",

            "calculate_revenue":
                "revenue_calculations",

            "groupby":
                "groupby_aggregation",

            "group_by":
                "groupby_aggregation",

            "groupby_aggregate":
                "groupby_aggregation",

            "aggregation":
                "groupby_aggregation",

            "max":
                "find_max",

            "highest":
                "find_max",

            "stats":
                "statistics",

            "calculate_statistics":
                "statistics",

            "categorical":
                "categorical_analysis",

            "category_analysis":
                "categorical_analysis",

            "rank":
                "rank_by_value",

            "ranking":
                "rank_by_value",

            "rank_by":
                "rank_by_value",

            "percentage_change":
                "calculate_percentage_change",

            "percent_change":
                "calculate_percentage_change",

            "calculate_percent_change":
                "calculate_percentage_change",

            "comparison":
                "compare_columns",

            "compare":
                "compare_columns",

            "compare_column":
                "compare_columns",

            "shift":
                "create_shifted_column",

            "lag":
                "create_shifted_column",

            "shift_column":
                "create_shifted_column",

            "previous_value":
                "create_shifted_column",

            "previous_order":
                "create_shifted_column",

            "prior_value":
                "create_shifted_column",

            "bar_chart":
                "generate_bar_chart",

            "bar":
                "generate_bar_chart",

            "chart_generator":
                "generate_bar_chart",

            "line_chart":
                "generate_line_chart",

            "line":
                "generate_line_chart",
        }

        replacement = aliases.get(
            normalized
        )

        if replacement:

            step["operation"] = replacement

            changes.append(
                f"Repaired operation "
                f"'{operation_string}' "
                f"→ '{replacement}'."
            )

    # ------------------------------------------------------------------
    # Parameter repair
    # ------------------------------------------------------------------

    def _repair_parameters(
        self,
        step: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        parameters = step.get(
            "parameters"
        )

        if not isinstance(
            parameters,
            dict,
        ):

            parameters = {}

            step["parameters"] = parameters

        operation = step.get(
            "operation"
        )

        if operation in {
            "revenue_calculation",
            "revenue_calculations",
        }:

            self._repair_revenue_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation in {
            "groupby_aggregation",
            "group_by",
        }:

            self._repair_groupby_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "find_max":

            self._repair_value_parameter(
                parameters,
                changes,
                derived_columns,
            )

        elif operation in {
            "statistics",
            "calculate_statistics",
        }:

            self._repair_value_parameter(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "categorical_analysis":

            self._repair_category_parameter(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "rank_by_value":

            self._repair_rank_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "calculate_percentage_change":

            self._repair_percentage_change_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "compare_columns":

            self._repair_compare_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation in SHIFT_OPERATIONS:

            self._repair_shift_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "generate_bar_chart":

            self._repair_bar_chart_parameters(
                parameters,
                changes,
                derived_columns,
            )

        elif operation == "generate_line_chart":

            self._repair_line_chart_parameters(
                parameters,
                changes,
                derived_columns,
            )

    # ------------------------------------------------------------------
    # Existing operation parameter repair
    # ------------------------------------------------------------------

    def _repair_revenue_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        for key in (
            "revenue_column",
            "value_column",
            "column",
            "units_column",
            "price_column",
        ):

            if key not in parameters:
                continue

            original = parameters[key]

            resolved = self._find_column(
                original,
                derived_columns,
            )

            if resolved and resolved != original:

                parameters[key] = resolved

                changes.append(
                    f"Repaired column "
                    f"'{original}' → '{resolved}'."
                )

    def _repair_groupby_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        category_keys = (
            "group_by_column",
            "groupby_column",
            "category_column",
            "group_column",
        )

        for key in category_keys:

            if key in parameters:

                original = parameters[key]

                resolved = self._find_column(
                    original,
                    derived_columns,
                )

                if resolved and resolved != original:

                    parameters[key] = resolved

                    changes.append(
                        f"Repaired grouping column "
                        f"'{original}' → '{resolved}'."
                    )

                break

        value_keys = (
            "value_column",
            "metric_column",
            "aggregation_column",
        )

        for key in value_keys:

            if key in parameters:

                original = parameters[key]

                resolved = self._find_column(
                    original,
                    derived_columns,
                )

                if resolved and resolved != original:

                    parameters[key] = resolved

                    changes.append(
                        f"Repaired metric column "
                        f"'{original}' → '{resolved}'."
                    )

                break

    def _repair_value_parameter(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        for key in (
            "value_column",
            "metric_column",
            "column",
        ):

            if key not in parameters:
                continue

            original = parameters[key]

            resolved = self._find_column(
                original,
                derived_columns,
            )

            if resolved and resolved != original:

                parameters[key] = resolved

                changes.append(
                    f"Repaired value column "
                    f"'{original}' → '{resolved}'."
                )

            break

    def _repair_category_parameter(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        for key in (
            "category_column",
            "column",
            "categorical_column",
        ):

            if key not in parameters:
                continue

            original = parameters[key]

            resolved = self._find_column(
                original,
                derived_columns,
            )

            if resolved and resolved != original:

                parameters[key] = resolved

                changes.append(
                    f"Repaired category column "
                    f"'{original}' → '{resolved}'."
                )

            break

    # ------------------------------------------------------------------
    # Advanced analytics
    # ------------------------------------------------------------------

    def _repair_rank_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        self._repair_named_column(
            parameters,
            "group_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "value_column",
            changes,
            derived_columns,
        )

        if "group_column" not in parameters:

            for alias in (
                "group_by_column",
                "category_column",
                "groupby_column",
            ):

                if alias in parameters:

                    parameters[
                        "group_column"
                    ] = parameters.pop(alias)

                    changes.append(
                        f"Mapped '{alias}' "
                        "to 'group_column'."
                    )

                    self._repair_named_column(
                        parameters,
                        "group_column",
                        changes,
                        derived_columns,
                    )

                    break

        if "value_column" not in parameters:

            for alias in (
                "metric_column",
                "aggregation_column",
            ):

                if alias in parameters:

                    parameters[
                        "value_column"
                    ] = parameters.pop(alias)

                    changes.append(
                        f"Mapped '{alias}' "
                        "to 'value_column'."
                    )

                    self._repair_named_column(
                        parameters,
                        "value_column",
                        changes,
                        derived_columns,
                    )

                    break

        aggregation = parameters.get(
            "aggregation"
        )

        if aggregation is not None:

            normalized = str(
                aggregation
            ).strip().lower()

            aggregation_aliases = {
                "total": "sum",
                "total_sum": "sum",
                "average": "mean",
                "avg": "mean",
                "maximum": "max",
                "minimum": "min",
                "count_rows": "count",
            }

            replacement = aggregation_aliases.get(
                normalized
            )

            if (
                replacement
                and replacement != aggregation
            ):

                parameters[
                    "aggregation"
                ] = replacement

                changes.append(
                    f"Repaired aggregation "
                    f"'{aggregation}' "
                    f"→ '{replacement}'."
                )

        top_n = parameters.get(
            "top_n"
        )

        if top_n is not None:

            try:

                repaired_top_n = int(
                    top_n
                )

                if repaired_top_n != top_n:

                    parameters[
                        "top_n"
                    ] = repaired_top_n

                    changes.append(
                        f"Converted top_n "
                        f"'{top_n}' → "
                        f"'{repaired_top_n}'."
                    )

            except (
                TypeError,
                ValueError,
            ):

                parameters.pop(
                    "top_n",
                    None,
                )

                changes.append(
                    f"Removed invalid "
                    f"top_n '{top_n}'."
                )

    def _repair_percentage_change_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        aliases = {
            "current": "current_column",
            "current_value": "current_column",
            "current_period": "current_column",
            "new_column": "current_column",
            "latest_column": "current_column",

            "previous": "previous_column",
            "previous_value": "previous_column",
            "previous_period": "previous_column",
            "old_column": "previous_column",
            "prior_column": "previous_column",
        }

        for alias, canonical in aliases.items():

            if (
                alias in parameters
                and canonical not in parameters
            ):

                parameters[
                    canonical
                ] = parameters.pop(alias)

                changes.append(
                    f"Mapped '{alias}' "
                    f"to '{canonical}'."
                )

        self._repair_named_column(
            parameters,
            "current_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "previous_column",
            changes,
            derived_columns,
        )

        if "output_column" not in parameters:

            parameters[
                "output_column"
            ] = "Percentage Change"

            changes.append(
                "Added missing output column "
                "'Percentage Change'."
            )

    def _repair_compare_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        aliases = {
            "left": "left_column",
            "left_value": "left_column",
            "first_column": "left_column",
            "column_1": "left_column",

            "right": "right_column",
            "right_value": "right_column",
            "second_column": "right_column",
            "column_2": "right_column",
        }

        for alias, canonical in aliases.items():

            if (
                alias in parameters
                and canonical not in parameters
            ):

                parameters[
                    canonical
                ] = parameters.pop(alias)

                changes.append(
                    f"Mapped '{alias}' "
                    f"to '{canonical}'."
                )

        self._repair_named_column(
            parameters,
            "left_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "right_column",
            changes,
            derived_columns,
        )

    # ------------------------------------------------------------------
    # Shift / lag repair
    # ------------------------------------------------------------------

    def _repair_shift_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        aliases = {
            "column": "source_column",
            "source": "source_column",
            "value_column": "source_column",
            "current_column": "source_column",

            "new_column": "output_column",
            "target_column": "output_column",
            "name": "output_column",

            "order_by": "sort_column",
            "sort_by": "sort_column",

            "lag": "periods",
            "shift": "periods",
        }

        for alias, canonical in aliases.items():

            if (
                alias in parameters
                and canonical not in parameters
            ):

                parameters[
                    canonical
                ] = parameters.pop(alias)

                changes.append(
                    f"Mapped '{alias}' "
                    f"to '{canonical}'."
                )

        self._repair_named_column(
            parameters,
            "source_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "sort_column",
            changes,
            derived_columns,
        )

        if "periods" not in parameters:

            parameters[
                "periods"
            ] = 1

            changes.append(
                "Added default shift period "
                "'1'."
            )

        else:

            try:

                original = parameters[
                    "periods"
                ]

                repaired = int(
                    original
                )

                if repaired == 0:
                    repaired = 1

                if repaired != original:

                    parameters[
                        "periods"
                    ] = repaired

                    changes.append(
                        f"Repaired periods "
                        f"'{original}' → "
                        f"'{repaired}'."
                    )

            except (
                TypeError,
                ValueError,
            ):

                parameters[
                    "periods"
                ] = 1

                changes.append(
                    "Replaced invalid shift "
                    "periods with '1'."
                )

        if "output_column" not in parameters:

            source = parameters.get(
                "source_column",
                "Value",
            )

            parameters[
                "output_column"
            ] = f"Previous {source}"

            changes.append(
                f"Added shift output column "
                f"'{parameters['output_column']}'."
            )

    # ------------------------------------------------------------------
    # Chart parameter repair
    # ------------------------------------------------------------------

    def _repair_bar_chart_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        self._repair_named_column(
            parameters,
            "category_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "value_column",
            changes,
            derived_columns,
        )

        self._ensure_chart_output_path(
            parameters,
            "bar",
            changes,
        )

    def _repair_line_chart_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        self._repair_named_column(
            parameters,
            "x_column",
            changes,
            derived_columns,
        )

        self._repair_named_column(
            parameters,
            "y_column",
            changes,
            derived_columns,
        )

        self._ensure_chart_output_path(
            parameters,
            "line",
            changes,
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _repair_named_column(
        self,
        parameters: dict[str, Any],
        key: str,
        changes: list[str],
        derived_columns: set[str],
    ) -> None:

        if key not in parameters:
            return

        original = parameters[key]

        resolved = self._find_column(
            original,
            derived_columns,
        )

        if resolved and resolved != original:

            parameters[key] = resolved

            changes.append(
                f"Repaired {key} "
                f"'{original}' → "
                f"'{resolved}'."
            )

    def _register_derived_column(
        self,
        step: dict[str, Any],
        derived_columns: set[str],
    ) -> None:

        operation = str(
            step.get(
                "operation",
                "",
            )
        ).strip().lower()

        parameters = step.get(
            "parameters",
            {},
        )

        if not isinstance(
            parameters,
            dict,
        ):
            return

        if operation in {
            "create_shifted_column",
            "shift_column",
            "shift",
            "lag",
        }:

            output_column = (
                parameters.get(
                    "output_column"
                )
                or parameters.get(
                    "new_column"
                )
            )

            if isinstance(
                output_column,
                str,
            ) and output_column.strip():

                derived_columns.add(
                    output_column.strip()
                )

        elif operation in {
            "revenue_calculations",
            "revenue_calculation",
        }:

            output_column = parameters.get(
                "output_column",
                "Revenue",
            )

            if isinstance(
                output_column,
                str,
            ):

                derived_columns.add(
                    output_column
                )

                derived_columns.add(
                    "Revenue"
                )

        elif operation == "calculate_percentage_change":

            output_column = parameters.get(
                "output_column",
                "Percentage Change",
            )

            if isinstance(
                output_column,
                str,
            ):

                derived_columns.add(
                    output_column
                )

    def _ensure_chart_output_path(
        self,
        parameters: dict[str, Any],
        chart_type: str,
        changes: list[str],
    ) -> None:

        output_path = parameters.get(
            "output_path"
        )

        if output_path:
            return

        if chart_type == "bar":

            output_path = (
                "reports/chart_repaired_bar.png"
            )

        else:

            output_path = (
                "reports/chart_repaired_line.png"
            )

        parameters[
            "output_path"
        ] = output_path

        changes.append(
            f"Added missing chart output path "
            f"'{output_path}'."
        )


def repair_plan(
    plan: dict[str, Any],
    validation_result: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:

    repairer = PlanRepair(
        context=context
    )

    return repairer.repair(
        plan=plan,
        validation_result=validation_result,
    )
