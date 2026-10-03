from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from typing import Any

from src.agent.plan_validator import ALLOWED_OPERATIONS, ALLOWED_TOOLS
from src.tools.registry import get_operation, get_tool


class PlanRepair:
    """
    Deterministic repair layer for invalid LLM-generated analysis plans.

    The repair process tries safe, schema-aware corrections before
    asking the LLM to generate another plan.
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
        """
        Repair an invalid plan using deterministic rules.

        Returns:
            {
                "repaired": bool,
                "plan": repaired_plan,
                "changes": [...],
                "remaining_errors": [...]
            }
        """
        repaired_plan = deepcopy(plan)
        changes: list[str] = []

        analysis_plan = repaired_plan.get("analysis_plan")

        if not isinstance(analysis_plan, list):
            return {
                "repaired": False,
                "plan": repaired_plan,
                "changes": [],
                "remaining_errors": validation_result.get("errors", []),
            }

        for step in analysis_plan:
            if not isinstance(step, dict):
                continue

            self._repair_tool(step, changes)
            self._repair_operation(step, changes)
            self._repair_parameters(step, changes)

        return {
            "repaired": bool(changes),
            "plan": repaired_plan,
            "changes": changes,
            "remaining_errors": validation_result.get("errors", []),
        }

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_columns(self) -> list[str]:
        columns = self.schema.get("columns", [])

        result = []

        for column in columns:
            if isinstance(column, dict):
                name = column.get("name")
                if name:
                    result.append(str(name))

        return result

    def _find_column(self, requested: Any) -> str | None:
        if requested is None:
            return None

        requested = str(requested).strip()

        if not requested:
            return None

        # Exact match
        for column in self.columns:
            if column == requested:
                return column

        # Case-insensitive match
        normalized_requested = self._normalize(requested)

        for column in self.columns:
            if self._normalize(column) == normalized_requested:
                return column

        # Common semantic aliases
        aliases = {
            "revenue": [
                "Total Revenue",
                "Revenue",
            ],
            "total revenue": [
                "Total Revenue",
                "Revenue",
            ],
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

        normalized_alias = normalized_requested

        for candidate in aliases.get(normalized_alias, []):
            for column in self.columns:
                if self._normalize(column) == self._normalize(candidate):
                    return column

        # Fuzzy match
        matches = get_close_matches(
            requested,
            self.columns,
            n=1,
            cutoff=0.75,
        )

        if matches:
            return matches[0]

        return None

    @staticmethod
    def _normalize(value: str) -> str:
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
        tool = step.get("tool")

        if not tool:
            operation = step.get("operation")

            inferred_tool = self._tool_for_operation(operation)

            if inferred_tool:
                step["tool"] = inferred_tool
                changes.append(
                    f"Added missing tool '{inferred_tool}' "
                    f"for operation '{operation}'."
                )

            return

        if tool in ALLOWED_TOOLS:
            return

        normalized_tool = str(tool).strip().lower()

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

        replacement = aliases.get(normalized_tool)

        if replacement:
            step["tool"] = replacement
            changes.append(
                f"Repaired tool '{tool}' → '{replacement}'."
            )

    def _tool_for_operation(self, operation: Any) -> str | None:
        if not operation:
            return None

        operation = str(operation).strip()

        mapping = {
            "revenue_calculation": "pandas_analysis",
            "revenue_calculations": "pandas_analysis",
            "groupby_aggregation": "pandas_analysis",
            "group_by": "pandas_analysis",
            "find_max": "pandas_analysis",
            "statistics": "pandas_analysis",
            "calculate_statistics": "pandas_analysis",
            "categorical_analysis": "pandas_analysis",
            "generate_bar_chart": "chart_generator",
            "generate_line_chart": "chart_generator",
        }

        return mapping.get(operation)

    # ------------------------------------------------------------------
    # Operation repair
    # ------------------------------------------------------------------

    def _repair_operation(
        self,
        step: dict[str, Any],
        changes: list[str],
    ) -> None:
        operation = step.get("operation")

        if not operation:
            return

        operation_string = str(operation).strip()

        if operation_string in ALLOWED_OPERATIONS:
            return

        aliases = {
            "revenue": "revenue_calculations",
            "revenue_calculation": "revenue_calculations",
            "calculate_revenue": "revenue_calculations",
            "groupby": "groupby_aggregation",
            "group_by": "groupby_aggregation",
            "groupby_aggregate": "groupby_aggregation",
            "aggregation": "groupby_aggregation",
            "max": "find_max",
            "highest": "find_max",
            "stats": "statistics",
            "calculate_statistics": "statistics",
            "categorical": "categorical_analysis",
            "category_analysis": "categorical_analysis",
            "bar_chart": "generate_bar_chart",
            "bar": "generate_bar_chart",
            "chart_generator": "generate_bar_chart",
            "line_chart": "generate_line_chart",
            "line": "generate_line_chart",
        }

        normalized = (
            operation_string.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        replacement = aliases.get(normalized)

        if replacement:
            step["operation"] = replacement
            changes.append(
                f"Repaired operation '{operation_string}' "
                f"→ '{replacement}'."
            )

    # ------------------------------------------------------------------
    # Parameter repair
    # ------------------------------------------------------------------

    def _repair_parameters(
        self,
        step: dict[str, Any],
        changes: list[str],
    ) -> None:
        parameters = step.get("parameters")

        if not isinstance(parameters, dict):
            parameters = {}
            step["parameters"] = parameters

        operation = step.get("operation")

        if operation in {
            "revenue_calculation",
            "revenue_calculations",
        }:
            self._repair_revenue_parameters(parameters, changes)

        elif operation in {
            "groupby_aggregation",
            "group_by",
        }:
            self._repair_groupby_parameters(parameters, changes)

        elif operation == "find_max":
            self._repair_value_parameter(parameters, changes)

        elif operation in {
            "statistics",
            "calculate_statistics",
        }:
            self._repair_value_parameter(parameters, changes)

        elif operation == "categorical_analysis":
            self._repair_category_parameter(parameters, changes)

        elif operation == "generate_bar_chart":
            self._repair_bar_chart_parameters(parameters, changes)

        elif operation == "generate_line_chart":
            self._repair_line_chart_parameters(parameters, changes)

    def _repair_revenue_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
    ) -> None:
        for key in (
            "revenue_column",
            "value_column",
            "column",
        ):
            if key not in parameters:
                continue

            original = parameters[key]

            resolved = self._find_column(original)

            if resolved and resolved != original:
                parameters[key] = resolved
                changes.append(
                    f"Repaired column '{original}' → '{resolved}'."
                )

    def _repair_groupby_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
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
                resolved = self._find_column(original)

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
                resolved = self._find_column(original)

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
    ) -> None:
        for key in (
            "value_column",
            "metric_column",
            "column",
        ):
            if key not in parameters:
                continue

            original = parameters[key]
            resolved = self._find_column(original)

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
    ) -> None:
        for key in (
            "category_column",
            "column",
            "categorical_column",
        ):
            if key not in parameters:
                continue

            original = parameters[key]
            resolved = self._find_column(original)

            if resolved and resolved != original:
                parameters[key] = resolved
                changes.append(
                    f"Repaired category column "
                    f"'{original}' → '{resolved}'."
                )

            break

    def _repair_bar_chart_parameters(
        self,
        parameters: dict[str, Any],
        changes: list[str],
    ) -> None:
        self._repair_named_column(
            parameters,
            "category_column",
            changes,
        )

        self._repair_named_column(
            parameters,
            "value_column",
            changes,
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
    ) -> None:
        self._repair_named_column(
            parameters,
            "x_column",
            changes,
        )

        self._repair_named_column(
            parameters,
            "y_column",
            changes,
        )

        self._ensure_chart_output_path(
            parameters,
            "line",
            changes,
        )

    def _repair_named_column(
        self,
        parameters: dict[str, Any],
        key: str,
        changes: list[str],
    ) -> None:
        if key not in parameters:
            return

        original = parameters[key]
        resolved = self._find_column(original)

        if resolved and resolved != original:
            parameters[key] = resolved
            changes.append(
                f"Repaired {key} '{original}' → '{resolved}'."
            )

    def _ensure_chart_output_path(
        self,
        parameters: dict[str, Any],
        chart_type: str,
        changes: list[str],
    ) -> None:
        output_path = parameters.get("output_path")

        if output_path:
            return

        if chart_type == "bar":
            output_path = "reports/chart_repaired_bar.png"
        else:
            output_path = "reports/chart_repaired_line.png"

        parameters["output_path"] = output_path

        changes.append(
            f"Added missing chart output path '{output_path}'."
        )


def repair_plan(
    plan: dict[str, Any],
    validation_result: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function for repairing an invalid plan.
    """
    repairer = PlanRepair(context=context)

    return repairer.repair(
        plan=plan,
        validation_result=validation_result,
    )
