import json
import re
from typing import Any, Dict

import pandas as pd

from src.agent.plan_repair import repair_plan
from src.agent.context import build_agent_context
from src.agent.llm_planner import LLMPlanner
from src.agent.plan_adapter import adapt_plan
from src.agent.plan_validator import validate_plan
from src.agent.response_generator import ResponseGenerator
from src.data.profiler import profile_dataset
from src.data.schema import build_schema
from src.execution.operation_executor import OperationExecutor


class AnalysisPipeline:
    """
    End-to-end autonomous AI data analysis pipeline.
    """

    def __init__(
        self,
        provider: str = "nvidia",
    ):
        self.provider = provider

        self.planner = LLMPlanner(
            provider=provider
        )

        self.executor = OperationExecutor()

        self.response_generator = (
            ResponseGenerator(
                provider=provider
            )
        )

    # ======================================================
    # JSON CLEANING
    # ======================================================

    def _clean_json_text(
        self,
        raw_text: str,
    ) -> str:
        """
        Remove Markdown code fences.
        """

        text = raw_text.strip()

        if text.startswith("```json"):

            text = text[
                len("```json"):
            ].strip()

            if text.endswith("```"):

                text = text[
                    :-3
                ].strip()

        elif text.startswith("```"):

            text = text[
                3:
            ].strip()

            if text.endswith("```"):

                text = text[
                    :-3
                ].strip()

        return text

    # ======================================================
    # MARKDOWN VALUE EXTRACTION
    # ======================================================

    def _extract_markdown_value(
        self,
        block: str,
        names: list[str],
    ) -> str | None:
        """
        Extract a quoted or unquoted Markdown parameter.

        Supports:

            `quantity_col`: "Units Sold"
            quantity_col: "Units Sold"
            `metric`: "max"
        """

        for name in names:

            pattern = (
                rf"`?{re.escape(name)}`?"
                rf"\s*:\s*"
                rf"(?:"
                rf"['\"]([^'\"]+)"
                rf"['\"]"
                rf"|"
                rf"`([^`]+)`"
                rf"|"
                rf"([^\n,]+)"
                rf")"
            )

            match = re.search(
                pattern,
                block,
                re.IGNORECASE,
            )

            if not match:
                continue

            value = (
                match.group(1)
                or match.group(2)
                or match.group(3)
            )

            if value is None:
                continue

            value = value.strip()

            value = value.rstrip(
                ".,"
            ).strip()

            if value:
                return value

        return None

    # ======================================================
    # MARKDOWN PLAN PARSER
    # ======================================================

    def _parse_markdown_plan(
        self,
        raw_text: str,
    ) -> Dict[str, Any]:
        """
        Convert Markdown planner output into
        the internal plan structure.
        """

        text = raw_text.strip()

        # --------------------------------------------------
        # Find numbered steps.
        # --------------------------------------------------

        matches = list(
            re.finditer(
                r"(?ms)^\s*(\d+)\.\s+"
                r"(.+?)"
                r"(?=^\s*\d+\.\s+|\Z)",
                text,
            )
        )

        if not matches:

            raise ValueError(
                "Markdown planner response did not "
                "contain numbered analysis steps."
            )

        analysis_plan = []

        for index, match in enumerate(
            matches,
            start=1,
        ):

            step_number = int(
                match.group(1)
            )

            block = match.group(2).strip()

            # ----------------------------------------------
            # Ignore numbered explanatory bullets.
            #
            # Example:
            # "Compute revenue per order..."
            #
            # This is explanatory text, not an executable
            # planner step.
            # ----------------------------------------------

            has_compact_operation = bool(
                re.search(
                    r"\\([^)]*?(?:→|->)\\s*`?[a-zA-Z0-9_]+`?\\)",
                    block,
                    flags=re.IGNORECASE,
                )
            )

            has_explicit_operation = bool(
                re.search(
                    r"(?im)^\\s*(?:[-*]\\s*)?(?:operation|sub_tool|type)\\s*:",
                    block,
                )
            )

            if (
                not has_compact_operation
                and not has_explicit_operation
                and "pandas_analysis" not in block.lower()
            ):
                continue

            # ----------------------------------------------
            # Tool
            # ----------------------------------------------

            tool = (
                self._extract_markdown_value(
                    block,
                    [
                        "Tool",
                        "tool",
                    ],
                )
                or "pandas_analysis"
            )

            # ----------------------------------------------
            # Operation
            # ----------------------------------------------

            operation = (
                self._extract_markdown_value(
                    block,
                    [
                        "Operation",
                        "operation",
                    ],
                )
            )

            # --------------------------------------------------
            # Detect operation from compact Markdown notation.
            #
            # Example:
            # (`pandas_analysis` → `revenue_calculations`)
            # --------------------------------------------------

            if not operation:

                compact_match = re.search(
                    r"\([^)]*?(?:→|->)\s*`?([a-zA-Z0-9_]+)`?\)",
                    block,
                    flags=re.IGNORECASE,
                )

                if compact_match:

                    operation = (
                        compact_match.group(1)
                        .strip()
                        .lower()
                    )

            if operation:

                operation = (
                    operation
                    .strip()
                    .lower()
                )

            # ----------------------------------------------
            # Parameters
            # ----------------------------------------------

            parameters: Dict[str, Any] = {}

            units_column = (
                self._extract_markdown_value(
                    block,
                    [
                        "quantity_col",
                        "quantity_column",
                        "units_column",
                        "unit_column",
                    ],
                )
            )

            if units_column:

                parameters[
                    "units_column"
                ] = units_column

            price_column = (
                self._extract_markdown_value(
                    block,
                    [
                        "price_col",
                        "price_column",
                    ],
                )
            )

            if price_column:

                parameters[
                    "price_column"
                ] = price_column

            group_column = (
                self._extract_markdown_value(
                    block,
                    [
                        "group_by",
                        "group_column",
                    ],
                )
            )

            if group_column:

                parameters[
                    "group_by"
                ] = group_column

            value_column = (
                self._extract_markdown_value(
                    block,
                    [
                        "value_col",
                        "value_column",
                        "metric_column",
                    ],
                )
            )

            if value_column:

                parameters[
                    "value_column"
                ] = value_column

            revenue_column = (
                self._extract_markdown_value(
                    block,
                    [
                        "revenue_column",
                    ],
                )
            )

            if revenue_column:

                parameters[
                    "revenue_column"
                ] = revenue_column

            revenue_name = (
                self._extract_markdown_value(
                    block,
                    [
                        "revenue_name",
                        "output_column",
                    ],
                )
            )

            if revenue_name:

                parameters[
                    "output_column"
                ] = revenue_name

            metric = (
                self._extract_markdown_value(
                    block,
                    [
                        "metric",
                    ],
                )
            )

            if metric:

                parameters[
                    "metric"
                ] = metric

            aggregation = (
                self._extract_markdown_value(
                    block,
                    [
                        "aggregation",
                    ],
                )
            )

            if aggregation:

                parameters[
                    "aggregation"
                ] = aggregation

            # ----------------------------------------------
            # Infer operation if omitted.
            # ----------------------------------------------

            block_lower = block.lower()

            if not operation:

                if (
                    "revenue_calculations"
                    in block_lower
                    or (
                        "revenue"
                        in block_lower
                        and "units sold"
                        in block_lower
                    )
                ):

                    operation = (
                        "revenue_calculations"
                    )

                elif (
                    "business_metrics"
                    in block_lower
                    or "highest"
                    in block_lower
                    or "maximum"
                    in block_lower
                    or "metric" in block_lower
                ):

                    operation = (
                        "business_metrics"
                    )

            # ----------------------------------------------
            # Normalize business metric intent.
            # ----------------------------------------------

            if (
                operation
                == "business_metrics"
            ):

                metric_value = str(
                    parameters.get(
                        "metric",
                        "",
                    )
                ).lower().strip()

                if metric_value in {
                    "max",
                    "maximum",
                    "highest",
                }:

                    parameters[
                        "aggregation"
                    ] = "max"

            # ----------------------------------------------
            # Description
            # ----------------------------------------------

            first_line = block.split(
                "\n",
                1,
            )[0].strip()

            first_line = re.sub(
                r"^\*\*(.*?)\*\*$",
                r"\1",
                first_line,
            )

            step = {
                "step": step_number or index,
                "tool": tool,
                "operation": operation,
                "parameters": parameters,
                "description": first_line,
            }

            analysis_plan.append(
                step
            )

        if not analysis_plan:

            raise ValueError(
                "No analysis steps could be extracted "
                "from Markdown planner response."
            )

        return {
            "analysis_plan": analysis_plan
        }

    # ======================================================
    # UNIVERSAL PLANNER RESPONSE PARSER
    # ======================================================

    def _parse_planner_response(
        self,
        raw_plan: str,
    ) -> Dict[str, Any]:
        """
        Parse planner output.

        Priority:

        1. Direct JSON
        2. JSON embedded in text
        3. Markdown fallback
        """

        if not isinstance(
            raw_plan,
            str,
        ):

            raise ValueError(
                "Planner response must be text."
            )

        cleaned = self._clean_json_text(
            raw_plan
        )

        # --------------------------------------------------
        # Direct JSON
        # --------------------------------------------------

        try:

            return json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            pass

        # --------------------------------------------------
        # Embedded JSON object
        # --------------------------------------------------

        object_start = cleaned.find(
            "{"
        )

        object_end = cleaned.rfind(
            "}"
        )

        if (
            object_start != -1
            and object_end > object_start
        ):

            candidate = cleaned[
                object_start:
                object_end + 1
            ]

            try:

                return json.loads(
                    candidate
                )

            except json.JSONDecodeError:

                pass

        # --------------------------------------------------
        # Embedded JSON list
        # --------------------------------------------------

        list_start = cleaned.find(
            "["
        )

        list_end = cleaned.rfind(
            "]"
        )

        if (
            list_start != -1
            and list_end > list_start
        ):

            candidate = cleaned[
                list_start:
                list_end + 1
            ]

            try:

                return json.loads(
                    candidate
                )

            except json.JSONDecodeError:

                pass

        # --------------------------------------------------
        # Markdown fallback
        # --------------------------------------------------

        return self._parse_markdown_plan(
            cleaned
        )

    # ======================================================
    # PLAN NORMALIZATION
    # ======================================================

    def _normalize_plan(
        self,
        raw_plan: Any,
    ) -> Dict[str, Any]:
        """
        Normalize planner output.
        """

        if isinstance(
            raw_plan,
            list,
        ):

            raw_plan = {
                "analysis_plan": raw_plan
            }

        if not isinstance(
            raw_plan,
            dict,
        ):

            raise ValueError(
                "LLM plan must be either a "
                "dictionary or a list."
            )

        analysis_plan = raw_plan.get(
            "analysis_plan"
        )

        if not isinstance(
            analysis_plan,
            list,
        ):

            raise ValueError(
                "LLM plan must contain an "
                "'analysis_plan' list."
            )

        normalized_steps = []

        for index, step in enumerate(
            analysis_plan,
            start=1,
        ):

            if not isinstance(
                step,
                dict,
            ):

                raise ValueError(
                    f"Plan step {index} "
                    "must be an object."
                )

            normalized_step = dict(
                step
            )

            parameters = normalized_step.get(
                "parameters",
                {},
            )

            if not isinstance(
                parameters,
                dict,
            ):

                parameters = {}

            parameters = dict(
                parameters
            )

            operation = (
                normalized_step.get(
                    "operation"
                )
                or normalized_step.get(
                    "sub_tool"
                )
                or normalized_step.get(
                    "subtype"
                )
            )

            if operation:

                operation = (
                    str(operation)
                    .lower()
                    .strip()
                )

            # ----------------------------------------------
            # Parameter aliases
            # ----------------------------------------------

            aliases = {
                "quantity_col": "units_column",
                "quantity_column": "units_column",
                "unit_column": "units_column",

                "price_col": "price_column",

                "value_col": "value_column",
                "metric_column": "value_column",

                "revenue_name": "output_column",
            }

            for old_key, new_key in aliases.items():

                if (
                    old_key in parameters
                    and new_key not in parameters
                ):

                    parameters[
                        new_key
                    ] = parameters[
                        old_key
                    ]

            # ----------------------------------------------
            # Business metrics normalization
            # ----------------------------------------------

            if (
                operation
                == "business_metrics"
            ):

                metric = str(
                    parameters.get(
                        "metric",
                        "",
                    )
                ).lower().strip()

                aggregation = str(
                    parameters.get(
                        "aggregation",
                        "",
                    )
                ).lower().strip()

                if metric in {
                    "max",
                    "maximum",
                    "highest",
                    "max_revenue_region",
                    "highest_revenue_region",
                } or aggregation == "max":

                    parameters[
                        "aggregation"
                    ] = "max"

            parameters.pop(
                "operation",
                None,
            )

            parameters.pop(
                "subtype",
                None,
            )

            normalized_step[
                "parameters"
            ] = parameters

            if operation:

                normalized_step[
                    "operation"
                ] = operation

            normalized_step.setdefault(
                "step",
                index,
            )

            normalized_steps.append(
                normalized_step
            )

        return {
            "analysis_plan": normalized_steps
        }

    # ======================================================
    # MAIN PIPELINE
    # ======================================================

    def run(
        self,
        df: pd.DataFrame,
        user_query: str,
    ) -> Dict[str, Any]:
        """
        Run the complete autonomous analysis pipeline.
        """

        profile = profile_dataset(
            df
        )

        schema = build_schema(
            df
        )

        context = build_agent_context(
            df=df,
            profile=profile,
            schema=schema,
            user_query=user_query,
        )

        # ==================================================
        # Planner
        # ==================================================

        try:

            raw_plan = self.planner.create_plan(
                user_query=user_query,
                context=context,
            )

        except Exception as e:

            return {
                "status": "planner_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": None,
                "plan": None,
                "validation": None,
                "adapted_plan": None,
                "execution": None,
                "final_response": None,
                "error": (
                    "Failed to generate analysis plan: "
                    f"{str(e)}"
                ),
            }

        # ==================================================
        # Parse
        # ==================================================

        try:

            parsed_plan = (
                self._parse_planner_response(
                    raw_plan
                )
            )

        except Exception as e:

            return {
                "status": "planner_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": raw_plan,
                "plan": None,
                "validation": None,
                "adapted_plan": None,
                "execution": None,
                "final_response": None,
                "error": (
                    "Nemotron returned an "
                    "unrecognized plan format: "
                    f"{str(e)}"
                ),
            }

        # ==================================================
        # Normalize
        # ==================================================

        try:

            plan = self._normalize_plan(
                parsed_plan
            )

        except Exception as e:

            return {
                "status": "plan_normalization_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": raw_plan,
                "plan": parsed_plan,
                "validation": None,
                "adapted_plan": None,
                "execution": None,
                "final_response": None,
                "error": (
                    "Failed to normalize "
                    f"LLM plan: {str(e)}"
                ),
            }

        # ==================================================
        # Validation
        # ==================================================

        validation = validate_plan(
            plan=plan,
            context=context,
        )

        # ==================================================
        # Plan Repair
        # ==================================================

        repair_result = None

        if not validation["valid"]:

            repair_result = repair_plan(
                plan=plan,
                validation_result=validation,
                context=context,
            )

            if repair_result["repaired"]:

                repaired_plan = repair_result["plan"]

                repaired_validation = validate_plan(
                    plan=repaired_plan,
                    context=context,
                )

                if repaired_validation["valid"]:

                    plan = repaired_plan
                    validation = repaired_validation

                else:

                    return {
                        "status": "validation_failed",
                        "profile": profile,
                        "schema": schema,
                        "context": context,
                        "raw_plan": raw_plan,
                        "plan": repaired_plan,
                        "validation": repaired_validation,
                        "repair": repair_result,
                        "adapted_plan": None,
                        "execution": None,
                        "final_response": None,
                        "error": (
                            "LLM-generated plan failed "
                            "validation and automatic "
                            "repair could not produce "
                            "a valid plan."
                        ),
                    }

            else:

                return {
                    "status": "validation_failed",
                    "profile": profile,
                    "schema": schema,
                    "context": context,
                    "raw_plan": raw_plan,
                    "plan": plan,
                    "validation": validation,
                    "repair": repair_result,
                    "adapted_plan": None,
                    "execution": None,
                    "final_response": None,
                    "error": (
                        "LLM-generated plan failed "
                        "validation and no automatic "
                        "repair was possible."
                    ),
                }

        # ==================================================
        # Adapter
        # ==================================================

        try:

            adapted_plan = adapt_plan(
                validation[
                    "validated_plan"
                ],
                context=context,
            )

        except Exception as e:

            return {
                "status": "adapter_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": raw_plan,
                "plan": plan,
                "validation": validation,
                "adapted_plan": None,
                "execution": None,
                "final_response": None,
                "error": (
                    "Failed to adapt the "
                    f"validated plan: {str(e)}"
                ),
            }

        # ==================================================
        # Execution
        # ==================================================

        execution = self.executor.execute(
            df=df,
            execution_steps=(
                adapted_plan[
                    "execution_steps"
                ]
            ),
        )

        if execution["status"] == "failed":

            return {
                "status": "execution_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": raw_plan,
                "plan": plan,
                "validation": validation,
                "adapted_plan": adapted_plan,
                "execution": execution,
                "final_response": None,
                "error": (
                    "Analysis execution failed."
                ),
            }

        # ==================================================
        # Final response
        # ==================================================

        try:

            final_response = (
                self.response_generator.generate(
                    user_query=user_query,
                    execution_results=execution,
                )
            )

        except Exception as e:

            return {
                "status": "response_error",
                "profile": profile,
                "schema": schema,
                "context": context,
                "raw_plan": raw_plan,
                "plan": plan,
                "validation": validation,
                "adapted_plan": adapted_plan,
                "execution": execution,
                "final_response": None,
                "error": (
                    "Failed to generate final "
                    f"response: {str(e)}"
                ),
            }

        status = "success"

        if execution["status"] == "partial_failure":

            status = "partial_success"

        return {
            "status": status,
            "profile": profile,
            "schema": schema,
            "context": context,
            "raw_plan": raw_plan,
            "plan": plan,
            "validation": validation,
            "adapted_plan": adapted_plan,
            "execution": execution,
            "final_response": final_response,
            "error": None,
        }
