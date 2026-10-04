import json
import re
from typing import Any, Dict

from src.llm.router import LLMRouter


class ResultInterpreter:
    """
    Decide whether executed analysis results are sufficient
    to answer the user's question.

    The interpreter prefers the LLM decision, but contains a
    deterministic fallback so malformed LLM output cannot crash
    the autonomous loop.
    """

    ALLOWED_DECISIONS = {
        "final",
        "continue",
        "repair",
    }

    AVAILABLE_OPERATIONS = {
        "calculate_revenue",
        "groupby_aggregate",
        "find_max",
        "calculate_statistics",
        "categorical_analysis",
        "rank_by_value",
        "calculate_percentage_change",
        "compare_columns",
        "create_shifted_column",
        "generate_bar_chart",
        "generate_line_chart",
    }

    def __init__(
        self,
        provider: str = "nvidia",
    ):
        self.provider = provider
        self.router = LLMRouter()

    # ==========================================================
    # PUBLIC INTERPRET
    # ==========================================================

    def interpret(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
    ) -> Dict[str, Any]:

        safe_results = self._sanitize(
            execution_results
        )

        system_prompt = """
You are the Result Interpreter of an autonomous
AI data analyst.

Your job is to inspect executed analysis results and
decide whether the user's question has been completely
answered.

Return ONLY valid JSON.

The JSON schema MUST be:

{
  "decision": "final" | "continue" | "repair",
  "reason": "short explanation",
  "next_operations": [
    {
      "operation": "one allowed operation",
      "parameters": {},
      "description": "optional explanation"
    }
  ]
}

Allowed operations:

- calculate_revenue
- groupby_aggregate
- find_max
- calculate_statistics
- categorical_analysis
- rank_by_value
- calculate_percentage_change
- compare_columns
- create_shifted_column
- generate_bar_chart
- generate_line_chart

Rules:

1. Use "final" only when the user's requested analysis
   is complete.

2. Use "continue" when an additional analysis operation
   is required.

3. Use "repair" when an executed operation failed and
   must be repaired.

4. If a requested chart has not been created yet,
   use "continue" and request the appropriate chart
   operation.

5. For a bar chart based on a previous group-by result,
   provide:
   - x_column
   - y_column
   - title

6. Never include raw dataframe rows as operation parameters.

7. Return JSON only.
"""

        user_prompt = f"""
USER QUESTION:
{user_query}

EXECUTED ANALYSIS RESULTS:
{json.dumps(
    safe_results,
    indent=2,
    ensure_ascii=False,
)}

Determine whether the user's question has been completely
answered.

Return JSON only.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            raw_response = self.router.generate(
                messages=messages,
                provider=self.provider,
                temperature=0.0,
                max_tokens=800,
            )

        except Exception as exc:
            return self._deterministic_fallback(
                user_query=user_query,
                execution_results=execution_results,
                reason=(
                    "LLM result interpretation failed: "
                    f"{exc}"
                ),
            )

        try:
            return self._parse_response(
                raw_response
            )

        except Exception as exc:
            return self._deterministic_fallback(
                user_query=user_query,
                execution_results=execution_results,
                reason=(
                    "LLM returned an invalid interpreter "
                    f"response: {exc}"
                ),
            )

    # ==========================================================
    # RESPONSE PARSER
    # ==========================================================

    def _parse_response(
        self,
        raw_response: str,
    ) -> Dict[str, Any]:

        if raw_response is None:
            raise ValueError(
                "Result interpreter returned no response."
            )

        if not isinstance(
            raw_response,
            str,
        ):
            raw_response = str(
                raw_response
            )

        text = raw_response.strip()

        if not text:
            raise ValueError(
                "Result interpreter returned an empty response."
            )

        parsed = self._extract_json(
            text
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Result interpreter response "
                "must be a JSON object."
            )

        decision = parsed.get(
            "decision"
        )

        reason = parsed.get(
            "reason"
        )

        next_operations = parsed.get(
            "next_operations"
        )

        if decision not in self.ALLOWED_DECISIONS:
            raise ValueError(
                f"Invalid interpreter decision: {decision}"
            )

        if not isinstance(
            reason,
            str,
        ):
            reason = str(
                reason or ""
            )

        reason = reason.strip()

        if not isinstance(
            next_operations,
            list,
        ):
            next_operations = []

        cleaned_operations = []

        for operation in next_operations:

            if isinstance(
                operation,
                str,
            ):
                operation = {
                    "operation": operation,
                    "parameters": {},
                }

            if not isinstance(
                operation,
                dict,
            ):
                continue

            operation_name = operation.get(
                "operation"
            )

            if not operation_name:
                continue

            operation_name = str(
                operation_name
            ).strip().lower()

            # Common LLM aliases
            aliases = {
                "create_bar_chart":
                    "generate_bar_chart",
                "bar_chart":
                    "generate_bar_chart",
                "create_line_chart":
                    "generate_line_chart",
                "line_chart":
                    "generate_line_chart",
                "group_by":
                    "groupby_aggregate",
                "groupby":
                    "groupby_aggregate",
                "max":
                    "find_max",
                "ranking":
                    "rank_by_value",
            }

            operation_name = aliases.get(
                operation_name,
                operation_name,
            )

            if operation_name not in (
                self.AVAILABLE_OPERATIONS
            ):
                continue

            parameters = operation.get(
                "parameters",
                {},
            )

            if not isinstance(
                parameters,
                dict,
            ):
                parameters = {}

            cleaned_operation = {
                "operation": operation_name,
                "parameters": parameters,
            }

            if operation.get(
                "description"
            ):
                cleaned_operation[
                    "description"
                ] = str(
                    operation[
                        "description"
                    ]
                )

            cleaned_operations.append(
                cleaned_operation
            )

        if decision == "final":
            cleaned_operations = []

        if (
            decision == "repair"
            and not cleaned_operations
        ):
            reason = (
                reason
                or
                "Execution requires repair, "
                "but no repair operation was provided."
            )

        if (
            decision == "continue"
            and not cleaned_operations
        ):
            reason = (
                reason
                or
                "Additional analysis was requested, "
                "but no valid continuation operations "
                "were provided."
            )

        return {
            "decision": decision,
            "reason": reason,
            "next_operations": cleaned_operations,
        }

    # ==========================================================
    # JSON EXTRACTION
    # ==========================================================

    def _extract_json(
        self,
        text: str,
    ) -> Dict[str, Any]:

        # ------------------------------------------------------
        # Direct JSON
        # ------------------------------------------------------

        try:
            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        # ------------------------------------------------------
        # Markdown fenced JSON
        # ------------------------------------------------------

        fenced_matches = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        for candidate in fenced_matches:

            try:
                parsed = json.loads(
                    candidate
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except json.JSONDecodeError:
                continue

        # ------------------------------------------------------
        # Embedded JSON object
        # ------------------------------------------------------

        decoder = json.JSONDecoder()

        for match in re.finditer(
            r"\{",
            text,
        ):

            start = match.start()

            try:
                parsed, _ = decoder.raw_decode(
                    text[start:]
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except json.JSONDecodeError:
                continue

        raise ValueError(
            "Result interpreter did not return valid JSON."
        )

    # ==========================================================
    # DETERMINISTIC FALLBACK
    # ==========================================================

    def _deterministic_fallback(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:

        results = execution_results.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            results = []

        query_lower = (
            str(user_query)
            .lower()
        )

        # ------------------------------------------------------
        # Failed operations → repair
        # ------------------------------------------------------

        failed_operations = []

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            if result.get(
                "status"
            ) == "error":

                operation = result.get(
                    "operation"
                )

                if operation:
                    failed_operations.append(
                        str(operation)
                    )

        if failed_operations:

            return {
                "decision": "repair",
                "reason": (
                    reason
                    + " Failed operations detected: "
                    + ", ".join(
                        failed_operations
                    )
                ),
                "next_operations": [],
            }

        # ------------------------------------------------------
        # Detect whether a chart was requested
        # ------------------------------------------------------

        chart_requested = any(
            keyword in query_lower
            for keyword in (
                "chart",
                "graph",
                "visual",
                "plot",
            )
        )

        chart_created = False

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            operation = str(
                result.get(
                    "operation",
                    ""
                )
            ).lower()

            if operation in {
                "generate_bar_chart",
                "generate_line_chart",
                "bar_chart",
                "line_chart",
            }:
                if result.get(
                    "status"
                ) == "success":
                    chart_created = True

        # ------------------------------------------------------
        # If chart is requested but missing,
        # infer chart from groupby result.
        # ------------------------------------------------------

        if (
            chart_requested
            and not chart_created
        ):

            group_column = None
            value_column = None

            for result in reversed(
                results
            ):

                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                operation = str(
                    result.get(
                        "operation",
                        ""
                    )
                ).lower()

                if operation not in {
                    "groupby_aggregate",
                    "groupby_aggregation",
                    "group_by",
                }:
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

                group_column = (
                    result_data.get(
                        "group_column"
                    )
                )

                value_column = (
                    result_data.get(
                        "value_column"
                    )
                )

                if (
                    group_column
                    and value_column
                ):
                    break

            if (
                group_column
                and value_column
            ):

                return {
                    "decision": "continue",
                    "reason": (
                        "The requested chart has not "
                        "been created yet."
                    ),
                    "next_operations": [
                        {
                            "operation":
                                "generate_bar_chart",
                            "parameters": {
                                "x_column":
                                    group_column,
                                "y_column":
                                    value_column,
                                "title": (
                                    f"{value_column} "
                                    f"by {group_column}"
                                ),
                            },
                            "description": (
                                "Create the requested "
                                "revenue-by-region chart."
                            ),
                        }
                    ],
                }

        # ------------------------------------------------------
        # Otherwise execution is sufficient.
        # ------------------------------------------------------

        return {
            "decision": "final",
            "reason": (
                "Execution completed successfully "
                "and no additional operation was "
                "deterministically required."
            ),
            "next_operations": [],
        }

    # ==========================================================
    # SANITIZATION
    # ==========================================================

    def _sanitize(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            dict,
        ):

            cleaned = {}

            for key, item in value.items():

                if key == "revenue":
                    continue

                cleaned[key] = self._sanitize(
                    item
                )

            return cleaned

        if isinstance(
            value,
            list,
        ):

            return [
                self._sanitize(item)
                for item in value
            ]

        if hasattr(
            value,
            "item",
        ):

            try:
                return value.item()

            except Exception:
                pass

        if hasattr(
            value,
            "tolist",
        ):

            try:
                return value.tolist()

            except Exception:
                pass

        if (
            value is None
            or isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
        ):
            return value

        return str(value)
