import json
from typing import Any, Dict

from src.llm.router import LLMRouter


class ResponseGenerator:
    """
    Generate the final natural-language response from
    deterministic and autonomous analysis results.

    The generator is responsible only for explaining
    already-executed results. It must never perform
    new analysis or invent missing facts.
    """

    def __init__(
        self,
        provider: str = "nvidia",
    ):
        self.provider = provider
        self.router = LLMRouter()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def generate(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
    ) -> str:
        """
        Generate a concise business-friendly answer.

        Falls back to deterministic formatting if the LLM
        is unavailable or returns an unusable response.
        """

        safe_results = self._sanitize(
            execution_results
        )

        if not safe_results:
            return (
                "I could not generate an answer because "
                "no analysis results were available."
            )

        system_prompt = """
You are the final response engine of an
autonomous AI data analyst.

Your job is to explain analysis results that
have ALREADY been executed.

The execution results are authoritative.

STRICT RULES:

1. Answer the user's original question directly.

2. Use ONLY facts contained in the provided
   execution results.

3. Never invent numbers, rankings, trends,
   categories, dates, business facts, or conclusions.

4. Never perform additional calculations that
   are not explicitly supported by the results.

5. Never invent a currency symbol or currency code.

6. If the dataset does not explicitly identify
   a currency, report monetary values without
   assuming INR, USD, EUR, GBP, or another currency.

7. Preserve the meaning and values of the
   execution results.

8. You may format large numbers with commas
   for readability.

9. If a regional or categorical breakdown exists,
   include the important breakdown when relevant.

10. If a maximum/minimum result exists, clearly
    identify it.

11. If a chart was generated and an output path
    exists, mention that the chart was generated
    and provide the path.

12. Do not claim that a chart exists unless the
    execution results explicitly contain an
    output_path.

13. Do not expose internal implementation details.

14. Never mention:
    - planner
    - adapter
    - executor
    - validator
    - LLM
    - model
    - API
    - prompt
    - autonomous loop

15. If the execution results are incomplete,
    explicitly state what could not be determined.

16. Do not apologize unnecessarily.

17. Keep the final answer concise, clear,
    professional, and business-friendly.

18. Return ONLY the final natural-language answer.
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


Write the final answer using ONLY the
executed analysis results above.
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
            response = self.router.generate(
                messages=messages,
                provider=self.provider,
                temperature=0.2,
                max_tokens=1200,
            )

            response = (
                response.strip()
                if isinstance(response, str)
                else ""
            )

            if response:
                return response

        except Exception:
            pass

        return self._deterministic_fallback(
            user_query=user_query,
            execution_results=safe_results,
        )

    # ==========================================================
    # SANITIZATION
    # ==========================================================

    def _sanitize(
        self,
        value: Any,
    ) -> Any:
        """
        Convert execution results into JSON-safe data.

        Internal pandas Series/DataFrame objects and
        potentially huge intermediate values are removed.
        """

        if isinstance(value, dict):

            sanitized = {}

            for key, item in value.items():

                key_string = str(key)

                # Internal intermediate revenue Series
                # should never be sent to the LLM.
                if key_string == "revenue":
                    continue

                # Internal dataframe-like objects should
                # not enter the final prompt.
                if key_string in {
                    "working_df",
                    "dataframe",
                    "df",
                }:
                    continue

                sanitized[key_string] = (
                    self._sanitize(item)
                )

            return sanitized

        if isinstance(value, list):

            return [
                self._sanitize(item)
                for item in value
            ]

        if isinstance(value, tuple):

            return [
                self._sanitize(item)
                for item in value
            ]

        if hasattr(value, "item"):

            try:
                return value.item()

            except Exception:
                pass

        if hasattr(value, "tolist"):

            try:
                converted = value.tolist()

                return self._sanitize(
                    converted
                )

            except Exception:
                pass

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ) or value is None:

            return value

        return str(value)

    # ==========================================================
    # DETERMINISTIC FALLBACK
    # ==========================================================

    def _deterministic_fallback(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
    ) -> str:
        """
        Produce a useful deterministic answer when
        the LLM is unavailable.

        This fallback uses only explicit execution results.
        """

        results = execution_results.get(
            "results",
            [],
        )

        if not isinstance(results, list):
            return (
                "The analysis completed, but a readable "
                "summary could not be generated."
            )

        successful_results = []

        for item in results:

            if not isinstance(item, dict):
                continue

            if item.get("status") != "success":
                continue

            result = item.get(
                "result",
                {},
            )

            if isinstance(result, dict):
                successful_results.append(result)

        if not successful_results:

            return (
                "The analysis did not produce enough "
                "successful results to answer the question."
            )

        sections = []

        # ------------------------------------------------------
        # Revenue calculation
        # ------------------------------------------------------

        for result in successful_results:

            operation = result.get(
                "operation"
            )

            if operation == "calculate_revenue":

                total_revenue = result.get(
                    "total_revenue"
                )

                rows_calculated = result.get(
                    "rows_calculated"
                )

                if total_revenue is not None:

                    message = (
                        f"Total calculated revenue is "
                        f"{self._format_number(total_revenue)}"
                    )

                    if rows_calculated is not None:

                        message += (
                            f" across "
                            f"{rows_calculated} valid rows."
                        )

                    sections.append(
                        message
                    )

        # ------------------------------------------------------
        # GroupBy results
        # ------------------------------------------------------

        for result in successful_results:

            operation = result.get(
                "operation"
            )

            if operation not in {
                "groupby_aggregate",
                "groupby_aggregation",
            }:
                continue

            rows = result.get(
                "results"
            )

            group_column = result.get(
                "group_column",
                "category",
            )

            value_column = result.get(
                "value_column",
                "value",
            )

            if not isinstance(rows, list):
                continue

            valid_rows = [
                row
                for row in rows
                if isinstance(row, dict)
            ]

            if not valid_rows:
                continue

            breakdown_parts = []

            for row in valid_rows:

                group_value = row.get(
                    group_column
                )

                value = row.get(
                    value_column
                )

                if (
                    group_value is None
                    or value is None
                ):
                    continue

                breakdown_parts.append(
                    f"{group_value}: "
                    f"{self._format_number(value)}"
                )

            if breakdown_parts:

                sections.append(
                    f"{value_column} by "
                    f"{group_column}: "
                    + "; ".join(
                        breakdown_parts
                    )
                    + "."
                )

        # ------------------------------------------------------
        # Find maximum
        # ------------------------------------------------------

        for result in successful_results:

            if result.get(
                "operation"
            ) != "find_max":
                continue

            group = result.get(
                "group"
            )

            value = result.get(
                "value"
            )

            if group is None or value is None:
                continue

            sections.append(
                f"The highest value is "
                f"{self._format_number(value)} "
                f"for {group}."
            )

        # ------------------------------------------------------
        # Chart
        # ------------------------------------------------------

        for result in successful_results:

            operation = result.get(
                "operation"
            )

            if operation not in {
                "generate_bar_chart",
                "generate_line_chart",
            }:
                continue

            output_path = result.get(
                "output_path"
            )

            if output_path:

                chart_type = result.get(
                    "chart_type",
                    "chart",
                )

                sections.append(
                    f"A {chart_type} chart was generated "
                    f"and saved to {output_path}."
                )

        # ------------------------------------------------------
        # Final fallback
        # ------------------------------------------------------

        if not sections:

            return (
                "The analysis completed successfully, "
                "but the available results do not contain "
                "enough information for a concise summary."
            )

        return "\n\n".join(
            sections
        )

    # ==========================================================
    # NUMBER FORMATTING
    # ==========================================================

    @staticmethod
    def _format_number(
        value: Any,
    ) -> str:
        """
        Format numeric values without inventing
        currency information.
        """

        if isinstance(
            value,
            bool,
        ):
            return str(value)

        if isinstance(
            value,
            int,
        ):
            return f"{value:,}"

        if isinstance(
            value,
            float,
        ):

            if value.is_integer():
                return f"{int(value):,}"

            return f"{value:,.2f}"

        try:

            numeric_value = float(value)

            if numeric_value.is_integer():
                return f"{int(numeric_value):,}"

            return f"{numeric_value:,.2f}"

        except (
            TypeError,
            ValueError,
        ):

            return str(value)
