from __future__ import annotations

import json
from typing import Any, Dict, List

from src.llm.router import LLMRouter


class ResponseGenerator:
    """
    Production response engine for the autonomous AI data analyst.

    Responsibilities:
    - Explain already-executed analysis results.
    - Use verified business insights when available.
    - Resolve conversational references from supplied history.
    - Never perform new dataframe analysis.
    - Never invent unsupported facts.
    - Provide a deterministic fallback when the LLM is unavailable.
    """

    def __init__(self, provider: str = "nvidia"):
        self.provider = provider
        self.router = LLMRouter()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def generate(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
        insights: list[dict] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        safe_results = self._sanitize(execution_results)
        safe_insights = self._sanitize(insights or [])
        history = conversation_history or []

        has_results = bool(
            isinstance(safe_results, dict)
            and safe_results.get("results")
        )

        if not has_results and history:
            return self._generate_follow_up_from_history(
                user_query=user_query,
                conversation_history=history,
            )

        if not has_results:
            return (
                "I could not generate an answer because "
                "no analysis results were available."
            )

        system_prompt = self._build_system_prompt()

        user_prompt = self._build_user_prompt(
            user_query=user_query,
            execution_results=safe_results,
            insights=safe_insights,
            conversation_history=history,
        )

        try:
            response = self.router.generate(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                provider=self.provider,
                temperature=0.15,
                max_tokens=1200,
            )

            response = (
                response.strip()
                if isinstance(response, str)
                else ""
            )

            if self._is_usable_response(response):
                return response

        except Exception:
            pass

        return self._deterministic_fallback(
            user_query=user_query,
            execution_results=safe_results,
            insights=safe_insights,
        )

    # ==========================================================
    # LLM PROMPTS
    # ==========================================================

    def _build_system_prompt(self) -> str:
        return """
You are the final business response engine of an autonomous AI
data analyst.

Your job is to explain analysis results that have ALREADY been
executed and verified.

The execution results and verified insights are authoritative.

STRICT RULES:

1. Answer the user's question directly.

2. Use ONLY information contained in the supplied execution
   results, verified insights, and conversation history.

3. Never invent numbers, rankings, categories, dates, trends,
   business facts, causes, or recommendations.

4. Never perform new calculations unless the exact calculation
   is already explicitly represented by the supplied results.

5. Never invent or assume a currency.

6. If currency is not explicitly identified, report monetary
   values without a currency symbol or currency code.

7. Prefer verified business insights when they directly answer
   the user's question.

8. Do not dump every available result. Select only the facts
   relevant to the user's question.

9. For ranking questions, clearly identify the relevant highest,
   lowest, or requested ranked items.

10. For trend questions, mention the important high/low periods
    and verified overall movement when available.

11. For comparison questions, clearly state the verified
    comparison.

12. For categorical questions, identify the relevant dominant
    category when verified.

13. If a chart was actually generated and an output_path exists,
    mention it only when relevant.

14. Do not claim a chart exists unless output_path is explicitly
    present.

15. If the results are incomplete, clearly say what cannot be
    determined.

16. Resolve conversational references such as "it", "its",
    "that", "those", "the highest", "the lowest", "previous",
    and "earlier" using the supplied conversation history.

17. Do not perform new dataset analysis.

18. Do not mention internal implementation details.

Never mention:
- planner
- adapter
- executor
- validator
- model
- API
- prompt
- autonomous loop
- internal pipeline

19. Keep the response concise, clear, professional, and
    business-friendly.

20. Return ONLY the final answer.
""".strip()

    def _build_user_prompt(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
        insights: list[dict],
        conversation_history: list[dict],
    ) -> str:
        return f"""
USER QUESTION:
{user_query}

PREVIOUS CONVERSATION:
{json.dumps(
    conversation_history,
    indent=2,
    ensure_ascii=False,
)}

EXECUTED ANALYSIS RESULTS:
{json.dumps(
    execution_results,
    indent=2,
    ensure_ascii=False,
)}

VERIFIED BUSINESS INSIGHTS:
{json.dumps(
    insights,
    indent=2,
    ensure_ascii=False,
)}

RESPONSE INSTRUCTIONS:

First determine what the user is asking.

Then select only the verified facts needed to answer that
question.

If the current execution results answer the question, prefer
them over conversation history.

If the current execution results do not contain the answer but
the conversation history explicitly contains it, answer from
the history.

Do not invent missing information.

Write only the final business-friendly answer.
""".strip()

    def _generate_follow_up_from_history(
        self,
        user_query: str,
        conversation_history: list[dict],
    ) -> str:
        system_prompt = """
You are a conversational response engine for a data analyst.

The current question has no new analysis results.

Answer ONLY from the supplied previous conversation.

Resolve references such as:
- it
- its
- that
- those
- the highest
- the lowest
- previous
- earlier

STRICT RULES:

1. Never invent facts or numbers.
2. Never perform new dataset analysis.
3. Never create a chart.
4. If the history does not contain enough information, clearly
   say that the previous conversation does not contain enough
   information.
5. Keep the answer concise.
6. Do not mention internal implementation details.
7. Return only the final answer.
""".strip()

        user_prompt = f"""
PREVIOUS CONVERSATION:
{json.dumps(
    conversation_history,
    indent=2,
    ensure_ascii=False,
)}

CURRENT QUESTION:
{user_query}
""".strip()

        try:
            response = self.router.generate(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                provider=self.provider,
                temperature=0.1,
                max_tokens=800,
            )

            response = (
                response.strip()
                if isinstance(response, str)
                else ""
            )

            if self._is_usable_response(response):
                return response

        except Exception:
            pass

        return (
            "I could not resolve that question from "
            "the previous conversation."
        )

    # ==========================================================
    # SANITIZATION
    # ==========================================================

    def _sanitize(self, value: Any) -> Any:
        """
        Convert results into JSON-safe prompt data.

        Large/internal dataframe objects and intermediate revenue
        Series are excluded from the final LLM context.
        """

        if isinstance(value, dict):
            sanitized = {}

            for key, item in value.items():
                key_string = str(key)

                if key_string in {
                    "revenue",
                    "working_df",
                    "dataframe",
                    "df",
                }:
                    continue

                sanitized[key_string] = self._sanitize(item)

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
                return self._sanitize(value.tolist())
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
    # RESPONSE QUALITY
    # ==========================================================

    @staticmethod
    def _is_usable_response(response: str) -> bool:
        if not response:
            return False

        cleaned = response.strip()

        if len(cleaned) < 3:
            return False

        blocked = {
            "i cannot answer",
            "i can't answer",
            "unable to answer",
            "no answer",
        }

        if cleaned.lower() in blocked:
            return False

        return True

    # ==========================================================
    # DETERMINISTIC FALLBACK
    # ==========================================================

    def _deterministic_fallback(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
        insights: list[dict] | None = None,
    ) -> str:
        """
        Deterministic fallback.

        Uses ONLY verified execution results and insights.
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

            result = item.get("result", {})

            if isinstance(result, dict):
                successful_results.append(result)

        if not successful_results:
            return (
                "The analysis did not produce enough "
                "successful results to answer the question."
            )

        sections: List[str] = []

        # ------------------------------------------------------
        # Verified insights
        # ------------------------------------------------------

        self._append_verified_insights(
            sections,
            insights or [],
        )

        # ------------------------------------------------------
        # Revenue
        # ------------------------------------------------------

        for result in successful_results:
            if result.get("operation") != "calculate_revenue":
                continue

            total_revenue = result.get("total_revenue")
            rows_calculated = result.get("rows_calculated")

            if total_revenue is None:
                continue

            message = (
                f"Total calculated revenue is "
                f"{self._format_number(total_revenue)}"
            )

            if rows_calculated is not None:
                message += (
                    f" across {rows_calculated} valid rows."
                )
            else:
                message += "."

            self._append_unique(sections, message)

        # ------------------------------------------------------
        # Ranking
        # ------------------------------------------------------

        for result in successful_results:
            if result.get("operation") != "rank_by_value":
                continue

            rows = result.get("results", [])

            if not isinstance(rows, list) or not rows:
                continue

            valid_rows = [
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("group") is not None
                and row.get("value") is not None
            ]

            if not valid_rows:
                continue

            top = valid_rows[0]

            message = (
                f"{top['group']} ranks highest with "
                f"{self._format_number(top['value'])}."
            )

            self._append_unique(sections, message)

            break

        # ------------------------------------------------------
        # GroupBy
        # ------------------------------------------------------

        for result in successful_results:
            operation = result.get("operation")

            if operation not in {
                "groupby_aggregate",
                "groupby_aggregation",
            }:
                continue

            rows = result.get("results")
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

            breakdown = []

            for row in valid_rows:
                group_value = row.get(group_column)
                value = row.get(value_column)

                if group_value is None or value is None:
                    continue

                breakdown.append(
                    f"{group_value}: "
                    f"{self._format_number(value)}"
                )

            if breakdown:
                self._append_unique(
                    sections,
                    (
                        f"{value_column} by {group_column}: "
                        + "; ".join(breakdown)
                        + "."
                    ),
                )

        # ------------------------------------------------------
        # Find maximum
        # ------------------------------------------------------

        for result in successful_results:
            if result.get("operation") != "find_max":
                continue

            group = result.get("group")
            value = result.get("value")

            if group is None or value is None:
                continue

            self._append_unique(
                sections,
                (
                    f"The highest value is "
                    f"{self._format_number(value)} "
                    f"for {group}."
                ),
            )

        # ------------------------------------------------------
        # Trend
        # ------------------------------------------------------

        for result in successful_results:
            if result.get("operation") != "trend_analysis":
                continue

            rows = result.get("results", [])

            if not isinstance(rows, list) or not rows:
                continue

            valid_rows = [
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("period") is not None
                and row.get("value") is not None
            ]

            if not valid_rows:
                continue

            highest = max(
                valid_rows,
                key=lambda row: float(row["value"]),
            )

            lowest = min(
                valid_rows,
                key=lambda row: float(row["value"]),
            )

            trend_message = (
                f"Highest period: {highest['period']} "
                f"({self._format_number(highest['value'])}). "
            )

            if len(valid_rows) > 1:
                trend_message += (
                    f"Lowest period: {lowest['period']} "
                    f"({self._format_number(lowest['value'])})."
                )

            self._append_unique(
                sections,
                trend_message,
            )

        # ------------------------------------------------------
        # Charts
        # ------------------------------------------------------

        for result in successful_results:
            operation = result.get("operation")

            if operation not in {
                "generate_bar_chart",
                "generate_line_chart",
            }:
                continue

            output_path = result.get("output_path")

            if not output_path:
                continue

            chart_type = result.get(
                "chart_type",
                "chart",
            )

            self._append_unique(
                sections,
                (
                    f"A {chart_type} chart was generated "
                    f"and saved to {output_path}."
                ),
            )

        if not sections:
            return (
                "The analysis completed successfully, "
                "but the available results do not contain "
                "enough information for a concise summary."
            )

        return "\n\n".join(sections)

    # ==========================================================
    # INSIGHT HELPERS
    # ==========================================================

    @staticmethod
    def _append_verified_insights(
        sections: List[str],
        insights: list[dict],
    ) -> None:
        """
        Add verified insight messages in priority order.

        Existing insight dictionaries use `message`; older
        structures may use `statement` or `insight`.
        """

        priority = {
            "maximum": 10,
            "highest_performer": 20,
            "lowest_performer": 30,
            "performance_gap": 40,
            "trend_highest_period": 50,
            "trend_lowest_period": 60,
            "trend_overall_change": 70,
            "trend_direction": 80,
            "group_highest": 90,
            "group_lowest": 100,
            "dominant_category": 110,
            "average_percentage_change": 120,
            "percentage_change_range": 130,
            "mean_median_relationship": 140,
        }

        usable = []

        for insight in insights:
            if not isinstance(insight, dict):
                continue

            message = (
                insight.get("message")
                or insight.get("statement")
                or insight.get("insight")
            )

            if not message:
                continue

            insight_type = str(
                insight.get("type", "")
            )

            usable.append(
                (
                    priority.get(
                        insight_type,
                        999,
                    ),
                    str(message),
                )
            )

        usable.sort(key=lambda item: item[0])

        for _, message in usable:
            if message not in sections:
                sections.append(message)

    @staticmethod
    def _append_unique(
        sections: List[str],
        message: str,
    ) -> None:
        if message and message not in sections:
            sections.append(message)

    # ==========================================================
    # NUMBER FORMATTING
    # ==========================================================

    @staticmethod
    def _format_number(value: Any) -> str:
        """
        Format numbers without assuming currency.
        """

        if isinstance(value, bool):
            return str(value)

        if isinstance(value, int):
            return f"{value:,}"

        if isinstance(value, float):
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
