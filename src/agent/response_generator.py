import json
from typing import Any, Dict

from src.llm.router import LLMRouter


class ResponseGenerator:
    """
    Generate the final natural-language response from
    deterministic analysis results.
    """

    def __init__(
        self,
        provider: str = "nvidia",
    ):
        self.provider = provider
        self.router = LLMRouter()

    def generate(
        self,
        user_query: str,
        execution_results: Dict[str, Any],
    ) -> str:

        safe_results = self._sanitize(
            execution_results
        )

        system_prompt = """
You are the final response engine of an
autonomous AI data analyst.

Your job is to explain analysis results that
have already been executed.

STRICT RULES:

1. Answer the user's original question directly.

2. Use ONLY the provided execution results.

3. Never invent numbers, columns, rankings,
   trends, business facts, or conclusions.

4. Never invent a currency symbol or currency code.

5. If the dataset does not explicitly provide
   currency information, report numeric monetary
   values without a currency symbol.

6. Do not assume USD, INR, EUR, GBP, or any
   other currency.

7. Preserve the exact numeric values provided
   by the execution results, using normal
   readable formatting when appropriate.

8. If a regional/category breakdown is provided,
   include it when useful to answer the question.

9. Do not describe internal Python implementation
   details.

10. Do not mention internal planner, adapter,
    validator, executor, model, or API details.

11. If the execution results are insufficient,
    clearly say what information is missing.

12. Keep the response concise and business-friendly.

Return only the final natural-language answer.
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


Provide the final answer using only these results.
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

        return self.router.generate(
            messages=messages,
            provider=self.provider,
            temperature=0.2,
            max_tokens=1200,
        )

    def _sanitize(
        self,
        value: Any,
    ) -> Any:

        if isinstance(value, dict):

            return {
                str(key): self._sanitize(item)
                for key, item in value.items()
                if key != "revenue"
            }

        if isinstance(value, list):

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
                return value.tolist()

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
