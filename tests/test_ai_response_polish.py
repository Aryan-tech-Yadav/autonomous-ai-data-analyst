import pytest

from src.agent.response_generator import ResponseGenerator
from src.agent.result_interpreter import ResultInterpreter


class FakeFailingRouter:
    """Simulate an unavailable or failing LLM."""

    def generate(self, **kwargs):
        raise RuntimeError("Simulated LLM failure")


@pytest.fixture
def response_generator():
    generator = ResponseGenerator()
    generator.router = FakeFailingRouter()
    return generator


@pytest.fixture
def result_interpreter():
    interpreter = ResultInterpreter()
    interpreter.router = FakeFailingRouter()
    return interpreter


def test_response_generator_uses_verified_revenue_after_llm_failure(
    response_generator,
):
    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "calculate_revenue",
                "status": "success",
                "result": {
                    "operation": "calculate_revenue",
                    "total_revenue": 12500,
                    "rows_calculated": 25,
                },
            }
        ],
    }

    response = response_generator.generate(
        user_query="What is the total revenue?",
        execution_results=execution_results,
    )

    assert "12,500" in response
    assert "25 valid rows" in response
    assert "$" not in response
    assert "₹" not in response


def test_response_generator_does_not_present_failed_result_as_success(
    response_generator,
):
    execution_results = {
        "status": "error",
        "results": [
            {
                "operation": "calculate_revenue",
                "status": "error",
                "error": "Invalid column",
                "result": {
                    "operation": "calculate_revenue",
                    "total_revenue": 999999,
                },
            }
        ],
    }

    response = response_generator.generate(
        user_query="What is the total revenue?",
        execution_results=execution_results,
    )

    assert "999,999" not in response
    assert "not produce enough successful results" in response


def test_response_generator_does_not_invent_results_when_none_succeeded(
    response_generator,
):
    response = response_generator.generate(
        user_query="Which region performed best?",
        execution_results={
            "results": [
                {
                    "operation": "rank_by_value",
                    "status": "error",
                    "error": "Analysis failed",
                }
            ]
        },
    )

    assert "not produce enough successful results" in response
    assert "East" not in response
    assert "West" not in response


def test_interpreter_requests_repair_when_operation_failed(
    result_interpreter,
):
    decision = result_interpreter._deterministic_fallback(
        user_query="Calculate revenue",
        execution_results={
            "results": [
                {
                    "operation": "calculate_revenue",
                    "status": "error",
                    "error": "Invalid column",
                }
            ]
        },
        reason="LLM unavailable",
    )

    assert decision["decision"] == "repair"
    assert decision["next_operations"] == []
    assert "calculate_revenue" in decision["reason"]


def test_interpreter_finishes_after_successful_trend_analysis(
    result_interpreter,
):
    decision = result_interpreter._deterministic_fallback(
        user_query="Show the monthly trend",
        execution_results={
            "results": [
                {
                    "operation": "trend_analysis",
                    "status": "success",
                    "result": {
                        "results": [
                            {"period": "January", "value": 100},
                            {"period": "February", "value": 150},
                        ]
                    },
                }
            ]
        },
        reason="LLM unavailable",
    )

    assert decision["decision"] == "final"
    assert decision["next_operations"] == []


def test_interpreter_rejects_invalid_decision():
    interpreter = ResultInterpreter()

    with pytest.raises(ValueError):
        interpreter._parse_response(
            """
            {
                "decision": "invent",
                "reason": "Invalid decision",
                "next_operations": []
            }
            """
        )


def test_interpreter_normalizes_chart_operation_alias():
    interpreter = ResultInterpreter()

    parsed = interpreter._parse_response(
        """
        {
            "decision": "continue",
            "reason": "Create a chart",
            "next_operations": [
                {
                    "operation": "bar_chart",
                    "parameters": {
                        "x_column": "Region",
                        "y_column": "Revenue"
                    }
                }
            ]
        }
        """
    )

    assert parsed["decision"] == "continue"
    assert parsed["next_operations"][0]["operation"] == (
        "generate_bar_chart"
    )
    assert parsed["next_operations"][0]["parameters"] == {
        "x_column": "Region",
        "y_column": "Revenue",
    }
