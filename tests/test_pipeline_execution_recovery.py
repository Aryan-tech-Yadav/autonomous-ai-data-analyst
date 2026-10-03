import pandas as pd

from src.agent.pipeline import AnalysisPipeline
from src.execution.execution_recovery import (
    recover_execution_error,
)


class FailThenSuccessExecutor:
    """
    First execution fails.
    Second execution succeeds.

    Used to verify that recovery can repair
    a failed plan and allow exactly one retry.
    """

    def __init__(self):
        self.calls = 0

    def execute(self, df, execution_steps):
        self.calls += 1

        if self.calls == 1:
            return {
                "status": "failed",
                "total_steps": 1,
                "successful_steps": 0,
                "failed_steps": 1,
                "results": [
                    {
                        "step": 1,
                        "operation": "generate_bar_chart",
                        "status": "error",
                        "error": (
                            "Category column "
                            "'region' does not exist."
                        ),
                    }
                ],
            }

        return {
            "status": "success",
            "total_steps": 1,
            "successful_steps": 1,
            "failed_steps": 0,
            "results": [
                {
                    "step": 1,
                    "operation": "generate_bar_chart",
                    "status": "success",
                    "result": {
                        "chart_path": (
                            "reports/"
                            "recovery_pipeline_test.png"
                        )
                    },
                }
            ],
        }


class AlwaysFailExecutor:
    """
    Every execution fails.

    Used to verify that recovery does not
    create an infinite retry loop.
    """

    def __init__(self):
        self.calls = 0

    def execute(self, df, execution_steps):
        self.calls += 1

        return {
            "status": "failed",
            "total_steps": 1,
            "successful_steps": 0,
            "failed_steps": 1,
            "results": [
                {
                    "step": 1,
                    "operation": "generate_bar_chart",
                    "status": "error",
                    "error": (
                        "Category column "
                        "'region' does not exist."
                    ),
                }
            ],
        }


def build_test_dataframe():
    return pd.DataFrame(
        {
            "Region": [
                "North",
                "South",
                "West",
            ],
            "Total Revenue": [
                1000,
                2000,
                1500,
            ],
        }
    )


def build_test_context():
    return {
        "schema": {
            "columns": [
                {"name": "Region"},
                {"name": "Total Revenue"},
            ]
        }
    }


def build_broken_plan():
    return {
        "execution_steps": [
            {
                "step": 1,
                "operation": "generate_bar_chart",
                "parameters": {
                    "category_column": "region",
                    "value_column": "Revenue",
                    "output_path": (
                        "reports/"
                        "recovery_pipeline_test.png"
                    ),
                },
            }
        ]
    }


def test_execution_recovery_repairs_failed_plan():
    df = build_test_dataframe()
    context = build_test_context()
    plan = build_broken_plan()

    executor = FailThenSuccessExecutor()

    first_execution = executor.execute(
        df=df,
        execution_steps=plan["execution_steps"],
    )

    assert first_execution["status"] == "failed"
    assert first_execution["failed_steps"] == 1

    recovery = recover_execution_error(
        execution_result=first_execution,
        execution_plan=plan,
        context=context,
    )

    assert recovery["recoverable"] is True
    assert recovery["repaired_plan"] is not None

    repaired_step = (
        recovery["repaired_plan"]
        ["execution_steps"][0]
    )

    assert (
        repaired_step["parameters"]
        ["category_column"]
        == "Region"
    )

    assert (
        repaired_step["parameters"]
        ["value_column"]
        == "Total Revenue"
    )


def test_recovery_retry_succeeds():
    df = build_test_dataframe()
    context = build_test_context()
    plan = build_broken_plan()

    executor = FailThenSuccessExecutor()

    first_execution = executor.execute(
        df=df,
        execution_steps=plan["execution_steps"],
    )

    recovery = recover_execution_error(
        execution_result=first_execution,
        execution_plan=plan,
        context=context,
    )

    assert recovery["recoverable"] is True

    retry_execution = executor.execute(
        df=df,
        execution_steps=(
            recovery[
                "repaired_plan"
            ]["execution_steps"]
        ),
    )

    assert retry_execution["status"] == "success"
    assert retry_execution["successful_steps"] == 1
    assert executor.calls == 2


def test_recovery_stops_after_failed_retry():
    df = build_test_dataframe()
    context = build_test_context()
    plan = build_broken_plan()

    executor = AlwaysFailExecutor()

    first_execution = executor.execute(
        df=df,
        execution_steps=plan["execution_steps"],
    )

    assert first_execution["status"] == "failed"

    recovery = recover_execution_error(
        execution_result=first_execution,
        execution_plan=plan,
        context=context,
    )

    assert recovery["recoverable"] is True

    retry_execution = executor.execute(
        df=df,
        execution_steps=(
            recovery[
                "repaired_plan"
            ]["execution_steps"]
        ),
    )

    assert retry_execution["status"] == "failed"

    # The executor itself was called exactly twice:
    # initial attempt + one recovery retry.
    assert executor.calls == 2


def test_recovery_does_not_run_on_success():
    df = build_test_dataframe()
    context = build_test_context()
    plan = build_broken_plan()

    success_execution = {
        "status": "success",
        "total_steps": 1,
        "successful_steps": 1,
        "failed_steps": 0,
        "results": [
            {
                "step": 1,
                "operation": "generate_bar_chart",
                "status": "success",
                "result": {},
            }
        ],
    }

    recovery = recover_execution_error(
        execution_result=success_execution,
        execution_plan=plan,
        context=context,
    )

    assert recovery["recoverable"] is False
    assert recovery["repaired_plan"] is None
