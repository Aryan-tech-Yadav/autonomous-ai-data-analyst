import json

import pandas as pd

from src.agent.pipeline import AnalysisPipeline


class FakePlanner:
    """
    Deterministic planner for pipeline integration testing.

    The real LLM planner returns text, so this fake planner
    intentionally returns a JSON string.
    """

    def create_plan(self, user_query, context):
        plan = {
            "analysis_plan": [
                {
                    "step": 1,
                    "operation": "revenue_calculations",
                    "parameters": {
                        "units_column": "Units Sold",
                        "price_column": "Unit Price",
                        "output_column": "Revenue",
                    },
                },
                {
                    "step": 2,
                    "operation": "rank_by_value",
                    "parameters": {
                        "group_column": "Region",
                        "value_column": "revenue",
                        "aggregation": "sum",
                        "ascending": False,
                        "top_n": 3,
                    },
                },
            ]
        }

        return json.dumps(plan)


class FakeResponseGenerator:
    """
    Deterministic response generator.

    No external LLM/API call is made.
    """

    def generate(self, user_query, execution_results):
        return (
            "Analysis completed successfully. "
            "Top regions were calculated."
        )


def build_test_dataframe():
    return pd.DataFrame(
        {
            "Region": [
                "North",
                "South",
                "East",
                "West",
                "Central",
            ],
            "Units Sold": [
                10,
                20,
                30,
                15,
                25,
            ],
            "Unit Price": [
                100,
                200,
                150,
                80,
                120,
            ],
        }
    )


def build_pipeline():
    pipeline = AnalysisPipeline(
        provider="nvidia"
    )

    pipeline.planner = FakePlanner()
    pipeline.response_generator = (
        FakeResponseGenerator()
    )

    return pipeline


def test_full_analysis_pipeline():
    df = build_test_dataframe()

    pipeline = build_pipeline()

    result = pipeline.run(
        df=df,
        user_query=(
            "Rank the regions by total revenue "
            "and show the top 3."
        ),
    )

    # ------------------------------------------------------
    # Pipeline status
    # ------------------------------------------------------

    assert result["status"] == "success"
    assert result["error"] is None

    # ------------------------------------------------------
    # Profile
    # ------------------------------------------------------

    assert result["profile"]["rows"] == 5
    assert result["profile"]["columns"] == 3

    # ------------------------------------------------------
    # Schema / context
    # ------------------------------------------------------

    assert result["schema"] is not None
    assert result["context"] is not None

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert result["validation"]["valid"] is True

    # ------------------------------------------------------
    # Execution
    # ------------------------------------------------------

    execution = result["execution"]

    assert execution["status"] == "success"
    assert execution["total_steps"] == 2
    assert execution["successful_steps"] == 2
    assert execution["failed_steps"] == 0

    # ------------------------------------------------------
    # Revenue calculation
    # ------------------------------------------------------

    revenue_step = execution["results"][0]

    assert revenue_step["status"] == "success"

    assert (
        revenue_step["result"]["total_revenue"]
        == 13700.0
    )

    # ------------------------------------------------------
    # Ranking
    # ------------------------------------------------------

    ranking_step = execution["results"][1]

    assert ranking_step["status"] == "success"

    ranking = ranking_step["result"]["results"]

    assert ranking[0]["group"] == "East"
    assert ranking[0]["value"] == 4500.0

    assert ranking[1]["group"] == "South"
    assert ranking[1]["value"] == 4000.0

    assert ranking[2]["group"] == "Central"
    assert ranking[2]["value"] == 3000.0

    # ------------------------------------------------------
    # Final response
    # ------------------------------------------------------

    assert result["final_response"] is not None
    assert "successfully" in result["final_response"]

    # ------------------------------------------------------
    # No execution recovery should be needed.
    #
    # PlanRepair should repair "revenue" during the
    # planning/validation stage before execution.
    # ------------------------------------------------------

    assert result["retry_performed"] is False


def test_pipeline_preserves_derived_revenue_repair():
    df = build_test_dataframe()

    pipeline = build_pipeline()

    result = pipeline.run(
        df=df,
        user_query="Rank regions by revenue.",
    )

    assert result["status"] == "success"

    execution = result["execution"]

    assert execution["status"] == "success"

    ranking_step = execution["results"][1]

    assert ranking_step["status"] == "success"

    assert (
        ranking_step["result"]["value_column"]
        == "Revenue"
    )
