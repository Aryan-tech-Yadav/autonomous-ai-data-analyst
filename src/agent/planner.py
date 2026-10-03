from typing import Dict, List


def create_analysis_plan(
    user_query: str,
    context: Dict
) -> Dict:
    """
    Create an analysis plan from user query
    and dataset context.
    """

    query = user_query.lower()

    steps: List[str] = []

    tools: List[str] = []


    # --------------------------------
    # Detect required analysis
    # --------------------------------

    if "sales" in query or "revenue" in query:

        steps.append(
            "Identify sales related columns"
        )

        steps.append(
            "Calculate sales metrics"
        )

        tools.append(
            "pandas_analysis"
        )


    if (
        "trend" in query
        or "time" in query
        or "date" in query
    ):

        steps.append(
            "Analyze time based patterns"
        )

        tools.append(
            "time_analysis"
        )


    if (
        "chart" in query
        or "visual"
        in query
    ):

        steps.append(
            "Generate visualization"
        )

        tools.append(
            "chart_generator"
        )


    # --------------------------------
    # Default plan
    # --------------------------------

    if not steps:

        steps = [
            "Inspect dataset structure",
            "Find important columns",
            "Generate basic statistics",
            "Provide insights"
        ]

        tools.append(
            "general_analysis"
        )


    plan = {

        "user_query": user_query,

        "steps": steps,

        "required_tools": tools,

        "dataset_size": {
            "rows": context["dataset"]["rows"],
            "columns": context["dataset"]["columns"]
        }

    }


    return plan
