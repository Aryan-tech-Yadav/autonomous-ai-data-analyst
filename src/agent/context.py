import pandas as pd


def build_agent_context(
    df: pd.DataFrame,
    profile: dict,
    schema: dict,
    user_query: str = "",
) -> dict:
    """
    Create a structured context package
    for the AI agent.
    """

    context = {

        "dataset": {
            "rows": profile["rows"],
            "columns": profile["columns"],
        },

        "data_quality": {
            "missing_values": profile["missing_values"],
            "duplicate_rows": profile["duplicate_rows"],
        },

        "schema": schema,

        "sample_data": (
            df.head(5)
            .to_dict(orient="records")
        ),

        "user_query": user_query,

    }

    return context
