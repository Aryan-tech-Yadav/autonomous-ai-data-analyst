import pandas as pd


def detect_column_role(
    column_name: str,
    series: pd.Series,
) -> str:
    """
    Estimate the role of a dataset column.
    """

    name = column_name.lower().strip()

    # Identifier detection
    if (
        name == "id"
        or name.endswith(" id")
        or name.endswith("_id")
        or "identifier" in name
    ):
        return "identifier"

    # Date detection
    if (
        "date" in name
        or "time" in name
    ):
        return "date_or_time"

    # Numeric detection
    if pd.api.types.is_numeric_dtype(series):

        if series.nunique(dropna=True) <= 20:
            return "numeric_discrete"

        return "numeric"

    # Categorical / text detection
    unique_count = series.nunique(dropna=True)

    if unique_count <= 20:
        return "categorical"

    return "text"


def build_schema(
    df: pd.DataFrame,
) -> dict:
    """
    Build an AI-friendly schema from a DataFrame.
    """

    schema = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": [],
    }

    for column in df.columns:

        series = df[column]

        column_schema = {
            "name": column,
            "dtype": str(series.dtype),
            "role": detect_column_role(
                column,
                series,
            ),
            "missing": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100),
                2,
            ),
            "unique": int(
                series.nunique(dropna=True)
            ),
        }

        schema["columns"].append(
            column_schema
        )

    return schema
