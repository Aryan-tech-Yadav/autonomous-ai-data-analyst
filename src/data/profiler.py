import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze the basic structure and quality of a DataFrame.
    """

    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_details": [],
    }

    for column in df.columns:
        column_info = {
            "name": column,
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        }

        profile["column_details"].append(column_info)

    return profile
