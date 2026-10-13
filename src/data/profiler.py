import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze dataset structure, missing values, and data quality.
    Preserve the existing profile fields for backward compatibility.
    """
    rows, columns = df.shape
    total_cells = rows * columns
    missing_values = int(df.isna().sum().sum())

    profile = {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "missing_percentage": (
            round(missing_values / total_cells * 100, 2)
            if total_cells else 0.0
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_details": [],
        "columns_with_missing_values": [],
        "data_quality": {
            "status": "good",
            "issues": [],
        },
    }

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        missing_percentage = (
            round(missing / rows * 100, 2) if rows else 0.0
        )
        unique = int(series.nunique(dropna=True))

        if missing == 0:
            status = "complete"
        elif missing_percentage < 5:
            status = "low_missingness"
        elif missing_percentage < 20:
            status = "moderate_missingness"
        else:
            status = "high_missingness"

        column_info = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_percentage": missing_percentage,
            "unique": unique,
            "status": status,
        }
        profile["column_details"].append(column_info)

        if missing > 0:
            profile["columns_with_missing_values"].append({
                "name": str(column),
                "missing": missing,
                "missing_percentage": missing_percentage,
                "priority": (
                    "high" if missing_percentage >= 20
                    else "medium" if missing_percentage >= 5
                    else "low"
                ),
            })

    issues = profile["data_quality"]["issues"]

    if rows == 0:
        issues.append("The dataset contains no rows.")

    if columns == 0:
        issues.append("The dataset contains no columns.")

    if missing_values:
        issues.append(
            f"{missing_values} missing cell(s) found "
            f"({profile['missing_percentage']}% of all cells)."
        )

    if profile["duplicate_rows"]:
        issues.append(
            f"{profile['duplicate_rows']} duplicate row(s) found."
        )

    if any(
        item["priority"] == "high"
        for item in profile["columns_with_missing_values"]
    ):
        issues.append(
            "One or more columns have at least 20% missing values."
        )

    profile["data_quality"]["status"] = (
        "needs_attention" if issues else "good"
    )

    return profile
