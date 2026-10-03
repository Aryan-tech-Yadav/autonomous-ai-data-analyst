import pandas as pd


def pandas_analysis(df: pd.DataFrame) -> dict:
    """
    Perform basic business analysis on a dataset.

    Returns:
        Dictionary containing analysis results.
    """

    results = {
        "dataset_summary": {},
        "numeric_statistics": {},
        "categorical_analysis": {},
        "business_metrics": {},
        "warnings": [],
    }

    # ==========================================
    # DATASET SUMMARY
    # ==========================================

    results["dataset_summary"] = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # ==========================================
    # NUMERIC ANALYSIS
    # ==========================================

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            results["warnings"].append(
                f"{column}: No numeric data available."
            )
            continue

        results["numeric_statistics"][column] = {
            "count": int(series.count()),
            "sum": float(series.sum()),
            "mean": float(series.mean()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "median": float(series.median()),
        }

    # ==========================================
    # CATEGORICAL ANALYSIS
    # ==========================================

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:

        # Avoid generating huge results for IDs
        if df[column].nunique(dropna=True) > 30:
            continue

        counts = df[column].value_counts(
            dropna=True
        )

        results["categorical_analysis"][column] = {
            str(key): int(value)
            for key, value in counts.items()
        }

    # ==========================================
    # BUSINESS METRICS
    # ==========================================

    if "Units Sold" in df.columns:

        results["business_metrics"][
            "total_units_sold"
        ] = float(
            df["Units Sold"].sum()
        )

    if "Unit Price" in df.columns:

        results["business_metrics"][
            "average_unit_price"
        ] = float(
            df["Unit Price"].mean()
        )

    # ==========================================
    # REVENUE CALCULATION
    # ==========================================

    required_columns = [
        "Units Sold",
        "Unit Price",
    ]

    if all(
        column in df.columns
        for column in required_columns
    ):

        # Calculate revenue without modifying
        # the original DataFrame.

        calculated_revenue = (
            df["Units Sold"]
            * df["Unit Price"]
        )

        results["business_metrics"][
            "calculated_total_revenue"
        ] = float(
            calculated_revenue.sum()
        )

        results["business_metrics"][
            "revenue_calculated_rows"
        ] = int(
            calculated_revenue.notna().sum()
        )

        # ======================================
        # REGION ANALYSIS
        # ======================================

        if "Region" in df.columns:

            regional_data = pd.DataFrame({
                "Region": df["Region"],
                "Revenue": calculated_revenue,
            })

            regional_revenue = (
                regional_data
                .groupby("Region")["Revenue"]
                .sum(min_count=1)
                .dropna()
                .sort_values(ascending=False)
            )

            results["business_metrics"][
                "regional_revenue"
            ] = {
                str(region): float(revenue)
                for region, revenue
                in regional_revenue.items()
            }

    # ==========================================
    # DATA QUALITY WARNINGS
    # ==========================================

    for column in df.columns:

        missing_percentage = (
            df[column].isna().mean() * 100
        )

        if missing_percentage == 100:

            results["warnings"].append(
                f"{column}: Column is completely empty."
            )

        elif missing_percentage > 50:

            results["warnings"].append(
                f"{column}: "
                f"{missing_percentage:.2f}% missing values."
            )

    return results
