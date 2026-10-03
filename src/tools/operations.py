from typing import Any, Dict

import pandas as pd


# ==========================================================
# Revenue Calculation
# ==========================================================

def calculate_revenue(
    df: pd.DataFrame,
    units_column: str,
    price_column: str,
    output_column: str = "Revenue",
) -> Dict[str, Any]:
    """
    Calculate row-level revenue.

    Revenue = Units Sold × Unit Price
    """

    if units_column not in df.columns:
        raise ValueError(
            f"Units column '{units_column}' does not exist."
        )

    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' does not exist."
        )

    units = pd.to_numeric(
        df[units_column],
        errors="coerce",
    )

    prices = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    revenue = units * prices

    return {
        "operation": "calculate_revenue",
        "output_column": output_column,
        "rows_calculated": int(
            revenue.notna().sum()
        ),
        "total_revenue": float(
            revenue.sum()
        ),
        "revenue": revenue,
    }


# ==========================================================
# Group-by Aggregation
# ==========================================================

def groupby_aggregate(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: str = "sum",
) -> Dict[str, Any]:
    """
    Group a DataFrame column and perform
    a controlled aggregation.

    Missing group values are excluded because
    NULL/NaN is not considered a valid business group.
    """

    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Value column '{value_column}' does not exist."
        )

    allowed_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "median",
    }

    aggregation = aggregation.lower().strip()

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'."
        )

    working_df = df[
        [
            group_column,
            value_column,
        ]
    ].copy()

    # Ignore rows where the grouping value is missing.
    working_df = working_df.dropna(
        subset=[group_column]
    )

    grouped = (
        working_df
        .groupby(
            group_column,
            dropna=False,
        )[value_column]
        .agg(aggregation)
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():

        group_value = row[
            group_column
        ]

        numeric_value = row[
            value_column
        ]

        results.append(
            {
                str(group_column): (
                    None
                    if pd.isna(group_value)
                    else str(group_value)
                ),
                str(value_column): (
                    None
                    if pd.isna(numeric_value)
                    else float(numeric_value)
                ),
            }
        )

    return {
        "operation": "groupby_aggregate",
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "results": results,
    }


# ==========================================================
# Find Maximum Group
# ==========================================================

def find_max(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> Dict[str, Any]:
    """
    Find the group with the highest aggregated value.

    Missing group values are ignored.
    """

    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Value column '{value_column}' does not exist."
        )

    working_df = df[
        [
            group_column,
            value_column,
        ]
    ].copy()

    # Convert values to numeric.
    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    # Remove invalid numeric values.
    working_df = working_df.dropna(
        subset=[value_column]
    )

    # Remove missing group values.
    working_df = working_df.dropna(
        subset=[group_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid grouped numeric values "
            "available for max calculation."
        )

    grouped = (
        working_df
        .groupby(
            group_column,
            dropna=False,
        )[value_column]
        .sum()
        .reset_index()
    )

    if grouped.empty:
        raise ValueError(
            "No grouped values available "
            "for max calculation."
        )

    max_index = grouped[
        value_column
    ].idxmax()

    row = grouped.loc[
        max_index
    ]

    group_value = row[
        group_column
    ]

    max_value = row[
        value_column
    ]

    return {
        "operation": "find_max",
        "group_column": group_column,
        "value_column": value_column,
        "group": (
            None
            if pd.isna(group_value)
            else str(group_value)
        ),
        "value": float(
            max_value
        ),
    }


# ==========================================================
# Numeric Statistics
# ==========================================================

def calculate_statistics(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:
    """
    Calculate basic statistics for a numeric column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    series = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if series.empty:
        raise ValueError(
            f"No numeric values available "
            f"in column '{column}'."
        )

    return {
        "operation": "calculate_statistics",
        "column": column,
        "count": int(
            series.count()
        ),
        "sum": float(
            series.sum()
        ),
        "mean": float(
            series.mean()
        ),
        "median": float(
            series.median()
        ),
        "minimum": float(
            series.min()
        ),
        "maximum": float(
            series.max()
        ),
    }


# ==========================================================
# Categorical Analysis
# ==========================================================

def categorical_analysis(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:
    """
    Count categorical values.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    counts = df[column].value_counts(
        dropna=False
    )

    results = {}

    for key, value in counts.items():

        if pd.isna(key):
            key = "NULL"
        else:
            key = str(key)

        results[key] = int(
            value
        )

    return {
        "operation": "categorical_analysis",
        "column": column,
        "counts": results,
    }


# ==========================================================
# Operation Registry
# ==========================================================

OPERATION_REGISTRY = {
    "revenue_calculation": calculate_revenue,
    "revenue_calculations": calculate_revenue,

    "groupby_aggregation": groupby_aggregate,
    "group_by": groupby_aggregate,

    "find_max": find_max,

    "statistics": calculate_statistics,
    "calculate_statistics": calculate_statistics,

    "categorical_analysis": categorical_analysis,
}


# ==========================================================
# Operation Lookup
# ==========================================================

def get_operation(
    operation_name: str,
):
    """
    Return a registered operation.
    """

    operation = OPERATION_REGISTRY.get(
        operation_name
    )

    if operation is None:

        available = ", ".join(
            OPERATION_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown operation '{operation_name}'. "
            f"Available operations: {available}"
        )

    return operation
