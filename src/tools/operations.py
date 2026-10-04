from typing import Any, Dict

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Revenue Calculation
# ==========================================================

def calculate_revenue(
    df: pd.DataFrame,
    units_column: str,
    price_column: str,
    output_column: str = "Revenue",
) -> Dict[str, Any]:

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

    price = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    revenue = units * price

    valid_revenue = revenue.dropna()

    return {
        "operation": "calculate_revenue",
        "output_column": output_column,
        "rows_calculated": int(valid_revenue.count()),
        "total_revenue": (
            None
            if valid_revenue.empty
            else float(valid_revenue.sum())
        ),
        "revenue": revenue,
    }


# ==========================================================
# Shift Column
# ==========================================================

def create_shifted_column(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    periods: int = 1,
    sort_column: str | None = None,
    ascending: bool = True,
) -> Dict[str, Any]:

    if source_column not in df.columns:
        raise ValueError(
            f"Source column '{source_column}' does not exist."
        )

    if not isinstance(periods, int):
        raise ValueError("periods must be an integer.")

    working_df = df.copy()

    if sort_column is not None:

        if sort_column not in working_df.columns:
            raise ValueError(
                f"Sort column '{sort_column}' does not exist."
            )

        working_df = working_df.sort_values(
            by=sort_column,
            ascending=ascending,
            kind="stable",
        )

    shifted = working_df[source_column].shift(periods)

    return {
        "operation": "create_shifted_column",
        "source_column": source_column,
        "output_column": output_column,
        "periods": periods,
        "sort_column": sort_column,
        "ascending": bool(ascending),
        "shifted_values": shifted,
        "sorted_index": shifted.index,
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

    working_df = df[[group_column, value_column]].copy()

    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[group_column]
    )

    grouped = (
        working_df
        .groupby(group_column, dropna=False)[value_column]
        .agg(aggregation)
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():

        group_value = row[group_column]
        numeric_value = row[value_column]

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

    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Value column '{value_column}' does not exist."
        )

    working_df = df[[group_column, value_column]].copy()

    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[value_column, group_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid grouped numeric values available for max calculation."
        )

    grouped = (
        working_df
        .groupby(group_column, dropna=False)[value_column]
        .sum()
        .reset_index()
    )

    if grouped.empty:
        raise ValueError(
            "No grouped values available for max calculation."
        )

    max_index = grouped[value_column].idxmax()
    row = grouped.loc[max_index]

    group_value = row[group_column]
    max_value = row[value_column]

    return {
        "operation": "find_max",
        "group_column": group_column,
        "value_column": value_column,
        "group": (
            None
            if pd.isna(group_value)
            else str(group_value)
        ),
        "value": float(max_value),
    }


# ==========================================================
# Numeric Statistics
# ==========================================================

def calculate_statistics(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:

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
            f"No numeric values available in column '{column}'."
        )

    return {
        "operation": "calculate_statistics",
        "column": column,
        "count": int(series.count()),
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
    }


# ==========================================================
# Categorical Analysis
# ==========================================================

def categorical_analysis(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    counts = df[column].value_counts(dropna=False)

    results = {}

    for key, value in counts.items():

        if pd.isna(key):
            key = "NULL"
        else:
            key = str(key)

        results[key] = int(value)

    return {
        "operation": "categorical_analysis",
        "column": column,
        "counts": results,
    }


# ==========================================================
# Ranking
# ==========================================================

def rank_by_value(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: str = "sum",
    ascending: bool = False,
    top_n: int | None = None,
) -> Dict[str, Any]:

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

    aggregation = str(aggregation).strip().lower()

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'."
        )

    if top_n is not None:

        if isinstance(top_n, bool) or not isinstance(top_n, int):
            raise ValueError("top_n must be an integer.")

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

    working_df = df[[group_column, value_column]].copy()

    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[group_column, value_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid grouped numeric values available for ranking."
        )

    ranked = (
        working_df
        .groupby(group_column, dropna=False)[value_column]
        .agg(aggregation)
        .reset_index()
    )

    ranked = ranked.sort_values(
        by=value_column,
        ascending=ascending,
        kind="stable",
    )

    ranked["rank"] = range(1, len(ranked) + 1)

    if top_n is not None:
        ranked = ranked.head(top_n)

    results = []

    for _, row in ranked.iterrows():

        group_value = row[group_column]
        value = row[value_column]

        results.append(
            {
                "rank": int(row["rank"]),
                "group": (
                    None
                    if pd.isna(group_value)
                    else str(group_value)
                ),
                "value": (
                    None
                    if pd.isna(value)
                    else float(value)
                ),
            }
        )

    return {
        "operation": "rank_by_value",
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "ascending": bool(ascending),
        "top_n": top_n,
        "results": results,
    }


# ==========================================================
# Percentage Change
# ==========================================================

def calculate_percentage_change(
    df: pd.DataFrame,
    current_column: str,
    previous_column: str,
    output_column: str = "Percentage Change",
) -> Dict[str, Any]:

    if current_column not in df.columns:
        raise ValueError(
            f"Current column '{current_column}' does not exist."
        )

    if previous_column not in df.columns:
        raise ValueError(
            f"Previous column '{previous_column}' does not exist."
        )

    current = pd.to_numeric(
        df[current_column],
        errors="coerce",
    )

    previous = pd.to_numeric(
        df[previous_column],
        errors="coerce",
    )

    percentage_change = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )

    valid = (
        current.notna()
        & previous.notna()
        & previous.ne(0)
    )

    percentage_change.loc[valid] = (
        (
            current.loc[valid]
            - previous.loc[valid]
        )
        / previous.loc[valid]
        * 100
    )

    valid_values = percentage_change.dropna()

    return {
        "operation": "calculate_percentage_change",
        "current_column": current_column,
        "previous_column": previous_column,
        "output_column": output_column,
        "rows_calculated": int(valid_values.count()),
        "mean_percentage_change": (
            None
            if valid_values.empty
            else float(valid_values.mean())
        ),
        "minimum_percentage_change": (
            None
            if valid_values.empty
            else float(valid_values.min())
        ),
        "maximum_percentage_change": (
            None
            if valid_values.empty
            else float(valid_values.max())
        ),
        "percentage_change": percentage_change,
    }


# ==========================================================
# Numeric Comparison
# ==========================================================

def compare_columns(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> Dict[str, Any]:

    if left_column not in df.columns:
        raise ValueError(
            f"Left column '{left_column}' does not exist."
        )

    if right_column not in df.columns:
        raise ValueError(
            f"Right column '{right_column}' does not exist."
        )

    left = pd.to_numeric(
        df[left_column],
        errors="coerce",
    )

    right = pd.to_numeric(
        df[right_column],
        errors="coerce",
    )

    valid = left.notna() & right.notna()

    left_greater = valid & left.gt(right)
    right_greater = valid & right.gt(left)
    equal = valid & left.eq(right)

    return {
        "operation": "compare_columns",
        "left_column": left_column,
        "right_column": right_column,
        "valid_comparisons": int(valid.sum()),
        "left_greater_count": int(left_greater.sum()),
        "right_greater_count": int(right_greater.sum()),
        "equal_count": int(equal.sum()),
    }


# ==========================================================
# Chart Helpers
# ==========================================================

def _validate_chart_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Chart column '{column}' does not exist."
            )


def _save_chart(
    fig,
    output_path: str | None = None,
) -> str | None:

    if output_path:

        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150,
        )

        return output_path

    return None


# ==========================================================
# Bar Chart
# ==========================================================

def generate_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str | None = None,
    aggregation: str = "sum",
    top_n: int | None = None,
    ascending: bool = False,
    output_path: str | None = None,
) -> Dict[str, Any]:

    _validate_chart_columns(
        df,
        [x_column, y_column],
    )

    allowed_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "median",
    }

    aggregation = str(
        aggregation
    ).strip().lower()

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported chart aggregation '{aggregation}'."
        )

    working_df = df[[x_column, y_column]].copy()

    working_df[y_column] = pd.to_numeric(
        working_df[y_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[x_column, y_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid data available for bar chart."
        )

    chart_data = (
        working_df
        .groupby(x_column, dropna=False)[y_column]
        .agg(aggregation)
        .reset_index()
    )

    chart_data = chart_data.sort_values(
        by=y_column,
        ascending=ascending,
        kind="stable",
    )

    if top_n is not None:

        if isinstance(top_n, bool) or not isinstance(top_n, int):
            raise ValueError(
                "top_n must be an integer."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        chart_data = chart_data.head(top_n)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        chart_data[x_column].astype(str),
        chart_data[y_column],
    )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    ax.set_title(
        title
        or f"{y_column} by {x_column}"
    )

    fig.tight_layout()

    saved_path = _save_chart(
        fig,
        output_path,
    )

    chart_data_records = []

    for _, row in chart_data.iterrows():

        chart_data_records.append(
            {
                "x": str(row[x_column]),
                "y": float(row[y_column]),
            }
        )

    plt.close(fig)

    return {
        "operation": "generate_bar_chart",
        "chart_type": "bar",
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": aggregation,
        "top_n": top_n,
        "ascending": bool(ascending),
        "title": title or f"{y_column} by {x_column}",
        "output_path": saved_path,
        "data": chart_data_records,
    }


# ==========================================================
# Line Chart
# ==========================================================

def generate_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str | None = None,
    aggregation: str = "sum",
    sort_x: bool = True,
    output_path: str | None = None,
) -> Dict[str, Any]:

    _validate_chart_columns(
        df,
        [x_column, y_column],
    )

    allowed_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "median",
    }

    aggregation = str(
        aggregation
    ).strip().lower()

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported chart aggregation '{aggregation}'."
        )

    working_df = df[[x_column, y_column]].copy()

    working_df[y_column] = pd.to_numeric(
        working_df[y_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[x_column, y_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid data available for line chart."
        )

    chart_data = (
        working_df
        .groupby(x_column, dropna=False)[y_column]
        .agg(aggregation)
        .reset_index()
    )

    if sort_x:
        chart_data = chart_data.sort_values(
            by=x_column,
            kind="stable",
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        chart_data[x_column].astype(str),
        chart_data[y_column],
        marker="o",
    )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    ax.set_title(
        title
        or f"{y_column} by {x_column}"
    )

    fig.tight_layout()

    saved_path = _save_chart(
        fig,
        output_path,
    )

    chart_data_records = []

    for _, row in chart_data.iterrows():

        chart_data_records.append(
            {
                "x": str(row[x_column]),
                "y": float(row[y_column]),
            }
        )

    plt.close(fig)

    return {
        "operation": "generate_line_chart",
        "chart_type": "line",
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": aggregation,
        "sort_x": bool(sort_x),
        "title": title or f"{y_column} by {x_column}",
        "output_path": saved_path,
        "data": chart_data_records,
    }


# ==========================================================
# Operation Registry
# ==========================================================

OPERATION_REGISTRY = {
    "revenue_calculation": calculate_revenue,
    "revenue_calculations": calculate_revenue,

    "create_shifted_column": create_shifted_column,
    "shift_column": create_shifted_column,

    "groupby_aggregation": groupby_aggregate,
    "group_by": groupby_aggregate,

    "find_max": find_max,

    "statistics": calculate_statistics,
    "calculate_statistics": calculate_statistics,

    "categorical_analysis": categorical_analysis,

    "rank_by_value": rank_by_value,
    "ranking": rank_by_value,

    "calculate_percentage_change":
        calculate_percentage_change,

    "percentage_change":
        calculate_percentage_change,

    "compare_columns": compare_columns,
    "comparison": compare_columns,

    "generate_bar_chart":
        generate_bar_chart,

    "generate_line_chart":
        generate_line_chart,
}


# ==========================================================
# Operation Lookup
# ==========================================================

def get_operation(operation_name: str):

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
