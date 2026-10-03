from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.tools.time_tool import prepare_time_series


# ============================================================
# BAR CHART
# ============================================================

def generate_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    output_path: str,
    title: str = "",
    ascending: bool = False,
) -> Dict[str, Any]:

    if category_column not in df.columns:
        raise ValueError(
            f"Category column '{category_column}' "
            f"does not exist."
        )

    if value_column not in df.columns:

        # Allow derived revenue for bar charts too.
        normalized = (
            str(value_column)
            .strip()
            .lower()
            .replace("_", " ")
        )

        revenue_aliases = {
            "revenue",
            "total revenue",
            "sales",
            "sales revenue",
        }

        if normalized in revenue_aliases:

            working_df = df.copy()

            if (
                "Units Sold" in working_df.columns
                and "Unit Price" in working_df.columns
            ):

                working_df[
                    "_analysis_revenue"
                ] = (
                    pd.to_numeric(
                        working_df["Units Sold"],
                        errors="coerce",
                    )
                    * pd.to_numeric(
                        working_df["Unit Price"],
                        errors="coerce",
                    )
                )

                value_column = "_analysis_revenue"

            else:

                raise ValueError(
                    f"Value column '{value_column}' "
                    f"does not exist and derived "
                    f"revenue cannot be calculated."
                )

        else:

            raise ValueError(
                f"Value column '{value_column}' "
                f"does not exist."
            )

    else:
        working_df = df.copy()

        # If an existing revenue column is completely
        # unusable, calculate derived revenue.
        normalized_value = (
            str(value_column)
            .strip()
            .lower()
            .replace("_", " ")
        )

        revenue_aliases = {
            "revenue",
            "total revenue",
            "sales",
            "sales revenue",
        }

        numeric_test = pd.to_numeric(
            working_df[value_column],
            errors="coerce",
        )

        if (
            normalized_value in revenue_aliases
            and numeric_test.notna().sum() == 0
            and "Units Sold" in working_df.columns
            and "Unit Price" in working_df.columns
        ):

            working_df[
                "_analysis_revenue"
            ] = (
                pd.to_numeric(
                    working_df["Units Sold"],
                    errors="coerce",
                )
                * pd.to_numeric(
                    working_df["Unit Price"],
                    errors="coerce",
                )
            )

            value_column = "_analysis_revenue"

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    working_df = working_df[
        [
            category_column,
            value_column,
        ]
    ].copy()

    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[
            category_column,
            value_column,
        ]
    )

    if working_df.empty:
        raise ValueError(
            "No valid data is available "
            "for the bar chart."
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        working_df
        .groupby(
            category_column,
            dropna=False,
        )[value_column]
        .sum()
        .sort_values(
            ascending=ascending
        )
    )

    # --------------------------------------------------------
    # Save chart
    # --------------------------------------------------------

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    grouped.plot(
        kind="bar"
    )

    plt.xlabel(
        category_column
    )

    display_value_column = (
        "Revenue"
        if value_column
        == "_analysis_revenue"
        else value_column
    )

    plt.ylabel(
        display_value_column
    )

    plt.title(
        title
        or f"{display_value_column} "
           f"by {category_column}"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "operation": "generate_bar_chart",
        "chart_type": "bar",
        "category_column": category_column,
        "value_column": display_value_column,
        "output_path": str(output),
        "categories": [
            str(value)
            for value in grouped.index
        ],
        "values": [
            float(value)
            for value in grouped.values
        ],
    }


# ============================================================
# LINE CHART
# ============================================================

def generate_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: str,
    title: str = "",
    frequency: str = "monthly",
    aggregation: str = "sum",
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Prepare an aggregated time series.
    #
    # This is intentionally delegated to the time tool so
    # chart generation and time-series analysis use the same
    # business logic.
    # --------------------------------------------------------

    time_series = prepare_time_series(
        df=df,
        x_column=x_column,
        y_column=y_column,
        frequency=frequency,
        aggregation=aggregation,
    )

    points = time_series.get(
        "points",
        [],
    )

    if not points:

        raise ValueError(
            "No time-series points were "
            "generated for the line chart."
        )

    # --------------------------------------------------------
    # Convert points to DataFrame
    # --------------------------------------------------------

    chart_df = pd.DataFrame(
        points
    )

    chart_df["date"] = pd.to_datetime(
        chart_df["date"],
        errors="coerce",
    )

    chart_df["value"] = pd.to_numeric(
        chart_df["value"],
        errors="coerce",
    )

    chart_df = chart_df.dropna(
        subset=[
            "date",
            "value",
        ]
    )

    chart_df = chart_df.sort_values(
        "date"
    )

    if chart_df.empty:

        raise ValueError(
            "No valid points are available "
            "for the line chart."
        )

    # --------------------------------------------------------
    # Save chart
    # --------------------------------------------------------

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        chart_df["date"],
        chart_df["value"],
        marker="o",
    )

    display_value_column = (
        time_series.get(
            "value_column",
            y_column,
        )
    )

    normalized_frequency = (
        time_series.get(
            "frequency",
            frequency,
        )
    )

    plt.xlabel(
        x_column
    )

    plt.ylabel(
        display_value_column
    )

    plt.title(
        title
        or f"{display_value_column} "
           f"over time"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "operation": "generate_line_chart",
        "chart_type": "line",
        "x_column": x_column,
        "y_column": display_value_column,
        "frequency": normalized_frequency,
        "aggregation": aggregation,
        "output_path": str(output),
        "points": int(
            len(chart_df)
        ),
        "data": [
            {
                "date": row[
                    "date"
                ].strftime(
                    "%Y-%m-%d"
                ),
                "value": float(
                    row[
                        "value"
                    ]
                ),
            }
            for _, row
            in chart_df.iterrows()
        ],
    }
