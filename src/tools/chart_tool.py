from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.tools.time_tool import prepare_time_series


# ============================================================
# SHARED HELPERS
# ============================================================

REVENUE_ALIASES = {
    "revenue",
    "total revenue",
    "sales",
    "sales revenue",
}


def _normalize_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )


def _is_revenue_alias(value: Any) -> bool:
    return _normalize_name(value) in REVENUE_ALIASES


def _compact_number(value: float) -> str:
    """Human-readable number for chart annotations."""

    value = float(value)

    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"

    if value == int(value):
        return f"{int(value):,}"

    return f"{value:,.1f}"


def _format_axis_value(value: float, _position: int) -> str:
    """Compact formatter for large numeric axes."""

    return _compact_number(value)


def _prepare_revenue_if_needed(
    df: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, str]:

    working_df = df.copy()

    if value_column in working_df.columns:

        numeric_test = pd.to_numeric(
            working_df[value_column],
            errors="coerce",
        )

        if (
            not _is_revenue_alias(value_column)
            or numeric_test.notna().sum() > 0
        ):
            return working_df, value_column

    if _is_revenue_alias(value_column):

        if (
            "Units Sold" not in working_df.columns
            or "Unit Price" not in working_df.columns
        ):
            raise ValueError(
                f"Value column '{value_column}' "
                "does not exist and derived revenue "
                "cannot be calculated."
            )

        working_df["_analysis_revenue"] = (
            pd.to_numeric(
                working_df["Units Sold"],
                errors="coerce",
            )
            * pd.to_numeric(
                working_df["Unit Price"],
                errors="coerce",
            )
        )

        return working_df, "_analysis_revenue"

    raise ValueError(
        f"Value column '{value_column}' "
        "does not exist."
    )


def _display_value_name(
    actual_column: str,
    requested_column: str,
) -> str:

    if actual_column == "_analysis_revenue":
        return "Revenue"

    return str(requested_column)


def _save_figure(
    fig,
    output_path: str,
) -> str:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return str(output)


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

    # --------------------------------------------------------
    # Validate category
    # --------------------------------------------------------

    if category_column not in df.columns:

        raise ValueError(
            f"Category column '{category_column}' "
            f"does not exist."
        )

    # --------------------------------------------------------
    # Resolve numeric / derived value
    # --------------------------------------------------------

    working_df, actual_value_column = (
        _prepare_revenue_if_needed(
            df,
            value_column,
        )
    )

    display_value_column = _display_value_name(
        actual_value_column,
        value_column,
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    working_df = working_df[
        [
            category_column,
            actual_value_column,
        ]
    ].copy()

    working_df[actual_value_column] = pd.to_numeric(
        working_df[actual_value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[
            category_column,
            actual_value_column,
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
        )[actual_value_column]
        .sum()
        .sort_values(
            ascending=ascending,
            kind="stable",
        )
    )

    if grouped.empty:

        raise ValueError(
            "No aggregated values are available "
            "for the bar chart."
        )

    # --------------------------------------------------------
    # Determine layout
    # --------------------------------------------------------

    horizontal = len(grouped) >= 7

    fig_height = max(
        6,
        min(12, 3.5 + len(grouped) * 0.42),
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            fig_height if horizontal else 6.5,
        )
    )

    # --------------------------------------------------------
    # Draw chart
    # --------------------------------------------------------

    if horizontal:

        bars = ax.barh(
            grouped.index.astype(str),
            grouped.values,
        )

        ax.set_xlabel(
            display_value_column
        )

        ax.set_ylabel(
            category_column
        )

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                _format_axis_value
            )
        )

        for bar, value in zip(
            bars,
            grouped.values,
        ):

            ax.text(
                bar.get_width(),
                bar.get_y()
                + bar.get_height() / 2,
                f" {_compact_number(value)}",
                va="center",
                ha="left",
                fontsize=9,
            )

    else:

        bars = ax.bar(
            grouped.index.astype(str),
            grouped.values,
        )

        ax.set_xlabel(
            category_column
        )

        ax.set_ylabel(
            display_value_column
        )

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                _format_axis_value
            )
        )

        ax.tick_params(
            axis="x",
            rotation=35,
        )

        for label in ax.get_xticklabels():

            label.set_horizontalalignment(
                "right"
            )

        for bar, value in zip(
            bars,
            grouped.values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                _compact_number(value),
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    # --------------------------------------------------------
    # Titles / grid / layout
    # --------------------------------------------------------

    chart_title = (
        title.strip()
        if title
        else (
            f"{display_value_column} "
            f"by {category_column}"
        )
    )

    ax.set_title(
        chart_title,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    ax.grid(
        axis="x" if horizontal else "y",
        alpha=0.25,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    saved_path = _save_figure(
        fig,
        output_path,
    )

    # --------------------------------------------------------
    # Structured result
    # --------------------------------------------------------

    return {
        "operation": "generate_bar_chart",
        "chart_type": "bar",
        "category_column": category_column,
        "value_column": display_value_column,
        "output_path": saved_path,
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
    # Prepare time series
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
    # Convert points
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
    # Metadata
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    ax.plot(
        chart_df["date"],
        chart_df["value"],
        marker="o",
        linewidth=2.5,
        markersize=6,
    )

    # --------------------------------------------------------
    # Point annotations
    # --------------------------------------------------------

    for _, row in chart_df.iterrows():

        ax.annotate(
            _compact_number(
                row["value"]
            ),
            (
                row["date"],
                row["value"],
            ),
            xytext=(
                0,
                9,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Period",
        fontsize=11,
    )

    ax.set_ylabel(
        display_value_column,
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            _format_axis_value
        )
    )

    # --------------------------------------------------------
    # Time labels
    # --------------------------------------------------------

    if normalized_frequency == "monthly":

        ax.set_xticks(
            chart_df["date"]
        )

        ax.set_xticklabels(
            [
                value.strftime(
                    "%b %Y"
                )
                for value
                in chart_df["date"]
            ],
            rotation=35,
            ha="right",
        )

    else:

        ax.tick_params(
            axis="x",
            rotation=35,
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    chart_title = (
        title.strip()
        if title
        else (
            f"{display_value_column} "
            f"Trend"
        )
    )

    ax.set_title(
        chart_title,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.tight_layout()

    saved_path = _save_figure(
        fig,
        output_path,
    )

    # --------------------------------------------------------
    # Structured result
    # --------------------------------------------------------

    return {
        "operation": "generate_line_chart",
        "chart_type": "line",
        "x_column": x_column,
        "y_column": display_value_column,
        "frequency": normalized_frequency,
        "aggregation": aggregation,
        "output_path": saved_path,
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
