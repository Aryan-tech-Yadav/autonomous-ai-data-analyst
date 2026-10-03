from __future__ import annotations

from typing import Any, Optional

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

REVENUE_ALIASES = {
    "revenue",
    "sales",
    "total revenue",
    "total_revenue",
    "revenue total",
    "sales revenue",
}


# ============================================================
# COLUMN HELPERS
# ============================================================

def _normalize_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _find_column_by_alias(
    df: pd.DataFrame,
    aliases: set[str],
) -> Optional[str]:
    normalized_aliases = {
        _normalize_name(alias)
        for alias in aliases
    }

    for column in df.columns:
        if _normalize_name(column) in normalized_aliases:
            return str(column)

    return None


def _find_date_column(
    df: pd.DataFrame,
    requested_column: Optional[str] = None,
) -> Optional[str]:

    # --------------------------------------------------------
    # Explicit requested column
    # --------------------------------------------------------

    if requested_column:

        if requested_column in df.columns:
            return str(requested_column)

        normalized_requested = _normalize_name(
            requested_column
        )

        for column in df.columns:
            if (
                _normalize_name(column)
                == normalized_requested
            ):
                return str(column)

    # --------------------------------------------------------
    # Common date/time names
    # --------------------------------------------------------

    date_aliases = {
        "date",
        "datetime",
        "date time",
        "timestamp",
        "time",
        "order date",
        "transaction date",
        "created date",
    }

    for column in df.columns:
        normalized = _normalize_name(column)

        if normalized in date_aliases:
            return str(column)

    # --------------------------------------------------------
    # Name-based fallback
    # --------------------------------------------------------

    for column in df.columns:
        normalized = _normalize_name(column)

        if (
            "date" in normalized
            or "time" in normalized
            or "timestamp" in normalized
        ):
            return str(column)

    # --------------------------------------------------------
    # Data-based fallback
    # --------------------------------------------------------

    for column in df.columns:

        converted = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        valid_ratio = float(
            converted.notna().mean()
        )

        if valid_ratio >= 0.70:
            return str(column)

    return None


# ============================================================
# REVENUE CALCULATION
# ============================================================

def _calculate_revenue(
    df: pd.DataFrame,
) -> Optional[pd.Series]:

    columns = {
        _normalize_name(column): column
        for column in df.columns
    }

    units_column = None
    price_column = None

    # --------------------------------------------------------
    # Units Sold
    # --------------------------------------------------------

    unit_aliases = {
        "units sold",
        "unit sold",
        "units",
        "quantity",
        "qty",
        "quantity sold",
    }

    for alias in unit_aliases:

        if alias in columns:
            units_column = columns[alias]
            break

    # --------------------------------------------------------
    # Unit Price
    # --------------------------------------------------------

    price_aliases = {
        "unit price",
        "price per unit",
        "selling price",
        "price",
    }

    for alias in price_aliases:

        if alias in columns:
            price_column = columns[alias]
            break

    if units_column is None or price_column is None:
        return None

    units = pd.to_numeric(
        df[units_column],
        errors="coerce",
    )

    price = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    revenue = units * price

    # Require at least one usable calculated value.
    if revenue.notna().sum() == 0:
        return None

    return revenue


def _is_usable_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> bool:

    if column not in df.columns:
        return False

    numeric = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    return bool(
        numeric.notna().sum() > 0
    )


# ============================================================
# VALUE COLUMN RESOLUTION
# ============================================================

def _resolve_value_column(
    df: pd.DataFrame,
    requested_column: Optional[str],
) -> tuple[pd.DataFrame, str]:

    working_df = df.copy()

    # --------------------------------------------------------
    # Explicit requested column
    # --------------------------------------------------------

    if requested_column:

        actual_column = None

        if requested_column in working_df.columns:
            actual_column = requested_column

        else:

            normalized_requested = (
                _normalize_name(
                    requested_column
                )
            )

            for column in working_df.columns:

                if (
                    _normalize_name(column)
                    == normalized_requested
                ):
                    actual_column = column
                    break

        if actual_column is not None:

            # ------------------------------------------------
            # IMPORTANT:
            # If requested column exists but contains no
            # usable numeric data, try derived revenue before
            # returning the empty column.
            # ------------------------------------------------

            if _is_usable_numeric_column(
                working_df,
                actual_column,
            ):

                return (
                    working_df,
                    actual_column,
                )

            normalized_requested = (
                _normalize_name(
                    requested_column
                )
            )

            if (
                normalized_requested
                in REVENUE_ALIASES
            ):

                revenue = _calculate_revenue(
                    working_df
                )

                if revenue is not None:

                    working_df[
                        "_analysis_revenue"
                    ] = revenue

                    return (
                        working_df,
                        "_analysis_revenue",
                    )

            # Continue to automatic detection
            # instead of returning an empty column.

    # --------------------------------------------------------
    # Explicit revenue aliases
    # --------------------------------------------------------

    if requested_column:

        normalized_requested = (
            _normalize_name(
                requested_column
            )
        )

        if (
            normalized_requested
            in REVENUE_ALIASES
        ):

            revenue = _calculate_revenue(
                working_df
            )

            if revenue is not None:

                working_df[
                    "_analysis_revenue"
                ] = revenue

                return (
                    working_df,
                    "_analysis_revenue",
                )

    # --------------------------------------------------------
    # Automatic revenue detection
    # --------------------------------------------------------

    revenue_column = _find_column_by_alias(
        working_df,
        REVENUE_ALIASES,
    )

    if revenue_column is not None:

        if _is_usable_numeric_column(
            working_df,
            revenue_column,
        ):

            return (
                working_df,
                revenue_column,
            )

    # --------------------------------------------------------
    # Automatic derived revenue
    # --------------------------------------------------------

    revenue = _calculate_revenue(
        working_df
    )

    if revenue is not None:

        working_df[
            "_analysis_revenue"
        ] = revenue

        return (
            working_df,
            "_analysis_revenue",
        )

    # --------------------------------------------------------
    # Numeric column fallback
    # --------------------------------------------------------

    numeric_columns = []

    for column in working_df.columns:

        numeric = pd.to_numeric(
            working_df[column],
            errors="coerce",
        )

        if numeric.notna().sum() > 0:
            numeric_columns.append(column)

    if not numeric_columns:

        raise ValueError(
            "No numeric metric column is "
            "available for time-series analysis."
        )

    return (
        working_df,
        numeric_columns[0],
    )


# ============================================================
# AGGREGATION
# ============================================================

def _aggregate_series(
    working_df: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:

    if aggregation == "count":

        return (
            working_df
            .groupby("_time_period")
            .size()
            .reset_index(
                name="_analysis_value"
            )
        )

    return (
        working_df
        .groupby("_time_period")[
            "_analysis_value"
        ]
        .agg(aggregation)
        .reset_index()
    )


# ============================================================
# TIME SERIES PREPARATION
# ============================================================

def prepare_time_series(
    df: pd.DataFrame,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    frequency: str = "monthly",
    aggregation: str = "sum",
) -> dict:
    """
    Prepare a clean aggregated time series.

    Supported frequencies:
        daily
        weekly
        monthly
        quarterly
        yearly

    Supported aggregations:
        sum
        mean
        min
        max
        median
        count
    """

    if df is None or df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    # --------------------------------------------------------
    # DATE COLUMN
    # --------------------------------------------------------

    date_column = _find_date_column(
        df,
        requested_column=x_column,
    )

    if date_column is None:
        raise ValueError(
            "No date or time column was found."
        )

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    working_df = df.copy()

    working_df[
        "_analysis_date"
    ] = dates

    working_df = working_df.dropna(
        subset=[
            "_analysis_date"
        ]
    )

    if working_df.empty:
        raise ValueError(
            "No valid dates are available."
        )

    # --------------------------------------------------------
    # VALUE COLUMN
    # --------------------------------------------------------

    (
        working_df,
        metric_column,
    ) = _resolve_value_column(
        working_df,
        y_column,
    )

    working_df[
        "_analysis_value"
    ] = pd.to_numeric(
        working_df[metric_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[
            "_analysis_value"
        ]
    )

    if working_df.empty:
        raise ValueError(
            "No valid numeric values are "
            "available for time-series analysis."
        )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    frequency_aliases = {

        "daily": "D",
        "day": "D",
        "d": "D",

        "weekly": "W",
        "week": "W",
        "w": "W",

        "monthly": "M",
        "month": "M",
        "m": "M",

        "quarterly": "Q",
        "quarter": "Q",
        "q": "Q",

        "yearly": "Y",
        "year": "Y",
        "y": "Y",
        "annual": "Y",
    }

    frequency_key = (
        str(frequency)
        .strip()
        .lower()
    )

    normalized_frequency = (
        frequency_aliases.get(
            frequency_key,
            frequency,
        )
    )

    allowed_frequencies = {
        "D",
        "W",
        "M",
        "Q",
        "Y",
    }

    if normalized_frequency not in allowed_frequencies:

        raise ValueError(
            f"Unsupported frequency "
            f"'{frequency}'. "
            f"Supported frequencies: "
            f"daily, weekly, monthly, "
            f"quarterly, yearly."
        )

    # --------------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------------

    aggregation = (
        str(aggregation)
        .strip()
        .lower()
    )

    allowed_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
        "median",
        "count",
    }

    if aggregation not in allowed_aggregations:

        raise ValueError(
            f"Unsupported aggregation "
            f"'{aggregation}'. "
            f"Supported aggregations: "
            f"sum, mean, min, max, "
            f"median, count."
        )

    # --------------------------------------------------------
    # CREATE TIME PERIOD
    # --------------------------------------------------------

    if normalized_frequency == "D":

        working_df[
            "_time_period"
        ] = (
            working_df[
                "_analysis_date"
            ].dt.floor("D")
        )

    elif normalized_frequency == "W":

        working_df[
            "_time_period"
        ] = (
            working_df[
                "_analysis_date"
            ]
            .dt.to_period("W")
            .dt.start_time
        )

    else:

        working_df[
            "_time_period"
        ] = (
            working_df[
                "_analysis_date"
            ]
            .dt.to_period(
                normalized_frequency
            )
            .dt.start_time
        )

    # --------------------------------------------------------
    # GROUP + AGGREGATE
    # --------------------------------------------------------

    grouped = _aggregate_series(
        working_df,
        aggregation,
    )

    grouped = grouped.sort_values(
        "_time_period"
    )

    # --------------------------------------------------------
    # RESULT POINTS
    # --------------------------------------------------------

    points = []

    for _, row in grouped.iterrows():

        timestamp = row[
            "_time_period"
        ]

        value = row[
            "_analysis_value"
        ]

        points.append(
            {
                "date": timestamp.strftime(
                    "%Y-%m-%d"
                ),
                "value": float(value),
            }
        )

    output_value_column = (
        "Revenue"
        if metric_column
        == "_analysis_revenue"
        else str(metric_column)
    )

    return {
        "date_column": str(
            date_column
        ),
        "value_column": (
            output_value_column
        ),
        "frequency": (
            normalized_frequency
        ),
        "aggregation": aggregation,
        "points": points,
    }


# ============================================================
# TIME ANALYSIS
# ============================================================

def time_analysis(
    df: pd.DataFrame,
) -> dict:

    if df is None or df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    date_column = _find_date_column(
        df
    )

    if date_column is None:

        raise ValueError(
            "No date or time column was found."
        )

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    valid_dates = dates.dropna()

    result = {
        "operation": "time_analysis",
        "date_column": str(
            date_column
        ),
        "valid_dates": int(
            valid_dates.notna().sum()
        ),
        "invalid_dates": int(
            dates.isna().sum()
        ),
        "date_range": None,
        "revenue_available": False,
        "monthly_trend": [],
        "daily_trend": [],
        "warnings": [],
    }

    if valid_dates.empty:

        result["warnings"].append(
            "No valid dates were found."
        )

        return result

    result["date_range"] = {
        "start": valid_dates.min().strftime(
            "%Y-%m-%d"
        ),
        "end": valid_dates.max().strftime(
            "%Y-%m-%d"
        ),
    }

    working_df = df.copy()

    working_df[
        "_analysis_date"
    ] = dates

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    try:

        (
            working_df,
            metric_column,
        ) = _resolve_value_column(
            working_df,
            "Total Revenue",
        )

        working_df[
            "_analysis_value"
        ] = pd.to_numeric(
            working_df[metric_column],
            errors="coerce",
        )

        revenue_df = (
            working_df
            .dropna(
                subset=[
                    "_analysis_date",
                    "_analysis_value",
                ]
            )
            .copy()
        )

        if not revenue_df.empty:

            result[
                "revenue_available"
            ] = True

            # ----------------------------------------------
            # Monthly trend
            # ----------------------------------------------

            monthly = (
                revenue_df
                .assign(
                    _month=revenue_df[
                        "_analysis_date"
                    ].dt.to_period("M")
                )
                .groupby("_month")[
                    "_analysis_value"
                ]
                .sum()
                .reset_index()
            )

            result[
                "monthly_trend"
            ] = [
                {
                    "month": str(
                        row["_month"]
                    ),
                    "revenue": float(
                        row[
                            "_analysis_value"
                        ]
                    ),
                }
                for _, row in monthly.iterrows()
            ]

            # ----------------------------------------------
            # Daily trend
            # ----------------------------------------------

            daily = (
                revenue_df
                .assign(
                    _day=revenue_df[
                        "_analysis_date"
                    ].dt.floor("D")
                )
                .groupby("_day")[
                    "_analysis_value"
                ]
                .sum()
                .reset_index()
            )

            result[
                "daily_trend"
            ] = [
                {
                    "date": row[
                        "_day"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                    "revenue": float(
                        row[
                            "_analysis_value"
                        ]
                    ),
                }
                for _, row in daily.iterrows()
            ]

        else:

            result["warnings"].append(
                "No valid numeric revenue values "
                "were available."
            )

    except ValueError:

        result["warnings"].append(
            "Revenue could not be calculated "
            "from the available columns."
        )

    # --------------------------------------------------------
    # Date warning
    # --------------------------------------------------------

    invalid_count = int(
        dates.isna().sum()
    )

    if invalid_count > 0:

        result["warnings"].append(
            f"{invalid_count} date values "
            f"could not be parsed."
        )

    return result
