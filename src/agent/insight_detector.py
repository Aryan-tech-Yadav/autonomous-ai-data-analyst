from __future__ import annotations

from typing import Any, Dict, List


class InsightDetector:
    """
    Detect deterministic business insights from verified
    operation-execution results.

    Important:
    - This class does NOT execute dataframe operations.
    - It does NOT invent new numeric results.
    - It only interprets successful results produced by
      trusted analysis operations.
    - It understands the OperationExecutor wrapper format:

        {
            "operation": "...",
            "status": "success",
            "result": {
                ...
            }
        }
    """

    def detect(
        self,
        execution_results: Dict[str, Any],
    ) -> Dict[str, Any]:

        results = execution_results.get(
            "results",
            [],
        )

        if not isinstance(results, list):
            results = []

        insights: List[Dict[str, Any]] = []

        for execution_item in results:

            if not isinstance(
                execution_item,
                dict,
            ):
                continue

            if execution_item.get(
                "status"
            ) != "success":
                continue

            operation = str(
                execution_item.get(
                    "operation",
                    "",
                )
            ).strip().lower()

            result = self._unwrap_result(
                execution_item
            )

            if not result:
                continue

            if operation in {
                "rank_by_value",
                "ranking",
            }:
                insights.extend(
                    self._from_ranking(
                        result
                    )
                )

            elif operation == "find_max":
                insights.extend(
                    self._from_find_max(
                        result
                    )
                )

            elif operation in {
                "calculate_percentage_change",
                "percentage_change",
            }:
                insights.extend(
                    self._from_percentage_change(
                        result
                    )
                )

            elif operation in {
                "statistics",
                "calculate_statistics",
            }:
                insights.extend(
                    self._from_statistics(
                        result
                    )
                )

            elif operation == "categorical_analysis":
                insights.extend(
                    self._from_categorical_analysis(
                        result
                    )
                )

            elif operation in {
                "groupby_aggregation",
                "groupby_aggregate",
                "group_by",
                "groupby",
            }:
                insights.extend(
                    self._from_groupby(
                        result
                    )
                )

            elif operation == "trend_analysis":
                insights.extend(
                    self._from_trend_analysis(
                        result
                    )
                )

        insights = self._deduplicate(
            insights
        )

        return {
            "status": "success",
            "insights": insights,
            "count": len(insights),
        }

    # ==========================================================
    # EXECUTOR RESULT NORMALIZATION
    # ==========================================================

    def _unwrap_result(
        self,
        execution_item: Dict[str, Any],
    ) -> Dict[str, Any]:

        nested = execution_item.get(
            "result"
        )

        if isinstance(nested, dict):
            return nested

        # Backward compatibility for direct operation results.
        return execution_item

    # ==========================================================
    # RANKING
    # ==========================================================

    def _from_ranking(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert rank_by_value results into deterministic
        business insights.

        The executor's `ascending` flag is authoritative:

        ascending=False -> highest-first ranking
        ascending=True  -> lowest-first ranking
        """

        rows = result.get(
            "results",
            [],
        )

        if not isinstance(rows, list):
            return []

        valid = []

        for item in rows:

            if not isinstance(item, dict):
                continue

            group = item.get("group")
            value = item.get("value")

            if group is None or value is None:
                continue

            try:
                numeric_value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            valid.append(
                {
                    "rank": item.get("rank"),
                    "group": str(group),
                    "value": numeric_value,
                }
            )

        if not valid:
            return []

        # ------------------------------------------------------
        # Lowest-first ranking
        # ------------------------------------------------------

        ascending = bool(
            result.get(
                "ascending",
                False,
            )
        )

        if ascending:

            lowest = valid[0]

            return [
                {
                    "type": "lowest_performer",
                    "operation": "rank_by_value",
                    "message": (
                        f"{lowest['group']} ranks lowest with "
                        f"{lowest['value']:,.2f}."
                    ),
                    "group": lowest["group"],
                    "value": lowest["value"],
                }
            ]

        # ------------------------------------------------------
        # Highest-first ranking
        # ------------------------------------------------------

        highest = valid[0]

        insights = [
            {
                "type": "highest_performer",
                "operation": "rank_by_value",
                "message": (
                    f"{highest['group']} ranks highest with "
                    f"{highest['value']:,.2f}."
                ),
                "group": highest["group"],
                "value": highest["value"],
            }
        ]

        # A single top-N result only establishes the highest
        # performer. Do not invent a lowest performer.
        if len(valid) <= 1:
            return insights

        # A complete descending ranking establishes both
        # highest and lowest performers.
        lowest = valid[-1]

        insights.append(
            {
                "type": "lowest_performer",
                "operation": "rank_by_value",
                "message": (
                    f"{lowest['group']} ranks lowest with "
                    f"{lowest['value']:,.2f}."
                ),
                "group": lowest["group"],
                "value": lowest["value"],
            }
        )

        gap = (
            highest["value"]
            - lowest["value"]
        )

        insights.append(
            {
                "type": "performance_gap",
                "operation": "rank_by_value",
                "message": (
                    "The gap between the highest and "
                    "lowest performers is "
                    f"{gap:,.2f}."
                ),
                "highest_group": highest["group"],
                "lowest_group": lowest["group"],
                "gap": gap,
            }
        )

        return insights

    # ==========================================================
    # FIND MAX
    # ==========================================================

    def _from_find_max(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        group = result.get(
            "group"
        )

        value = result.get(
            "value"
        )

        if group is None or value is None:
            return []

        try:
            numeric_value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return []

        return [
            {
                "type": "maximum",
                "operation": "find_max",
                "message": (
                    f"{group} has the highest value at "
                    f"{numeric_value:,.2f}."
                ),
                "group": str(group),
                "value": numeric_value,
            }
        ]

    # ==========================================================
    # PERCENTAGE CHANGE
    # ==========================================================

    def _from_percentage_change(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        insights = []

        mean_change = result.get(
            "mean_percentage_change"
        )

        minimum_change = result.get(
            "minimum_percentage_change"
        )

        maximum_change = result.get(
            "maximum_percentage_change"
        )

        if mean_change is not None:

            try:
                mean_value = float(
                    mean_change
                )
            except (
                TypeError,
                ValueError,
            ):
                mean_value = None

            if mean_value is not None:

                if mean_value > 0:
                    direction = "increase"

                elif mean_value < 0:
                    direction = "decrease"

                else:
                    direction = "no change"

                insights.append(
                    {
                        "type":
                            "average_percentage_change",
                        "operation":
                            "calculate_percentage_change",
                        "message": (
                            "The average percentage change "
                            f"is {mean_value:.2f}%, indicating "
                            f"{direction}."
                        ),
                        "value": mean_value,
                    }
                )

        if (
            minimum_change is not None
            and maximum_change is not None
        ):

            try:
                minimum_value = float(
                    minimum_change
                )

                maximum_value = float(
                    maximum_change
                )

            except (
                TypeError,
                ValueError,
            ):
                minimum_value = None
                maximum_value = None

            if (
                minimum_value is not None
                and maximum_value is not None
            ):

                insights.append(
                    {
                        "type":
                            "percentage_change_range",
                        "operation":
                            "calculate_percentage_change",
                        "message": (
                            "Percentage change ranges from "
                            f"{minimum_value:.2f}% to "
                            f"{maximum_value:.2f}%."
                        ),
                        "minimum": minimum_value,
                        "maximum": maximum_value,
                    }
                )

        return insights

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def _from_statistics(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        mean = result.get(
            "mean"
        )

        median = result.get(
            "median"
        )

        if mean is None or median is None:
            return []

        try:
            mean_value = float(
                mean
            )

            median_value = float(
                median
            )

        except (
            TypeError,
            ValueError,
        ):
            return []

        if mean_value > median_value:
            relationship = "above"

        elif mean_value < median_value:
            relationship = "below"

        else:
            relationship = "equal to"

        return [
            {
                "type": "mean_median_relationship",
                "operation": "statistics",
                "message": (
                    f"The mean ({mean_value:,.2f}) is "
                    f"{relationship} the median "
                    f"({median_value:,.2f})."
                ),
                "mean": mean_value,
                "median": median_value,
            }
        ]

    # ==========================================================
    # CATEGORICAL ANALYSIS
    # ==========================================================

    def _from_categorical_analysis(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        counts = result.get(
            "counts",
            {},
        )

        if not isinstance(
            counts,
            dict,
        ) or not counts:
            return []

        valid_counts = []

        for category, count in counts.items():

            try:
                numeric_count = int(
                    count
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            valid_counts.append(
                (
                    str(category),
                    numeric_count,
                )
            )

        if not valid_counts:
            return []

        highest_category, highest_count = max(
            valid_counts,
            key=lambda item: item[1],
        )

        total = sum(
            count
            for _, count in valid_counts
        )

        share = (
            highest_count
            / total
            * 100
            if total > 0
            else 0.0
        )

        return [
            {
                "type": "dominant_category",
                "operation": "categorical_analysis",
                "message": (
                    f"{highest_category} is the most common "
                    f"category with {highest_count} records "
                    f"({share:.2f}% of the observed records)."
                ),
                "category": highest_category,
                "count": highest_count,
                "share_percentage": share,
            }
        ]

    # ==========================================================
    # GROUPBY
    # ==========================================================

    def _from_groupby(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        rows = result.get(
            "results",
            [],
        )

        if not isinstance(
            rows,
            list,
        ) or not rows:
            return []

        group_column = result.get(
            "group_column"
        )

        value_column = result.get(
            "value_column"
        )

        if not group_column or not value_column:
            return []

        valid = []

        for item in rows:

            if not isinstance(
                item,
                dict,
            ):
                continue

            group = item.get(
                group_column
            )

            value = item.get(
                value_column
            )

            if group is None or value is None:
                continue

            try:
                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            valid.append(
                {
                    "group": str(group),
                    "value": numeric_value,
                }
            )

        if not valid:
            return []

        highest = max(
            valid,
            key=lambda item: item["value"],
        )

        lowest = min(
            valid,
            key=lambda item: item["value"],
        )

        insights = [
            {
                "type": "group_highest",
                "operation": "groupby_aggregation",
                "message": (
                    f"{highest['group']} has the highest "
                    "aggregated value at "
                    f"{highest['value']:,.2f}."
                ),
                "group": highest["group"],
                "value": highest["value"],
            }
        ]

        if len(valid) > 1:

            insights.append(
                {
                    "type": "group_lowest",
                    "operation": "groupby_aggregation",
                    "message": (
                        f"{lowest['group']} has the lowest "
                        "aggregated value at "
                        f"{lowest['value']:,.2f}."
                    ),
                    "group": lowest["group"],
                    "value": lowest["value"],
                }
            )

        return insights

    # ==========================================================
    # TREND ANALYSIS
    # ==========================================================

    def _from_trend_analysis(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert trend_analysis results into deterministic
        business insights.

        The trend operation provides monthly periods and values.
        This method only interprets those verified results.
        """

        rows = result.get(
            "results",
            [],
        )

        if not isinstance(rows, list):
            return []

        valid = []

        for item in rows:
            if not isinstance(item, dict):
                continue

            period = item.get("period")
            value = item.get("value")

            if period is None or value is None:
                continue

            try:
                numeric_value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            valid.append(
                {
                    "period": str(period),
                    "value": numeric_value,
                }
            )

        if not valid:
            return []

        insights: List[Dict[str, Any]] = []

        highest = max(
            valid,
            key=lambda item: item["value"],
        )

        lowest = min(
            valid,
            key=lambda item: item["value"],
        )

        insights.append(
            {
                "type": "trend_highest_period",
                "operation": "trend_analysis",
                "message": (
                    f"{highest['period']} has the highest "
                    f"value at {highest['value']:,.2f}."
                ),
                "period": highest["period"],
                "value": highest["value"],
            }
        )

        if len(valid) > 1:
            insights.append(
                {
                    "type": "trend_lowest_period",
                    "operation": "trend_analysis",
                    "message": (
                        f"{lowest['period']} has the lowest "
                        f"value at {lowest['value']:,.2f}."
                    ),
                    "period": lowest["period"],
                    "value": lowest["value"],
                }
            )

        overall_change = result.get(
            "overall_percentage_change"
        )

        if overall_change is not None:
            try:
                change_value = float(
                    overall_change
                )
            except (
                TypeError,
                ValueError,
            ):
                change_value = None

            if change_value is not None:
                if change_value > 0:
                    direction = "increased"
                elif change_value < 0:
                    direction = "decreased"
                else:
                    direction = "remained unchanged"

                insights.append(
                    {
                        "type": "trend_overall_change",
                        "operation": "trend_analysis",
                        "message": (
                            f"Overall value {direction} by "
                            f"{abs(change_value):.2f}% "
                            f"from the first to the last period."
                        ),
                        "value": change_value,
                    }
                )

        if len(valid) > 1:
            first = valid[0]
            last = valid[-1]

            if (
                first["value"] > 0
                and last["value"] != first["value"]
            ):
                insights.append(
                    {
                        "type": "trend_direction",
                        "operation": "trend_analysis",
                        "message": (
                            f"The trend moved from "
                            f"{first['period']} "
                            f"({first['value']:,.2f}) to "
                            f"{last['period']} "
                            f"({last['value']:,.2f})."
                        ),
                        "start_period": first["period"],
                        "start_value": first["value"],
                        "end_period": last["period"],
                        "end_value": last["value"],
                    }
                )

        return insights

    # ==========================================================
    # DEDUPLICATION
    # ==========================================================

    def _deduplicate(
        self,
        insights: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        # ------------------------------------------------------
        # First remove exact duplicate insights.
        # ------------------------------------------------------

        unique: List[Dict[str, Any]] = []
        seen = set()

        for insight in insights:

            if not isinstance(
                insight,
                dict,
            ):
                continue

            key = (
                insight.get("type"),
                insight.get("group"),
                insight.get("category"),
                insight.get("value"),
                insight.get("gap"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(insight)

        # ------------------------------------------------------
        # Prefer ranking insights over equivalent groupby
        # insights when both operations describe the same
        # grouped metric.
        #
        # rank_by_value is more explicit because it directly
        # establishes ordering.
        # ------------------------------------------------------

        has_ranking_highest = any(
            insight.get("type")
            == "highest_performer"
            and insight.get("operation")
            == "rank_by_value"
            for insight in unique
        )

        has_ranking_lowest = any(
            insight.get("type")
            == "lowest_performer"
            and insight.get("operation")
            == "rank_by_value"
            for insight in unique
        )

        # ------------------------------------------------------
        # Prefer explicit find_max results over equivalent
        # groupby highest insights.
        #
        # Example:
        #   find_max  -> East = 82,203.57
        #   groupby    -> East has highest = 82,203.57
        #
        # Both communicate the same fact, so keep only the
        # explicit maximum result.
        # ------------------------------------------------------

        find_max_highest = {
            (
                insight.get("group"),
                round(float(insight.get("value")), 10)
            )
            for insight in unique
            if insight.get("type") == "maximum"
            and insight.get("operation") == "find_max"
            and insight.get("group") is not None
            and insight.get("value") is not None
        }

        find_max_lowest = {
            (
                insight.get("group"),
                round(float(insight.get("value")), 10)
            )
            for insight in unique
            if insight.get("type") == "minimum"
            and insight.get("operation") == "find_max"
            and insight.get("group") is not None
            and insight.get("value") is not None
        }

        consolidated = []

        for insight in unique:

            insight_type = insight.get(
                "type"
            )

            if (
                insight_type == "group_highest"
                and has_ranking_highest
            ):
                continue

            if (
                insight_type == "group_lowest"
                and has_ranking_lowest
            ):
                continue

            # Remove groupby insights when an identical explicit
            # find_max result already establishes the same fact.
            if insight_type == "group_highest":
                key = (
                    insight.get("group"),
                    round(float(insight.get("value")), 10)
                    if insight.get("value") is not None
                    else None,
                )
                if key in find_max_highest:
                    continue

            if insight_type == "group_lowest":
                key = (
                    insight.get("group"),
                    round(float(insight.get("value")), 10)
                    if insight.get("value") is not None
                    else None,
                )
                if key in find_max_lowest:
                    continue

            consolidated.append(
                insight
            )

        return consolidated
