from src.agent.insight_detector import InsightDetector


def test_ranking_insights_with_executor_wrapper():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "rank_by_value",
                "status": "success",
                "result": {
                    "operation": "rank_by_value",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "ascending": False,
                    "top_n": None,
                    "results": [
                        {
                            "rank": 1,
                            "group": "East",
                            "value": 82203.57,
                        },
                        {
                            "rank": 2,
                            "group": "West",
                            "value": 59258.52,
                        },
                        {
                            "rank": 3,
                            "group": "Central",
                            "value": 55820.07,
                        },
                        {
                            "rank": 4,
                            "group": "North",
                            "value": 54883.24,
                        },
                        {
                            "rank": 5,
                            "group": "South",
                            "value": 29586.05,
                        },
                    ],
                },
            }
        ],
    }

    result = detector.detect(
        execution_results
    )

    assert result["status"] == "success"
    assert result["count"] == 3

    types = {
        insight["type"]
        for insight in result["insights"]
    }

    assert "highest_performer" in types
    assert "lowest_performer" in types
    assert "performance_gap" in types

    highest = next(
        insight
        for insight in result["insights"]
        if insight["type"] == "highest_performer"
    )

    assert highest["group"] == "East"
    assert highest["value"] == 82203.57

    lowest = next(
        insight
        for insight in result["insights"]
        if insight["type"] == "lowest_performer"
    )

    assert lowest["group"] == "South"
    assert lowest["value"] == 29586.05

    gap = next(
        insight
        for insight in result["insights"]
        if insight["type"] == "performance_gap"
    )

    assert round(
        gap["gap"],
        2,
    ) == 52617.52


def test_groupby_insights_with_executor_wrapper():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "groupby_aggregation",
                "status": "success",
                "result": {
                    "operation": "groupby_aggregate",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "results": [
                        {
                            "Region": "Central",
                            "Revenue": 55820.07,
                        },
                        {
                            "Region": "East",
                            "Revenue": 82203.57,
                        },
                        {
                            "Region": "North",
                            "Revenue": 54883.24,
                        },
                        {
                            "Region": "South",
                            "Revenue": 29586.05,
                        },
                        {
                            "Region": "West",
                            "Revenue": 59258.52,
                        },
                    ],
                },
            }
        ],
    }

    result = detector.detect(
        execution_results
    )

    assert result["status"] == "success"
    assert result["count"] == 2

    highest = next(
        insight
        for insight in result["insights"]
        if insight["type"] == "group_highest"
    )

    assert highest["group"] == "East"
    assert highest["value"] == 82203.57

    lowest = next(
        insight
        for insight in result["insights"]
        if insight["type"] == "group_lowest"
    )

    assert lowest["group"] == "South"
    assert lowest["value"] == 29586.05


def test_percentage_change_insights():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "calculate_percentage_change",
                "status": "success",
                "result": {
                    "operation": "calculate_percentage_change",
                    "mean_percentage_change": 17.0,
                    "minimum_percentage_change": 10.0,
                    "maximum_percentage_change": 25.0,
                },
            }
        ],
    }

    result = detector.detect(
        execution_results
    )

    assert result["status"] == "success"
    assert result["count"] == 2

    average = next(
        insight
        for insight in result["insights"]
        if insight["type"]
        == "average_percentage_change"
    )

    assert average["value"] == 17.0

    change_range = next(
        insight
        for insight in result["insights"]
        if insight["type"]
        == "percentage_change_range"
    )

    assert change_range["minimum"] == 10.0
    assert change_range["maximum"] == 25.0


def test_categorical_insight():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "categorical_analysis",
                "status": "success",
                "result": {
                    "operation": "categorical_analysis",
                    "column": "Region",
                    "counts": {
                        "East": 40,
                        "West": 30,
                        "North": 20,
                        "South": 10,
                    },
                },
            }
        ],
    }

    result = detector.detect(
        execution_results
    )

    assert result["status"] == "success"
    assert result["count"] == 1

    insight = result["insights"][0]

    assert insight["type"] == "dominant_category"
    assert insight["category"] == "East"
    assert insight["count"] == 40
    assert insight["share_percentage"] == 40.0


def test_failed_operations_are_ignored():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "rank_by_value",
                "status": "error",
                "result": {},
                "error": "Invalid column",
            }
        ],
    }

    result = detector.detect(
        execution_results
    )

    assert result["status"] == "success"
    assert result["count"] == 0
    assert result["insights"] == []


def test_duplicate_groupby_and_ranking_insights_are_consolidated():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "groupby_aggregation",
                "status": "success",
                "result": {
                    "operation": "groupby_aggregate",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "results": [
                        {
                            "Region": "East",
                            "Revenue": 82203.57,
                        },
                        {
                            "Region": "West",
                            "Revenue": 59258.52,
                        },
                        {
                            "Region": "South",
                            "Revenue": 29586.05,
                        },
                    ],
                },
            },
            {
                "operation": "rank_by_value",
                "status": "success",
                "result": {
                    "operation": "rank_by_value",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "ascending": False,
                    "top_n": None,
                    "results": [
                        {
                            "rank": 1,
                            "group": "East",
                            "value": 82203.57,
                        },
                        {
                            "rank": 2,
                            "group": "West",
                            "value": 59258.52,
                        },
                        {
                            "rank": 3,
                            "group": "South",
                            "value": 29586.05,
                        },
                    ],
                },
            },
        ],
    }

    result = detector.detect(
        execution_results
    )

    types = [
        insight["type"]
        for insight in result["insights"]
    ]

    assert "highest_performer" in types
    assert "lowest_performer" in types
    assert "performance_gap" in types

    assert "group_highest" not in types
    assert "group_lowest" not in types

    assert result["count"] == 3


def test_duplicate_groupby_and_ranking_insights_are_consolidated():
    detector = InsightDetector()

    execution_results = {
        "status": "success",
        "results": [
            {
                "operation": "groupby_aggregation",
                "status": "success",
                "result": {
                    "operation": "groupby_aggregate",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "results": [
                        {
                            "Region": "East",
                            "Revenue": 82203.57,
                        },
                        {
                            "Region": "West",
                            "Revenue": 59258.52,
                        },
                        {
                            "Region": "South",
                            "Revenue": 29586.05,
                        },
                    ],
                },
            },
            {
                "operation": "rank_by_value",
                "status": "success",
                "result": {
                    "operation": "rank_by_value",
                    "group_column": "Region",
                    "value_column": "Revenue",
                    "aggregation": "sum",
                    "ascending": False,
                    "top_n": None,
                    "results": [
                        {
                            "rank": 1,
                            "group": "East",
                            "value": 82203.57,
                        },
                        {
                            "rank": 2,
                            "group": "West",
                            "value": 59258.52,
                        },
                        {
                            "rank": 3,
                            "group": "South",
                            "value": 29586.05,
                        },
                    ],
                },
            },
        ],
    }

    result = detector.detect(
        execution_results
    )

    types = [
        insight["type"]
        for insight in result["insights"]
    ]

    assert "highest_performer" in types
    assert "lowest_performer" in types
    assert "performance_gap" in types

    assert "group_highest" not in types
    assert "group_lowest" not in types

    assert result["count"] == 3
