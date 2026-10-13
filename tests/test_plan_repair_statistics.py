from src.agent.plan_repair import PlanRepair


def build_context(columns):
    return {
        "schema": {
            "columns": columns,
        },
    }


def build_plan(parameters):
    return {
        "analysis_plan": [
            {
                "step": 1,
                "operation": "statistics",
                "tool": "pandas_analysis",
                "parameters": parameters,
            }
        ]
    }


def build_validation_result():
    return {
        "valid": False,
        "errors": ["Statistics column is missing."],
    }


def test_statistics_selects_numeric_column_when_missing():
    context = build_context([
        {
            "name": "Region",
            "dtype": "object",
            "role": "categorical",
        },
        {
            "name": "Revenue",
            "dtype": "float64",
            "role": "numeric",
        },
    ])

    repairer = PlanRepair(context)
    result = repairer.repair(
        build_plan({}),
        build_validation_result(),
    )

    parameters = result["plan"]["analysis_plan"][0]["parameters"]

    assert parameters["column"] == "Revenue"
    assert result["repaired"] is True
    assert any(
        "Selected numeric column 'Revenue'" in change
        for change in result["changes"]
    )


def test_statistics_preserves_existing_valid_column():
    context = build_context([
        {
            "name": "Revenue",
            "dtype": "float64",
            "role": "numeric",
        },
        {
            "name": "Quantity",
            "dtype": "int64",
            "role": "numeric",
        },
    ])

    repairer = PlanRepair(context)
    result = repairer.repair(
        build_plan({"column": "Quantity"}),
        build_validation_result(),
    )

    parameters = result["plan"]["analysis_plan"][0]["parameters"]

    assert parameters["column"] == "Quantity"
    assert not any(
        "Selected numeric column" in change
        for change in result["changes"]
    )


def test_statistics_records_diagnostic_without_numeric_column():
    context = build_context([
        {
            "name": "Region",
            "dtype": "object",
            "role": "categorical",
        },
        {
            "name": "Product",
            "dtype": "object",
            "role": "text",
        },
    ])

    repairer = PlanRepair(context)
    result = repairer.repair(
        build_plan({}),
        build_validation_result(),
    )

    parameters = result["plan"]["analysis_plan"][0]["parameters"]

    assert "column" not in parameters
    assert result["repaired"] is False
    assert any(
        "no suitable numeric column" in change
        for change in result["changes"]
    )
