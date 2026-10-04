import pandas as pd

from src.execution.operation_executor import OperationExecutor


def test_unknown_operation_is_rejected():
    df = pd.DataFrame({
        "Region": ["North", "South"],
        "Revenue": [100, 200],
    })

    executor = OperationExecutor()

    result = executor.execute(
        df=df,
        execution_steps=[
            {
                "step": 1,
                "operation": "unknown_malicious_operation",
                "parameters": {},
            }
        ],
    )

    assert result["status"] == "error"
    assert result["successful_steps"] == 0
    assert result["failed_steps"] == 1

    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "error"
    assert (
        "Operation is not allowed"
        in result["results"][0]["error"]
    )


def test_non_dict_parameters_are_rejected():
    df = pd.DataFrame({
        "Region": ["North", "South"],
        "Revenue": [100, 200],
    })

    executor = OperationExecutor()

    result = executor.execute(
        df=df,
        execution_steps=[
            {
                "step": 1,
                "operation": "find_max",
                "parameters": ["Region", "Revenue"],
            }
        ],
    )

    assert result["status"] == "error"
    assert result["successful_steps"] == 0
    assert result["failed_steps"] == 1

    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "error"
    assert (
        result["results"][0]["error"]
        == "Parameters must be a dictionary."
    )


def test_invalid_column_is_rejected_safely():
    df = pd.DataFrame({
        "Region": ["North", "South"],
        "Revenue": [100, 200],
    })

    executor = OperationExecutor()

    result = executor.execute(
        df=df,
        execution_steps=[
            {
                "step": 1,
                "operation": "find_max",
                "parameters": {
                    "group_column": "HACKED_COLUMN",
                    "value_column": "Revenue",
                },
            }
        ],
    )

    assert result["status"] == "error"
    assert result["successful_steps"] == 0
    assert result["failed_steps"] == 1

    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "error"
    assert (
        "does not exist"
        in result["results"][0]["error"]
    )
