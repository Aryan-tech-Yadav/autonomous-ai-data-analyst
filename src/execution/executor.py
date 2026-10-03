from typing import Any, Dict

import pandas as pd

from src.tools.registry import get_tool


def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the tools selected by the analysis planner.

    Only tools registered in the tool registry
    can be executed.
    """

    execution_results = {
        "status": "success",
        "executed_tools": [],
        "results": {},
        "errors": [],
    }

    required_tools = plan.get(
        "required_tools",
        [],
    )

    # ==========================================
    # NO TOOLS
    # ==========================================

    if not required_tools:

        execution_results["status"] = "no_tools"

        return execution_results

    # ==========================================
    # EXECUTE REGISTERED TOOLS
    # ==========================================

    for tool_name in required_tools:

        try:

            tool = get_tool(tool_name)

            result = tool(df)

            execution_results[
                "executed_tools"
            ].append(tool_name)

            execution_results[
                "results"
            ][tool_name] = result

        except Exception as e:

            execution_results["status"] = "partial_failure"

            execution_results[
                "errors"
            ].append(
                {
                    "tool": tool_name,
                    "error": str(e),
                }
            )

    # ==========================================
    # COMPLETE FAILURE
    # ==========================================

    if (
        not execution_results["executed_tools"]
        and execution_results["errors"]
    ):

        execution_results["status"] = "failed"

    return execution_results
