import json
import pandas as pd

from src.agent.context import build_agent_context
from src.agent.llm_planner import LLMPlanner
from src.agent.plan_validator import validate_plan
from src.agent.plan_adapter import adapt_plan
from src.execution.operation_executor import OperationExecutor
from src.agent.autonomous_e2e import AutonomousE2ERunner


DATA_PATH = "data/sample/messy_business_data.xlsx"
USER_QUERY = "Revenue kam kyun hua?"


def main():
    print("=" * 80)
    print("FULL AUTONOMOUS MULTI-ROUND E2E TEST")
    print("=" * 80)

    # ------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------

    df = pd.read_excel(DATA_PATH)

    print("\n[1] DATA LOADED")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # ------------------------------------------------------
    # 2. BUILD BASIC CONTEXT
    # ------------------------------------------------------

    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    schema = {
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
            }
            for column in df.columns
        ]
    }

    context = build_agent_context(
        df=df,
        profile=profile,
        schema=schema,
        user_query=USER_QUERY,
    )

    # ------------------------------------------------------
    # 3. INITIAL PLAN
    # ------------------------------------------------------

    print("\n[2] INITIAL LLM PLAN")
    print("-" * 80)

    planner = LLMPlanner(
        provider="nvidia",
    )

    raw_plan = planner.create_plan(
        user_query=USER_QUERY,
        context=context,
    )

    print(json.dumps(raw_plan, indent=2, default=str))

    # ------------------------------------------------------
    # 4. VALIDATE
    # ------------------------------------------------------

    print("\n[3] VALIDATION")
    print("-" * 80)

    validation = validate_plan(
        raw_plan,
        context=context,
    )

    print(json.dumps(validation, indent=2, default=str))

    if not validation.get("valid"):
        raise RuntimeError(
            "Initial plan validation failed."
        )

    validated_plan = validation[
        "validated_plan"
    ]

    # ------------------------------------------------------
    # 5. ADAPT
    # ------------------------------------------------------

    print("\n[4] PLAN ADAPTATION")
    print("-" * 80)

    adapted_plan = adapt_plan(
        validated_plan,
        context=context,
    )

    print(
        json.dumps(
            adapted_plan,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------
    # 6. INITIAL EXECUTION
    # ------------------------------------------------------

    print("\n[5] INITIAL EXECUTION")
    print("-" * 80)

    executor = OperationExecutor()

    initial_execution = executor.execute(
        df=df,
        execution_steps=adapted_plan["execution_steps"],
    )

    print(
        json.dumps(
            initial_execution,
            indent=2,
            default=str,
        )
    )

    if initial_execution.get("status") != "success":
        raise RuntimeError(
            "Initial execution failed."
        )

    # ------------------------------------------------------
    # 7. AUTONOMOUS LOOP
    # ------------------------------------------------------

    print("\n[6] AUTONOMOUS MULTI-ROUND LOOP")
    print("-" * 80)

    runner = AutonomousE2ERunner(
        provider="nvidia",
        max_iterations=4,
    )

    autonomous_result = runner.run(
        df=df,
        user_query=USER_QUERY,
        initial_execution=initial_execution,
        context=context,
    )

    # ------------------------------------------------------
    # 8. PRINT ROUND-BY-ROUND HISTORY
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("AUTONOMOUS INVESTIGATION HISTORY")
    print("=" * 80)

    history = autonomous_result.get(
        "history",
        [],
    )

    for item in history:

        iteration = item.get(
            "iteration"
        )

        execution = item.get(
            "execution",
            {},
        )

        print(
            f"\nROUND {iteration}"
        )

        print(
            "Status:",
            execution.get("status"),
        )

        print(
            "Total:",
            execution.get(
                "total_steps"
            ),
        )

        print(
            "Successful:",
            execution.get(
                "successful_steps"
            ),
        )

        print(
            "Failed:",
            execution.get(
                "failed_steps"
            ),
        )

        for result in execution.get(
            "results",
            [],
        ):

            operation = result.get(
                "operation"
            )

            status = result.get(
                "status"
            )

            print(
                f"  - {operation}: {status}"
            )

    # ------------------------------------------------------
    # 9. FINAL DECISION
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL AUTONOMOUS RESULT")
    print("=" * 80)

    print(
        json.dumps(
            autonomous_result,
            indent=2,
            default=str,
        )
    )

    print("\n" + "=" * 80)

    status = autonomous_result.get(
        "status"
    )

    if status == "success":
        print("STATUS: PASS")
        print(
            "Autonomous investigation completed."
        )
    else:
        print(
            "STATUS:",
            status,
        )


if __name__ == "__main__":
    main()
