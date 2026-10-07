from pathlib import Path
import pandas as pd

from src.execution.operation_executor import OperationExecutor
from src.agent.result_interpreter import ResultInterpreter
from src.agent.context import build_agent_context
from src.data.profiler import profile_dataset
from src.data.schema import build_schema


print("=" * 70)
print("TRUE MULTI-ROUND AUTONOMY — ROUND 0 TEST")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_excel(
    Path("data/sample/messy_business_data.xlsx")
)

user_query = "Revenue kam kyun hua?"


# ------------------------------------------------------------
# CONTEXT
# ------------------------------------------------------------

profile = profile_dataset(df)
schema = build_schema(df)

context = build_agent_context(
    df=df,
    profile=profile,
    schema=schema,
    user_query=user_query,
)


# ------------------------------------------------------------
# ONLY INITIAL EVIDENCE
# ------------------------------------------------------------

execution_steps = [
    {
        "step": 1,
        "operation": "revenue_calculations",
        "tool": "pandas_analysis",
        "description": "Calculate revenue.",
        "parameters": {
            "units_column": "Units Sold",
            "price_column": "Unit Price",
            "output_column": "Revenue",
        },
    },
    {
        "step": 2,
        "operation": "trend_analysis",
        "tool": "pandas_analysis",
        "description": "Analyze monthly revenue trend.",
        "parameters": {
            "date_column": "Date",
            "value_column": "Revenue",
            "aggregation": "sum",
        },
    },
]


# ------------------------------------------------------------
# EXECUTE ROUND 0
# ------------------------------------------------------------

executor = OperationExecutor()

execution = executor.execute(
    df=df,
    execution_steps=execution_steps,
)


print()
print("ROUND 0 EXECUTION")
print("STATUS:", execution["status"])
print("SUCCESSFUL:", execution["successful_steps"])
print("FAILED:", execution["failed_steps"])

for item in execution["results"]:
    print(
        f"STEP {item['step']} | "
        f"{item['operation']} | "
        f"{item['status']}"
    )


# ------------------------------------------------------------
# INTERPRET ROUND 0
# ------------------------------------------------------------

print()
print("=" * 70)
print("ROUND 0 INTERPRETATION")
print("=" * 70)

interpreter = ResultInterpreter(
    provider="nvidia"
)

decision = interpreter.interpret(
    user_query=user_query,
    execution_results=execution,
)

print()
print("DECISION:")
print(decision)


print()
print("=" * 70)

decision_type = decision.get("decision")

next_operations = decision.get(
    "next_operations",
    [],
)

print("DECISION TYPE:", decision_type)
print("NEXT OPERATIONS:", next_operations)

print()

if decision_type == "continue":
    print("STATUS: PASS")
    print()
    print(
        "The agent wants to continue investigation."
    )
    print(
        "True multi-round autonomy can proceed."
    )

elif decision_type == "final":
    print("STATUS: BLOCKED")
    print()
    print(
        "Interpreter stopped after trend analysis."
    )
    print(
        "The deterministic trend fallback is "
        "preventing true multi-round investigation."
    )

else:
    print("STATUS: NEEDS REVIEW")
    print(
        "Decision:",
        decision_type
    )
