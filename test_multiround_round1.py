from pathlib import Path
import pandas as pd

from src.execution.operation_executor import OperationExecutor
from src.agent.result_interpreter import ResultInterpreter
from src.agent.context import build_agent_context
from src.data.profiler import profile_dataset
from src.data.schema import build_schema


print("=" * 70)
print("TRUE MULTI-ROUND AUTONOMY — ROUND 1")
print("=" * 70)

df = pd.read_excel(
    Path("data/sample/messy_business_data.xlsx")
)

user_query = "Revenue kam kyun hua?"

profile = profile_dataset(df)
schema = build_schema(df)

context = build_agent_context(
    df=df,
    profile=profile,
    schema=schema,
    user_query=user_query,
)

executor = OperationExecutor()

# Round 0 evidence
round0_steps = [
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

round0 = executor.execute(
    df=df,
    execution_steps=round0_steps,
)

print()
print("ROUND 0:", round0["status"])


# Simulated agent decision from Round 0
next_operation = {
    "operation": "categorical_analysis",
    "parameters": {
        "column": "Product Category"
    },
    "description": (
        "Analyze revenue by product category "
        "to identify drivers of the decline."
    ),
}

print()
print("NEXT OPERATION SELECTED BY AGENT:")
print(next_operation)


# Resolve value column for categorical analysis.
# Revenue is a derived column, so explicitly calculate it first.
round1_steps = [
    {
        "step": 3,
        "operation": "revenue_calculations",
        "tool": "pandas_analysis",
        "description": "Ensure Revenue exists.",
        "parameters": {
            "units_column": "Units Sold",
            "price_column": "Unit Price",
            "output_column": "Revenue",
        },
    },
    {
        "step": 4,
        "operation": "groupby_aggregate",
        "tool": "pandas_analysis",
        "description": (
            "Analyze revenue by product category."
        ),
        "parameters": {
            "group_column": "Product Category",
            "value_column": "Revenue",
            "aggregation": "sum",
        },
    },
]

round1 = executor.execute(
    df=df,
    execution_steps=round1_steps,
)

print()
print("=" * 70)
print("ROUND 1 EXECUTION")
print("=" * 70)

print("STATUS:", round1["status"])
print("SUCCESSFUL:", round1["successful_steps"])
print("FAILED:", round1["failed_steps"])

for item in round1["results"]:
    print(
        f"STEP {item['step']} | "
        f"{item['operation']} | "
        f"{item['status']}"
    )

    if item["status"] == "error":
        print("ERROR:", item.get("error"))

print()
print("PRODUCT CATEGORY RESULT:")

for item in round1["results"]:
    if item["operation"] == "groupby_aggregate":
        print(item["result"])


# ------------------------------------------------------------
# ROUND 1 → INTERPRET
# ------------------------------------------------------------

print()
print("=" * 70)
print("ROUND 1 INTERPRETATION")
print("=" * 70)

interpreter = ResultInterpreter(
    provider="nvidia"
)

decision = interpreter.interpret(
    user_query=user_query,
    execution_results=round1,
)

print()
print("DECISION:")
print(decision)

decision_type = decision.get("decision")
next_operations = decision.get(
    "next_operations",
    [],
)

print()
print("DECISION TYPE:", decision_type)
print("NEXT OPERATIONS:", next_operations)

print()
print("=" * 70)

if decision_type == "continue":
    print("STATUS: PASS")
    print("Agent wants another investigation round.")

elif decision_type == "final":
    print("STATUS: PASS")
    print("Agent determined that evidence is sufficient.")

else:
    print("STATUS: REVIEW")
    print("Decision:", decision_type)
