from pathlib import Path
import pandas as pd

from src.execution.operation_executor import OperationExecutor
from src.agent.result_interpreter import ResultInterpreter
from src.agent.context import build_agent_context
from src.data.profiler import profile_dataset
from src.data.schema import build_schema


print("=" * 70)
print("HISTORY-AWARE AUTONOMOUS INTERPRETER TEST")
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


# ============================================================
# ROUND 0
# ============================================================

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
print("ROUND 0")
print(
    "Status:",
    round0["status"],
    "| Success:",
    round0["successful_steps"],
    "| Failed:",
    round0["failed_steps"],
)


# ============================================================
# ROUND 1
# ============================================================

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
        "description": "Analyze revenue by product category.",
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
print("ROUND 1")
print(
    "Status:",
    round1["status"],
    "| Success:",
    round1["successful_steps"],
    "| Failed:",
    round1["failed_steps"],
)


# ============================================================
# COMBINED HISTORY
# ============================================================

history = [
    {
        "iteration": 0,
        "execution": round0,
    },
    {
        "iteration": 1,
        "execution": round1,
    },
]


# ============================================================
# INTERPRET WITH FULL HISTORY
# ============================================================

print()
print("=" * 70)
print("HISTORY-AWARE INTERPRETATION")
print("=" * 70)

interpreter = ResultInterpreter(
    provider="nvidia"
)

decision = interpreter.interpret(
    user_query=user_query,
    execution_results=round1,
    execution_history=history,
)

print()
print("DECISION:")
print(decision)

decision_type = decision.get(
    "decision"
)

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
    print()
    print(
        "Interpreter successfully evaluated the "
        "current result with previous history."
    )

elif decision_type == "final":

    print("STATUS: PASS")
    print()
    print(
        "Interpreter determined the combined "
        "evidence is sufficient."
    )

else:

    print("STATUS: REVIEW")
    print()
    print(
        "Decision:",
        decision_type,
    )
