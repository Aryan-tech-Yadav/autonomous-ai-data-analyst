from pathlib import Path
import pandas as pd

from src.agent.llm_planner import LLMPlanner
from src.agent.plan_validator import validate_plan
from src.agent.plan_adapter import adapt_plan
from src.agent.autonomous_e2e import AutonomousE2ERunner
from src.data.profiler import profile_dataset
from src.data.schema import build_schema
from src.agent.context import build_agent_context


DATASET = Path(
    "data/sample/messy_business_data.xlsx"
)

USER_QUERY = "Revenue kam kyun hua?"


print("=" * 70)
print("REAL AUTONOMOUS INVESTIGATION TEST")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_excel(DATASET)

print()
print("DATASET:")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ------------------------------------------------------------
# 2. PROFILE + SCHEMA + CONTEXT
# ------------------------------------------------------------

profile = profile_dataset(df)
schema = build_schema(df)

context = build_agent_context(
    df=df,
    profile=profile,
    schema=schema,
    user_query=USER_QUERY,
)

print()
print("=" * 70)
print("1. PLANNING")
print("=" * 70)

planner = LLMPlanner(
    provider="nvidia"
)

raw_plan = planner.create_plan(
    user_query=USER_QUERY,
    context=context,
    conversation_history=[],
)

print("RAW PLAN TYPE:", type(raw_plan).__name__)

if isinstance(raw_plan, str):
    print(raw_plan[:5000])
else:
    print(raw_plan)


# ------------------------------------------------------------
# 3. VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("2. VALIDATION")
print("=" * 70)

validation = validate_plan(
    raw_plan,
    context,
)

print(validation)

if not validation.get("valid"):
    print()
    print("VALIDATION FAILED")
    raise SystemExit(1)

print()
print("VALIDATION: PASS")


# ------------------------------------------------------------
# 4. ADAPTATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("3. ADAPTATION")
print("=" * 70)

adapted = adapt_plan(
    validation["validated_plan"],
    context=context,
)

print(adapted)

print()
print(
    "EXECUTION STEPS:",
    len(adapted.get("execution_steps", []))
)

if not adapted.get("execution_steps"):
    print("ADAPTATION FAILED")
    raise SystemExit(1)

print("ADAPTATION: PASS")


# ------------------------------------------------------------
# 5. INITIAL EXECUTION
# ------------------------------------------------------------

print()
print("=" * 70)
print("4. INITIAL EXECUTION")
print("=" * 70)

from src.execution.operation_executor import OperationExecutor

executor = OperationExecutor()

initial_execution = executor.execute(
    df=df,
    execution_steps=adapted["execution_steps"],
)

print(initial_execution)

print()
print(
    "Initial successful:",
    initial_execution.get("successful_steps"),
)

print(
    "Initial failed:",
    initial_execution.get("failed_steps"),
)


# ------------------------------------------------------------
# 6. AUTONOMOUS INVESTIGATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("5. AUTONOMOUS INVESTIGATION")
print("=" * 70)

runner = AutonomousE2ERunner(
    provider="nvidia",
    max_iterations=3,
)

result = runner.run(
    df=df,
    user_query=USER_QUERY,
    initial_execution=initial_execution,
    context=context,
)

print()
print("=" * 70)
print("AUTONOMOUS RESULT")
print("=" * 70)

print("STATUS:", result.get("status"))
print("ROUNDS:", len(result.get("history", [])))

history = result.get("history", [])

for round_index, round_data in enumerate(history):

    print()
    print("-" * 60)
    print("ROUND:", round_index)

    execution = round_data.get("execution", {})

    print(
        "Execution status:",
        execution.get("status")
    )

    print(
        "Successful:",
        execution.get("successful_steps")
    )

    print(
        "Failed:",
        execution.get("failed_steps")
    )

    for item in execution.get("results", []):

        print(
            f"  STEP {item.get('step')} | "
            f"{item.get('operation')} | "
            f"{item.get('status')}"
        )

        if item.get("status") == "error":
            print(
                "    ERROR:",
                item.get("error")
            )

    decision = round_data.get("decision")

    if decision:
        print()
        print("Decision:")
        print(decision)


print()
print("=" * 70)
print("FINAL DECISION")
print("=" * 70)

print(
    result.get("final_decision")
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
