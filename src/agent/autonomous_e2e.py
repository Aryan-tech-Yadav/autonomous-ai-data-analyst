from typing import Any

from src.agent.autonomous_controller import AutonomousController
from src.agent.plan_adapter import adapt_plan
from src.execution.operation_executor import OperationExecutor


class AutonomousE2ERunner:
    """
    Runs the autonomous analysis loop:

    Initial execution
        ↓
    Result interpretation
        ↓
    Continuation / repair decision
        ↓
    Continuation plan
        ↓
    Plan adaptation
        ↓
    Execution
        ↓
    Result interpretation again
    """

    def __init__(
        self,
        provider: str = "nvidia",
        max_iterations: int = 3,
    ):
        self.controller = AutonomousController(
            provider=provider,
            max_iterations=max_iterations,
        )

        self.executor = OperationExecutor()

    def run(
        self,
        df,
        user_query: str,
        initial_execution: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        context = context or {}

        controller_start = self.controller.start()

        current_execution = initial_execution

        history = [
            {
                "iteration": 0,
                "execution": current_execution,
            }
        ]

        while True:

            inspection = self.controller.inspect_execution(
                user_query=user_query,
                execution_result=current_execution,
                context=context,
                execution_history=history,
            )

            status = inspection.get("status")

            # --------------------------------------------------
            # Final answer is ready
            # --------------------------------------------------

            if status == "final":

                return {
                    "status": "success",
                    "controller": controller_start,
                    "history": history,
                    "final_decision": inspection,
                }

            # --------------------------------------------------
            # Repair required
            # --------------------------------------------------

            if status == "repair":

                return {
                    "status": "repair_required",
                    "controller": controller_start,
                    "history": history,
                    "repair": inspection,
                }

            # --------------------------------------------------
            # Autonomous loop stopped
            # --------------------------------------------------

            if status in {
                "max_iterations",
                "stopped_duplicate",
                "error",
            }:

                return {
                    "status": status,
                    "controller": controller_start,
                    "history": history,
                    "details": inspection,
                }

            # --------------------------------------------------
            # Only continuation is valid here
            # --------------------------------------------------

            if status != "continue":

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Unexpected autonomous "
                            "controller status."
                        ),
                        "inspection": inspection,
                    },
                }

            # --------------------------------------------------
            # Get continuation plan
            # --------------------------------------------------

            next_plan = inspection.get(
                "next_plan"
            )

            if not isinstance(
                next_plan,
                dict,
            ):

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Controller returned an "
                            "invalid continuation plan."
                        ),
                        "inspection": inspection,
                    },
                }

            # --------------------------------------------------
            # Validate analysis plan
            # --------------------------------------------------

            analysis_plan = next_plan.get(
                "analysis_plan",
                [],
            )

            if not isinstance(
                analysis_plan,
                list,
            ) or not analysis_plan:

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Continuation plan contains "
                            "no execution steps."
                        ),
                        "plan": next_plan,
                    },
                }

            # --------------------------------------------------
            # IMPORTANT:
            #
            # Re-run ContinuationAdapter with the actual
            # previous execution results.
            #
            # AutonomousController already creates the basic
            # continuation plan. Here we repair any missing
            # parameters deterministically.
            # --------------------------------------------------

            next_operations = inspection.get(
                "decision",
                {},
            ).get(
                "next_operations",
                [],
            )

            if not isinstance(
                next_operations,
                list,
            ):

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Continuation decision "
                            "contains invalid operations."
                        ),
                        "decision": inspection.get(
                            "decision"
                        ),
                    },
                }

            try:

                repaired_plan = (
                    self.controller.continuation_adapter.adapt(
                        next_operations=next_operations,
                        start_step=self._next_step_number(
                            current_execution
                        ),
                        execution_results=current_execution,
                        context=context,
                    )
                )

                adapted_plan = adapt_plan(
                    repaired_plan,
                    context=context,
                )

            except Exception as exc:

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Failed to adapt continuation plan."
                        ),
                        "error": str(exc),
                    },
                }

            execution_steps = adapted_plan.get(
                "execution_steps",
                [],
            )

            if not isinstance(
                execution_steps,
                list,
            ) or not execution_steps:

                return {
                    "status": "error",
                    "controller": controller_start,
                    "history": history,
                    "details": {
                        "message": (
                            "Adapted continuation plan "
                            "contains no execution steps."
                        ),
                        "plan": adapted_plan,
                    },
                }

            # --------------------------------------------------
            # Execute continuation
            # --------------------------------------------------

            current_execution = self.executor.execute(
                df=df,
                execution_steps=execution_steps,
            )

            history.append(
                {
                    "iteration": len(history),
                    "plan": adapted_plan,
                    "execution": current_execution,
                }
            )

    def _next_step_number(
        self,
        execution_result: dict[str, Any],
    ) -> int:

        results = execution_result.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ) or not results:

            return 1

        step_numbers = []

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            step = result.get(
                "step"
            )

            if isinstance(
                step,
                int,
            ):
                step_numbers.append(
                    step
                )

        if not step_numbers:
            return 1

        return max(step_numbers) + 1
