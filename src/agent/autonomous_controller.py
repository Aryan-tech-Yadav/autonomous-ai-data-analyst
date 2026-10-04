from typing import Any

from src.agent.autonomous_loop import AutonomousLoopController
from src.agent.continuation_adapter import ContinuationAdapter
from src.agent.result_interpreter import ResultInterpreter


class AutonomousController:
    """
    Coordinates autonomous continuation of an analysis.

    Flow:

        execution results
              ↓
        ResultInterpreter
              ↓
        AutonomousLoopController
              ↓
        ContinuationAdapter
              ↓
        next analysis plan
    """

    def __init__(
        self,
        provider: str = "nvidia",
        max_iterations: int = 3,
    ):
        self.provider = provider

        self.interpreter = ResultInterpreter(
            provider=provider
        )

        self.loop = AutonomousLoopController(
            max_iterations=max_iterations
        )

        self.continuation_adapter = ContinuationAdapter()

    def start(self) -> dict[str, Any]:
        """Start a fresh autonomous analysis loop."""

        self.loop.start()

        return {
            "status": "started",
            "loop": self.loop.status(),
        }

    def inspect_execution(
        self,
        user_query: str,
        execution_result: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Interpret the latest execution result and decide
        whether the analysis should finish or continue.
        """

        decision = self.interpreter.interpret(
            user_query=user_query,
            execution_results=execution_result,
        )

        decision_type = decision.get("decision")

        # ------------------------------------------
        # FINAL
        # ------------------------------------------

        if decision_type == "final":
            return {
                "status": "final",
                "decision": decision,
                "loop": self.loop.status(),
            }

        # ------------------------------------------
        # REPAIR
        # ------------------------------------------

        if decision_type == "repair":
            return {
                "status": "repair",
                "decision": decision,
                "loop": self.loop.status(),
            }

        # ------------------------------------------
        # UNKNOWN DECISION
        # ------------------------------------------

        if decision_type != "continue":
            return {
                "status": "error",
                "decision": decision,
                "reason": (
                    f"Unsupported autonomous decision: "
                    f"{decision_type}"
                ),
                "loop": self.loop.status(),
            }

        # ------------------------------------------
        # LOOP LIMIT
        # ------------------------------------------

        if not self.loop.should_continue(decision):
            return {
                "status": "max_iterations",
                "decision": decision,
                "loop": self.loop.status(),
            }

        # ------------------------------------------
        # NEXT OPERATIONS
        # ------------------------------------------

        next_operations = decision.get(
            "next_operations",
            [],
        )

        if not isinstance(next_operations, list):
            return {
                "status": "error",
                "decision": decision,
                "reason": (
                    "ResultInterpreter returned invalid "
                    "next_operations."
                ),
                "loop": self.loop.status(),
            }

        if not next_operations:
            return {
                "status": "final",
                "decision": {
                    **decision,
                    "decision": "final",
                    "reason": (
                        "Continuation was requested but "
                        "no next operations were provided."
                    ),
                    "next_operations": [],
                },
                "loop": self.loop.status(),
            }

        # ------------------------------------------
        # DUPLICATE PROTECTION
        # ------------------------------------------

        if self.loop.has_seen_operations(
            next_operations
        ):
            return {
                "status": "stopped_duplicate",
                "decision": decision,
                "reason": (
                    "The same continuation operations "
                    "have already been attempted."
                ),
                "loop": self.loop.status(),
            }

        self.loop.register_operations(
            next_operations
        )

        # ------------------------------------------
        # ADAPT CONTINUATION
        # ------------------------------------------

        next_plan = self.continuation_adapter.adapt(
            next_operations=next_operations,
            start_step=self._next_step_number(
                execution_result
            ),
        )

        self.loop.next_iteration()

        return {
            "status": "continue",
            "decision": decision,
            "next_plan": next_plan,
            "loop": self.loop.status(),
        }

    def _next_step_number(
        self,
        execution_result: dict[str, Any],
    ) -> int:
        """
        Calculate the next execution step number
        from the latest execution result.
        """

        results = execution_result.get(
            "results",
            []
        )

        if not isinstance(results, list):
            return 1

        if not results:
            return 1

        step_numbers: list[int] = []

        for result in results:
            if not isinstance(result, dict):
                continue

            step = result.get("step")

            if isinstance(step, int):
                step_numbers.append(step)

        if not step_numbers:
            return 1

        return max(step_numbers) + 1
