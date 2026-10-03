from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agent.plan_repair import PlanRepair


class ExecutionRecovery:
    """
    Recovery layer for failed operation-executor steps.

    The recovery layer performs deterministic repairs on failed
    execution steps and prepares a retry plan.

    It does not execute operations itself.
    """

    def __init__(self, context: dict[str, Any]):
        self.context = context
        self.repairer = PlanRepair(context=context)

    def recover(
        self,
        execution_result: dict[str, Any],
        execution_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Attempt to recover from an execution failure.

        Returns:
            {
                "recoverable": bool,
                "repaired_plan": dict,
                "changes": list[str],
                "reason": str,
            }
        """

        if not isinstance(execution_result, dict):
            return self._failure(
                reason="Execution result is not a valid dictionary."
            )

        if execution_result.get("status") != "error":
            return self._failure(
                reason="Execution result does not contain an error."
            )

        failed_step = self._extract_failed_step(
            execution_result=execution_result,
            execution_plan=execution_plan,
        )

        if failed_step is None:
            return self._failure(
                reason="Could not identify the failed execution step."
            )

        original_plan = {
            "analysis_plan": [
                deepcopy(failed_step)
            ]
        }

        validation_result = {
            "valid": False,
            "errors": [
                str(execution_result.get("error", "Unknown execution error"))
            ],
            "warnings": [],
        }

        repair_result = self.repairer.repair(
            plan=original_plan,
            validation_result=validation_result,
        )

        if not repair_result.get("repaired"):
            return self._failure(
                reason=(
                    "Execution error could not be repaired "
                    "deterministically."
                ),
                changes=repair_result.get("changes", []),
            )

        repaired_steps = repair_result.get(
            "plan",
            {},
        ).get(
            "analysis_plan",
            [],
        )

        if not repaired_steps:
            return self._failure(
                reason="Recovery produced an empty repaired plan.",
                changes=repair_result.get("changes", []),
            )

        repaired_full_plan = deepcopy(execution_plan)

        analysis_plan = repaired_full_plan.get("execution_steps")

        if not isinstance(analysis_plan, list):
            analysis_plan = repaired_full_plan.get("analysis_plan")

        if not isinstance(analysis_plan, list):
            return self._failure(
                reason=(
                    "Execution plan does not contain "
                    "a supported step list."
                ),
                changes=repair_result.get("changes", []),
            )

        replaced = self._replace_failed_step(
            steps=analysis_plan,
            failed_step=failed_step,
            repaired_step=repaired_steps[0],
        )

        if not replaced:
            return self._failure(
                reason="Failed execution step could not be replaced.",
                changes=repair_result.get("changes", []),
            )

        if "execution_steps" in repaired_full_plan:
            repaired_full_plan["execution_steps"] = analysis_plan
        else:
            repaired_full_plan["analysis_plan"] = analysis_plan

        return {
            "recoverable": True,
            "repaired_plan": repaired_full_plan,
            "changes": repair_result.get("changes", []),
            "reason": (
                "Execution failure was repaired "
                "and a retry plan was prepared."
            ),
        }

    # ------------------------------------------------------------------
    # Failed-step extraction
    # ------------------------------------------------------------------

    def _extract_failed_step(
        self,
        execution_result: dict[str, Any],
        execution_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Identify the step associated with an execution error.
        """

        failed_step = execution_result.get("step")

        if isinstance(failed_step, dict):
            return deepcopy(failed_step)

        failed_index = execution_result.get("step_index")

        steps = self._get_steps(execution_plan)

        if isinstance(failed_index, int):
            if 0 <= failed_index < len(steps):
                return deepcopy(steps[failed_index])

        operation = execution_result.get("operation")

        if operation:
            for step in steps:
                if (
                    isinstance(step, dict)
                    and step.get("operation") == operation
                ):
                    return deepcopy(step)

        return None

    def _get_steps(
        self,
        execution_plan: dict[str, Any],
    ) -> list:
        steps = execution_plan.get("execution_steps")

        if isinstance(steps, list):
            return steps

        steps = execution_plan.get("analysis_plan")

        if isinstance(steps, list):
            return steps

        return []

    # ------------------------------------------------------------------
    # Step replacement
    # ------------------------------------------------------------------

    def _replace_failed_step(
        self,
        steps: list,
        failed_step: dict[str, Any],
        repaired_step: dict[str, Any],
    ) -> bool:
        """
        Replace the failed step with its repaired version.
        """

        failed_step_number = failed_step.get("step")

        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            if (
                failed_step_number is not None
                and step.get("step") == failed_step_number
            ):
                steps[index] = repaired_step
                return True

        failed_operation = failed_step.get("operation")

        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            if (
                failed_operation
                and step.get("operation") == failed_operation
            ):
                steps[index] = repaired_step
                return True

        return False

    # ------------------------------------------------------------------
    # Failure helper
    # ------------------------------------------------------------------

    @staticmethod
    def _failure(
        reason: str,
        changes: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "recoverable": False,
            "repaired_plan": None,
            "changes": changes or [],
            "reason": reason,
        }


def recover_execution_error(
    execution_result: dict[str, Any],
    execution_plan: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function for execution-error recovery.
    """

    recovery = ExecutionRecovery(context=context)

    return recovery.recover(
        execution_result=execution_result,
        execution_plan=execution_plan,
    )
