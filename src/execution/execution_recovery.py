from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.agent.plan_repair import PlanRepair


class ExecutionRecovery:
    """
    Recovery layer for failed operation-executor steps.

    The recovery layer:
    1. Detects failed steps from OperationExecutor output.
    2. Uses deterministic PlanRepair logic.
    3. Produces a repaired execution plan.
    4. Preserves successful steps.
    5. Does not execute operations itself.
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
        Recover failed execution steps.

        Expected OperationExecutor result:

        {
            "status": "success" | "partial" | "failed",
            "total_steps": int,
            "successful_steps": int,
            "failed_steps": int,
            "results": [
                {
                    "step": int,
                    "operation": str,
                    "status": "success" | "error",
                    "error": str
                }
            ]
        }
        """

        if not isinstance(execution_result, dict):
            return self._failure(
                "Execution result is not a valid dictionary."
            )

        execution_status = execution_result.get("status")

        if execution_status == "success":
            return self._failure(
                "Execution completed successfully. Recovery is not required."
            )

        results = execution_result.get("results")

        if not isinstance(results, list):
            return self._failure(
                "Execution result does not contain a valid results list."
            )

        failed_results = [
            item
            for item in results
            if isinstance(item, dict)
            and item.get("status") == "error"
        ]

        if not failed_results:
            return self._failure(
                "No failed execution steps were found."
            )

        steps = self._get_steps(execution_plan)

        if not steps:
            return self._failure(
                "Execution plan does not contain execution steps."
            )

        repaired_steps = deepcopy(steps)
        all_changes: list[str] = []
        recovery_failures: list[str] = []

        for failed_result in failed_results:

            failed_step = self._find_step(
                steps=steps,
                failed_result=failed_result,
            )

            if failed_step is None:
                recovery_failures.append(
                    (
                        "Could not identify execution-plan step "
                        f"for failed step {failed_result.get('step')}."
                    )
                )
                continue

            repair_result = self._repair_failed_step(
                failed_step=failed_step,
                failed_result=failed_result,
            )

            if not repair_result["repaired"]:
                recovery_failures.append(
                    repair_result["reason"]
                )
                all_changes.extend(
                    repair_result.get("changes", [])
                )
                continue

            repaired_step = repair_result["step"]

            replaced = self._replace_step(
                steps=repaired_steps,
                original_step=failed_step,
                repaired_step=repaired_step,
            )

            if not replaced:
                recovery_failures.append(
                    (
                        "Failed to replace repaired step "
                        f"{failed_step.get('step')}."
                    )
                )
                continue

            all_changes.extend(
                repair_result.get("changes", [])
            )

        if recovery_failures:
            return {
                "recoverable": False,
                "repaired_plan": None,
                "changes": all_changes,
                "reason": (
                    "One or more failed execution steps "
                    "could not be recovered."
                ),
                "failures": recovery_failures,
            }

        repaired_plan = self._build_repaired_plan(
            execution_plan=execution_plan,
            repaired_steps=repaired_steps,
        )

        return {
            "recoverable": True,
            "repaired_plan": repaired_plan,
            "changes": all_changes,
            "reason": (
                "Execution failures were repaired "
                "and a retry plan was prepared."
            ),
            "failures": [],
        }

    # ==============================================================
    # Failed step repair
    # ==============================================================

    def _repair_failed_step(
        self,
        failed_step: dict[str, Any],
        failed_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Repair one failed execution step.
        """

        candidate_plan = {
            "analysis_plan": [
                deepcopy(failed_step)
            ]
        }

        validation_result = {
            "valid": False,
            "errors": [
                str(
                    failed_result.get(
                        "error",
                        "Unknown execution error.",
                    )
                )
            ],
            "warnings": [],
        }

        try:
            repair_result = self.repairer.repair(
                plan=candidate_plan,
                validation_result=validation_result,
            )

        except Exception as exc:
            return {
                "repaired": False,
                "step": None,
                "changes": [],
                "reason": (
                    "Plan repair raised an exception: "
                    f"{str(exc)}"
                ),
            }

        if not repair_result.get("repaired"):
            return {
                "repaired": False,
                "step": None,
                "changes": repair_result.get(
                    "changes",
                    [],
                ),
                "reason": (
                    "Deterministic repair could not "
                    "repair failed step "
                    f"{failed_step.get('step')}."
                ),
            }

        repaired_plan = repair_result.get(
            "plan",
            {},
        )

        repaired_analysis_plan = repaired_plan.get(
            "analysis_plan",
            [],
        )

        if not repaired_analysis_plan:
            return {
                "repaired": False,
                "step": None,
                "changes": repair_result.get(
                    "changes",
                    [],
                ),
                "reason": (
                    "Repair produced an empty "
                    "analysis plan."
                ),
            }

        return {
            "repaired": True,
            "step": deepcopy(
                repaired_analysis_plan[0]
            ),
            "changes": repair_result.get(
                "changes",
                [],
            ),
            "reason": (
                "Failed step repaired successfully."
            ),
        }

    # ==============================================================
    # Step lookup
    # ==============================================================

    def _find_step(
        self,
        steps: list,
        failed_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Match an executor failure to its original plan step.
        """

        failed_step_number = failed_result.get(
            "step"
        )

        if failed_step_number is not None:

            for step in steps:

                if not isinstance(step, dict):
                    continue

                if step.get("step") == failed_step_number:
                    return deepcopy(step)

        failed_operation = failed_result.get(
            "operation"
        )

        if failed_operation:

            for step in steps:

                if not isinstance(step, dict):
                    continue

                if (
                    step.get("operation")
                    == failed_operation
                ):
                    return deepcopy(step)

        return None

    # ==============================================================
    # Step replacement
    # ==============================================================

    def _replace_step(
        self,
        steps: list,
        original_step: dict[str, Any],
        repaired_step: dict[str, Any],
    ) -> bool:
        """
        Replace the original failed step with repaired step.
        """

        original_step_number = original_step.get(
            "step"
        )

        if original_step_number is not None:

            for index, step in enumerate(steps):

                if not isinstance(step, dict):
                    continue

                if (
                    step.get("step")
                    == original_step_number
                ):
                    steps[index] = repaired_step
                    return True

        original_operation = original_step.get(
            "operation"
        )

        if original_operation:

            for index, step in enumerate(steps):

                if not isinstance(step, dict):
                    continue

                if (
                    step.get("operation")
                    == original_operation
                ):
                    steps[index] = repaired_step
                    return True

        return False

    # ==============================================================
    # Plan helpers
    # ==============================================================

    def _get_steps(
        self,
        execution_plan: dict[str, Any],
    ) -> list:
        """
        Support both adapted and raw plan formats.
        """

        execution_steps = execution_plan.get(
            "execution_steps"
        )

        if isinstance(execution_steps, list):
            return execution_steps

        analysis_plan = execution_plan.get(
            "analysis_plan"
        )

        if isinstance(analysis_plan, list):
            return analysis_plan

        return []

    def _build_repaired_plan(
        self,
        execution_plan: dict[str, Any],
        repaired_steps: list,
    ) -> dict[str, Any]:
        """
        Preserve the original plan format.
        """

        repaired_plan = deepcopy(
            execution_plan
        )

        if isinstance(
            repaired_plan.get("execution_steps"),
            list,
        ):
            repaired_plan[
                "execution_steps"
            ] = repaired_steps

        elif isinstance(
            repaired_plan.get("analysis_plan"),
            list,
        ):
            repaired_plan[
                "analysis_plan"
            ] = repaired_steps

        else:
            repaired_plan[
                "execution_steps"
            ] = repaired_steps

        return repaired_plan

    # ==============================================================
    # Failure helper
    # ==============================================================

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
            "failures": [],
        }


def recover_execution_error(
    execution_result: dict[str, Any],
    execution_plan: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function for execution-error recovery.
    """

    recovery = ExecutionRecovery(
        context=context
    )

    return recovery.recover(
        execution_result=execution_result,
        execution_plan=execution_plan,
    )
