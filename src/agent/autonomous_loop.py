from typing import Any, Dict


class AutonomousLoopController:
    """
    Controls the bounded autonomous analysis loop.

    Responsibilities:
    - Limit autonomous iterations.
    - Prevent repeated identical operations.
    - Keep loop state available for pipeline diagnostics.

    This class does NOT execute operations.
    """

    def __init__(
        self,
        max_iterations: int = 3,
    ):
        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        self.max_iterations = max_iterations
        self.iteration = 0
        self._operation_history: list[str] = []

    def start(self) -> None:
        """
        Reset controller state for a new analysis request.
        """
        self.iteration = 0
        self._operation_history = []

    def can_continue(self) -> bool:
        """
        Return True when another autonomous iteration is allowed.
        """
        return self.iteration < self.max_iterations

    def next_iteration(self) -> int:
        """
        Advance to the next autonomous iteration.
        """
        if not self.can_continue():
            raise RuntimeError(
                "Maximum autonomous iterations reached."
            )

        self.iteration += 1

        return self.iteration

    def should_continue(
        self,
        decision: Dict[str, Any],
    ) -> bool:
        """
        Determine whether the interpreter requested another
        autonomous iteration.
        """

        if not isinstance(decision, dict):
            return False

        decision_type = str(
            decision.get("decision", "")
        ).strip().lower()

        if decision_type != "continue":
            return False

        if not self.can_continue():
            return False

        next_operations = decision.get(
            "next_operations",
            [],
        )

        if not isinstance(
            next_operations,
            list,
        ):
            return False

        if not next_operations:
            return False

        return True

    def register_operations(
        self,
        operations: list[dict[str, Any]],
    ) -> None:
        """
        Record operations requested by the interpreter.
        """

        if not isinstance(
            operations,
            list,
        ):
            return

        for operation in operations:

            if not isinstance(
                operation,
                dict,
            ):
                continue

            operation_name = operation.get(
                "operation"
            )

            if not operation_name:
                continue

            signature = self._operation_signature(
                operation
            )

            self._operation_history.append(
                signature
            )

    def has_seen_operations(
        self,
        operations: list[dict[str, Any]],
    ) -> bool:
        """
        Return True when all requested operations have
        already been requested previously.
        """

        if not operations:
            return False

        valid_signatures = []

        for operation in operations:

            if not isinstance(
                operation,
                dict,
            ):
                continue

            if not operation.get("operation"):
                continue

            valid_signatures.append(
                self._operation_signature(
                    operation
                )
            )

        if not valid_signatures:
            return False

        return all(
            signature in self._operation_history
            for signature in valid_signatures
        )

    def _operation_signature(
        self,
        operation: dict[str, Any],
    ) -> str:
        """
        Create a stable signature for an operation.
        """

        operation_name = str(
            operation.get(
                "operation",
                "",
            )
        ).strip().lower()

        parameters = operation.get(
            "parameters",
            {},
        )

        if not isinstance(
            parameters,
            dict,
        ):
            parameters = {}

        parameter_items = sorted(
            (
                str(key),
                repr(value),
            )
            for key, value in parameters.items()
        )

        return (
            operation_name
            + "|"
            + repr(parameter_items)
        )

    def status(self) -> Dict[str, Any]:
        """
        Return current loop state.
        """

        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "can_continue": self.can_continue(),
            "operation_history": list(
                self._operation_history
            ),
        }
