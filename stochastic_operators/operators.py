"""
stochastic_operators.operators
==============================

Core finite-dimensional stochastic operator abstractions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from operator_core import (
    LinearOperator,
    NonSquareOperatorError,
    OperatorError,
    readonly_array,
    require_nonnegative_integer,
)


class StochasticConvention(str, Enum):
    """
    Supported stochastic-matrix conventions.
    """

    ROW = "row"
    COLUMN = "column"


_CONVENTION_ALIASES = {
    StochasticConvention.ROW: StochasticConvention.ROW,
    "row": StochasticConvention.ROW,
    "rows": StochasticConvention.ROW,
    "row_stochastic": StochasticConvention.ROW,

    StochasticConvention.COLUMN: StochasticConvention.COLUMN,
    "column": StochasticConvention.COLUMN,
    "columns": StochasticConvention.COLUMN,
    "col": StochasticConvention.COLUMN,
    "column_stochastic": StochasticConvention.COLUMN,
}


class StochasticOperator(LinearOperator):
    """
    Finite-dimensional stochastic operator.

    Parameters
    ----------
    matrix
        Square matrix with nonnegative entries.

    convention
        Whether rows or columns sum to one.

    tol
        Numerical tolerance used when validating nonnegativity and
        stochastic normalization.

    name
        Human-readable operator name.

    metadata
        Optional user metadata.
    """

    def __init__(
        self,
        matrix,
        *,
        convention: StochasticConvention | str = StochasticConvention.ROW,
        tol: float = 1e-10,
        name: str = "StochasticOperator",
        metadata: dict | None = None,
    ):
        try:
            convention = _CONVENTION_ALIASES[convention]
        except KeyError as exc:
            raise OperatorError(
                "convention must be 'row' or 'column'."
            ) from exc

        if isinstance(tol, bool) or not np.isscalar(tol):
            raise OperatorError(
                "tol must be a nonnegative scalar."
            )

        tol = float(tol)

        if not np.isfinite(tol) or tol < 0:
            raise OperatorError(
                "tol must be finite and nonnegative."
            )

        array = np.asarray(matrix)

        if array.ndim != 2:
            raise OperatorError(
                "stochastic matrix must be two-dimensional."
            )

        if array.shape[0] != array.shape[1]:
            raise NonSquareOperatorError(
                "stochastic operators must be square."
            )

        if np.iscomplexobj(array):
            raise OperatorError(
                "stochastic operators must be real-valued."
            )

        array = np.array(
            array,
            dtype=float,
            copy=True,
        )

        if not np.all(np.isfinite(array)):
            raise OperatorError(
                "stochastic matrix entries must be finite."
            )

        if np.any(array < -tol):
            raise OperatorError(
                "stochastic matrix entries must be nonnegative."
            )

        # Remove tiny negative values caused by floating-point error.
        array[np.abs(array) <= tol] = 0.0

        axis = (
            1
            if convention is StochasticConvention.ROW
            else 0
        )

        sums = np.sum(array, axis=axis)

        if not np.allclose(
            sums,
            1.0,
            atol=tol,
            rtol=tol,
        ):
            direction = (
                "rows"
                if convention is StochasticConvention.ROW
                else "columns"
            )

            raise OperatorError(
                f"stochastic matrix {direction} must sum to one."
            )

        operator_metadata = {
            "operator": "stochastic",
            "convention": convention.value,
            "tolerance": tol,
        }

        if metadata is not None:
            if not isinstance(metadata, dict):
                raise OperatorError(
                    "metadata must be a dictionary."
                )

            operator_metadata.update(metadata)

        super().__init__(
            matrix=array,
            name=name,
            metadata=operator_metadata,
        )

        object.__setattr__(
            self,
            "convention",
            convention,
        )
        object.__setattr__(
            self,
            "tol",
            tol,
        )

    @property
    def dimension(self) -> int:
        """
        Return the number of stochastic states.
        """

        return self.rows

    @property
    def is_row_stochastic(self) -> bool:
        """
        Return whether the operator uses row normalization.
        """

        return (
            self.convention
            is StochasticConvention.ROW
        )

    @property
    def is_column_stochastic(self) -> bool:
        """
        Return whether the operator uses column normalization.
        """

        return (
            self.convention
            is StochasticConvention.COLUMN
        )

    def apply_distribution(
        self,
        distribution,
    ) -> np.ndarray:
        """
        Apply one stochastic transition to a probability distribution.
        """

        vector = np.asarray(
            distribution,
            dtype=float,
        )

        if vector.ndim != 1:
            raise OperatorError(
                "distribution must be one-dimensional."
            )

        if len(vector) != self.dimension:
            raise OperatorError(
                "distribution dimension does not match operator."
            )

        if not np.all(np.isfinite(vector)):
            raise OperatorError(
                "distribution values must be finite."
            )

        if np.any(vector < -self.tol):
            raise OperatorError(
                "distribution values must be nonnegative."
            )

        if not np.isclose(
            np.sum(vector),
            1.0,
            atol=self.tol,
            rtol=self.tol,
        ):
            raise OperatorError(
                "distribution must sum to one."
            )

        if self.is_row_stochastic:
            result = vector @ self.matrix
        else:
            result = self.matrix @ vector

        return readonly_array(
            result,
            name="propagated distribution",
            ndim=1,
        )

    def transition_power(
        self,
        steps: int,
    ) -> "StochasticOperator":
        """
        Return the stochastic operator after a given number of steps.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        powered = np.linalg.matrix_power(
            self.matrix,
            steps,
        )

        return StochasticOperator(
            powered,
            convention=self.convention,
            tol=self.tol,
            name=f"{self.name}^{steps}",
            metadata={
                **self.metadata,
                "operation": "transition_power",
                "steps": steps,
            },
        )
