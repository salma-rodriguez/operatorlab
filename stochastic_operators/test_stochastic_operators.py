import numpy as np
import pytest

from operator_core import (
    NonSquareOperatorError,
    OperatorError,
)
from stochastic_operators import (
    StochasticConvention,
    StochasticOperator,
)


def test_row_stochastic_operator():
    operator = StochasticOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    assert operator.is_row_stochastic
    assert not operator.is_column_stochastic
    assert operator.dimension == 2


def test_column_stochastic_operator():
    operator = StochasticOperator(
        [
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        convention="column",
    )

    assert operator.convention is StochasticConvention.COLUMN


def test_non_square_matrix_rejected():
    with pytest.raises(NonSquareOperatorError):
        StochasticOperator(
            np.ones((2, 3))
        )


def test_negative_entries_rejected():
    with pytest.raises(OperatorError):
        StochasticOperator([
            [1.1, -0.1],
            [0.0, 1.0],
        ])


def test_invalid_row_sums_rejected():
    with pytest.raises(OperatorError):
        StochasticOperator([
            [0.8, 0.1],
            [0.2, 0.7],
        ])


def test_apply_row_distribution():
    operator = StochasticOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    result = operator.apply_distribution(
        [1.0, 0.0]
    )

    assert np.allclose(
        result,
        [0.8, 0.2],
    )


def test_apply_column_distribution():
    operator = StochasticOperator(
        [
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        convention="column",
    )

    result = operator.apply_distribution(
        [1.0, 0.0]
    )

    assert np.allclose(
        result,
        [0.8, 0.2],
    )


def test_transition_power_zero_is_identity():
    operator = StochasticOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    identity = operator.transition_power(0)

    assert np.allclose(
        identity.matrix,
        np.eye(2),
    )
