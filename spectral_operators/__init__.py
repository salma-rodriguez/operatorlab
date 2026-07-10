"""
rh_operator
===========

Research library for operator-theoretic investigations of the
Riemann Hypothesis and related spectral problems.

This package provides reusable mathematical objects for
constructing, composing, and analyzing linear operators,
including finite-difference, weighted, graded, adelic,
Hamiltonian, and spectral operators.

The package is intentionally independent of any specific
numerical experiment so that operators may be reused in
different research contexts.

Author:
    Salma Rodriguez

License:
    Research / Development

"""

__version__ = "0.1.0"

__author__ = "Salma Rodriguez"

__all__ = [
    "LinearOperator",
    "OperatorFactory",

    "FiniteDifferenceOperator",
    "WeightOperator",
    "GradedOperator",
    "AdelicOperator",
    "Hamiltonian",

    "SpectralAnalyzer",
    "GeometryAnalyzer",
    "EvolutionOperator",

    "ZetaCorrespondence",
]

# ---------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------

from .algebra import (
    LinearOperator,
    Field,
    Norm,
    OperatorFactory
)

"""
# ---------------------------------------------------------------------
# Operator Construction
# ---------------------------------------------------------------------

from .operators import (
    FiniteDifferenceOperator,
    WeightOperator,
    GradedOperator,
    AdelicOperator,
    Hamiltonian,
)

# ---------------------------------------------------------------------
# Spectral Theory
# ---------------------------------------------------------------------

from .spectrum import (
    SpectralAnalyzer,
)

# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

from .geometry import (
    GeometryAnalyzer,
)

# ---------------------------------------------------------------------
# Time Evolution
# ---------------------------------------------------------------------

from .evolution import (
    EvolutionOperator,
)

# ---------------------------------------------------------------------
# Zeta Correspondence
# ---------------------------------------------------------------------

from .zeta import (
    ZetaCorrespondence,
)
"""
