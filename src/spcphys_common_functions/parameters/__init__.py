"""
Parameters module for space physics calculations.

This module provides functions for calculating various physical parameters
commonly used in space physics and heliophysics research.

Submodules
----------
alfvenic_parameters
    Alfvénic parameter calculations (cross-helicity, residual energy, etc.)
plasma_beta
    Plasma beta and pressure calculations
vth_E_T
    Thermal velocity, energy, and temperature conversions
coulomb_collision
    Coulomb collisional age calculations
effectsize
    Statistical effect size calculations
minimum_variance
    Minimum Variance Analysis (MVA)
"""

import sys

from . import alfvenic_parameters
from . import plasma_beta
from . import vth_E_T
from . import coulomb_collision
from . import effectsize
from . import minimum_variance

# Backward compatibility: keep old submodule import path working.
sys.modules[__name__ + ".coulomb_collisional_age"] = coulomb_collision
coulomb_collisional_age = coulomb_collision

__all__ = [
    "alfvenic_parameters",
    "plasma_beta",
    "vth_E_T",
    "coulomb_collision",
    "coulomb_collisional_age",
    "effectsize",
    "minimum_variance",
]