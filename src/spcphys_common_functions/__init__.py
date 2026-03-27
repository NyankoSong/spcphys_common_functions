"""
spcphys-common-functions: A Python toolkit for space physics research.

This package provides commonly used functions for space physics research,
including Alfvénic parameter calculations, plasma analysis, data processing,
and visualization tools for heliophysics and solar wind studies.

Modules
-------
parameters
    Functions for calculating physical parameters (Alfvén parameters, plasma beta, etc.)
processing
    Data processing tools (backmapping, CDF processing, time windows, etc.)
utils
    Utility functions for common operations

Example
-------
>>> from spcphys_common_functions.parameters import alfvenic_parameters
>>> from spcphys_common_functions.processing import backmapping

"""

__version__ = "0.1.0"
__author__ = "Shuyi Meng (NyankoSong)"
__email__ = "nyankosong@gmail.com"
__license__ = "MIT"

from .parameters import (
    alfvenic_parameters,
    plasma_beta,
    vth_E_T,
    coulomb_collision,
    coulomb_collisional_age,
    effectsize,
    minimum_variance,
)
from .processing import (
    backmapping,
    cdf_process,
    net_tools,
    preprocess,
    plot_tools,
    time_window,
    vdf_process,
    vec_cart_sph,
)
from .utils import utils

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Parameters modules
    "alfvenic_parameters",
    "plasma_beta",
    "vth_E_T",
    "coulomb_collision",
    "coulomb_collisional_age",
    "effectsize",
    "minimum_variance",
    # Processing modules
    "backmapping",
    "cdf_process",
    "net_tools",
    "preprocess",
    "plot_tools",
    "time_window",
    "vdf_process",
    "vec_cart_sph",
    # Utils
    "utils",
]

# Runtime type checking with beartype
from beartype.claw import beartype_this_package
beartype_this_package()

