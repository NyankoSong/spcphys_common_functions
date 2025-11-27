"""
Processing module for space physics data.

This module provides tools for processing spacecraft data, including
CDF file handling, data preprocessing, time series analysis, and
coordinate transformations.

Submodules
----------
backmapping
    Ballistic backmapping for solar wind source identification
cdf_process
    CDF file reading and processing utilities
net_tools
    Network tools for fetching data from online sources (CDAWeb)
preprocess
    Data preprocessing (NaN handling, interpolation, etc.)
plot_tools
    Plotting utilities for time series and distributions
time_window
    Sliding time window analysis tools
vdf_process
    Velocity Distribution Function processing
vec_cart_sph
    Coordinate transformations (Cartesian <-> Spherical, quaternions)
"""

from . import backmapping
from . import cdf_process
from . import net_tools
from . import preprocess
from . import plot_tools
from . import time_window
from . import vdf_process
from . import vec_cart_sph

__all__ = [
    "backmapping",
    "cdf_process",
    "net_tools",
    "preprocess",
    "plot_tools",
    "time_window",
    "vdf_process",
    "vec_cart_sph",
]