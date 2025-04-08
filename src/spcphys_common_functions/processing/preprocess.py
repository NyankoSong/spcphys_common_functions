from datetime import datetime
from typing import List, Tuple
import numpy as np
from astropy import units as u
import pandas as pd
import xarray

from ..utils.utils import check_parameters


def _check_condition(condition: str) -> float:
    """Convert string condition to float value.
    
    :param condition: String representation of a number or condition
    :type condition: str
    :return: Float value of the condition
    :rtype: float
    
    Handles special cases like 'inf', '-inf', and scientific notation like '1E3'.
    """
    
    if 'inf' in condition:
        return -np.inf if condition[0] == '-' else np.inf
    elif 'E' in condition:
        s = condition.split('E')
        return float(s[0]) * 10 ** float(s[1])
    else:
        return float(condition)


def _get_boundary(boundary: str|List[str|float]|Tuple[str|float]|None=None):
    """Process and normalize boundary conditions.
    
    :param boundary: Boundary conditions specification
    :type boundary: str, List[str|float], Tuple[str|float], or None, optional
    :return: Normalized boundary conditions as a list of two float values
    :rtype: List[float]
    
    If boundary is None, 'none', or incomplete, returns default boundary [-1E30, 1E30].
    Otherwise, converts string representations to float values.
    """
    
    if boundary is None or (isinstance(boundary, str) and boundary.lower() == 'none') or boundary[0].lower() == 'none' or len(boundary) < 2:
        boundary = [-1E30, 1E30]
    else:
        for i, b in enumerate(boundary):
            if not isinstance(b, float):
                boundary[i] = _check_condition(b)
                
    return boundary


def find_argnan(x: np.ndarray|u.Quantity, boundary: str|List[str|float]|Tuple[str|float]|None=None) -> np.ndarray:
    """Find the indices of NaN values and out-of-bound values in the input array.
    
    :param x: Input array
    :type x: numpy.ndarray or astropy.units.Quantity
    :param boundary: Boundary conditions for the input array, defaults to [-1E30, 1E30]
    :type boundary: str, List[str|float], Tuple[str|float], or None, optional
    :return: Indices of NaN values or out-of-bound values
    :rtype: numpy.ndarray
    """
    
    boundary = _get_boundary(boundary)
    
    return np.where((x < boundary[0]) | (x > boundary[1]))[0]


def process_nan(data: np.ndarray|u.Quantity, boundary: str|List[str|float]|Tuple[str|float]|None=None, time: List[datetime]|np.ndarray|None=None, method:str|None=None) -> List[np.ndarray]:
    """Process NaN values in the input data.
    
    :param data: Input data array
    :type data: numpy.ndarray or astropy.units.Quantity
    :param boundary: Boundary conditions for the input data array, defaults to [-1E30, 1E30]
    :type boundary: str, List[str|float], Tuple[str|float], or None, optional
    :param time: Array of datetime objects associated with the data, defaults to None
    :type time: List[datetime] or numpy.ndarray or None, optional
    :param method: Method for handling NaN values (e.g., 'drop', 'interpolate'), defaults to None
    :type method: str or None, optional
    :return: Processed data array with out-of-bound values replaced by NaN
    :rtype: numpy.ndarray or astropy.units.Quantity
    """
    
    boundary = _get_boundary(boundary)
    
    data[(data < boundary[0]) | (data > boundary[1])] = np.nan
    
    # if method is not None:
    #     if method == 'drop':
    #         df = pd.DataFrame(data, index=time).dropna()
    #         data = df.values
    #         time = list(df.index.to_pydatetime())
    #     elif method == 'interpolate':
    #         data = pd.DataFrame(data, index=time).interpolate().values

    return data


def npdt64_to_dt(npdt64: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64 array to datetime array.
    
    :param npdt64: Input numpy datetime64 array
    :type npdt64: numpy.ndarray
    :return: Converted datetime array
    :rtype: numpy.ndarray
    """
    
    return np.array([pd.to_datetime(date).to_pydatetime() for date in npdt64])


@check_parameters
def interpolate(x: datetime|List[datetime]|np.ndarray, xp: List[datetime]|np.ndarray, yp: np.ndarray|u.Quantity, vector_norm_interp: bool =False) -> np.ndarray|u.Quantity:
    """Interpolate values at specified datetime indices.
    
    :param x: Datetime or list of datetime objects to interpolate
    :type x: datetime or List[datetime] or numpy.ndarray
    :param xp: Datetime or list of datetime objects for interpolation
    :type xp: List[datetime] or numpy.ndarray
    :param yp: Values to be interpolated, can be a 1D or 2D array
    :type yp: numpy.ndarray or astropy.units.Quantity
    :param vector_norm_interp: Flag for vector norm interpolation, defaults to False
    :type vector_norm_interp: bool, optional
    :return: Interpolated values
    :rtype: numpy.ndarray or astropy.units.Quantity
    """
    
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(xp, list):
        xp = np.array(xp)
    if isinstance(x, datetime):
        x = np.array([x])
        
    if len(yp.shape) > 1 and yp.shape[1] > 1 and vector_norm_interp:
        yp_norm = np.linalg.norm(yp, axis=1, keepdims=True)
        y_interp_df = pd.DataFrame(np.concatenate((yp, yp_norm), axis=1), index=xp)
        y_interp_df = y_interp_df[~y_interp_df.index.duplicated(keep='first')].reindex(np.concatenate((x, xp))).sort_index().interpolate(method='time').loc[x, :]
        y_interp_df.loc[y_interp_df.index > xp[-1], :] = np.nan
        y = y_interp_df.loc[~y_interp_df.index.duplicated(keep='first')].values
        y = y[:, :-1] / np.linalg.norm(y[:, :-1], axis=1, keepdims=True) * y[:, -1][:, None]
        if isinstance(yp, u.Quantity):
            y = y * yp.unit
    else:
        if isinstance(yp, u.Quantity) and yp.unit.is_equivalent(u.deg):
            y_interp_df = pd.DataFrame(np.exp(1j * yp.to(u.rad).value), index=xp)
            y_interp_df = y_interp_df[~y_interp_df.index.duplicated(keep='first')].reindex(np.concatenate((x, xp))).sort_index().interpolate(method='time').loc[x, :]
            y_interp_df.loc[y_interp_df.index > xp[-1], :] = np.nan
            y = (np.angle(y_interp_df.loc[~y_interp_df.index.duplicated(keep='first')].values, deg=True) + 360) % 360 * u.deg
            y[y > 180 * u.deg] -= 360 * u.deg
            y = y.to(yp.unit)
        else:
            y_interp_df = pd.DataFrame(yp, index=xp)
            y_interp_df = y_interp_df[~y_interp_df.index.duplicated(keep='first')].reindex(np.concatenate((x, xp))).sort_index().interpolate(method='time').loc[x, :]
            y_interp_df.loc[y_interp_df.index > xp[-1], :] = np.nan
            y = y_interp_df.loc[~y_interp_df.index.duplicated(keep='first')].values
            if isinstance(yp, u.Quantity):
                y = y * yp.unit
            
    if len(yp.shape) == 1:
        y = y.flatten()
        
    return y