

from datetime import datetime
from typing import List, Tuple
import numpy as np
from astropy import units as u
import pandas as pd
import xarray

from ..utils.utils import check_parameters


def _check_condition(condition: str) -> float:
    
    if 'inf' in condition:
        return -np.inf if condition[0] == '-' else np.inf
    elif 'E' in condition:
        s = condition.split('E')
        return float(s[0]) * 10 ** float(s[1])
    else:
        return float(condition)


def _get_boundary(boundary: str|List[str|float]|Tuple[str|float]|None=None):
    
    if boundary is None or (isinstance(boundary, str) and boundary.lower == 'none') or boundary[0].lower() == 'none':
        boundary = [-1E30, 1E30]
    else:
        for i, b in enumerate(boundary):
            if not isinstance(b, float):
                boundary[i] = _check_condition(b)
                
    return boundary


def find_argnan(x: np.ndarray|u.Quantity, boundary: str|List[str|float]|Tuple[str|float]|None=None) -> np.ndarray:
    '''
    Find the indices of NaN values in the input array.
    
    :param x: Input array.
    :param boundary: Boundary conditions for the input array. Default is [-1E30, 1E30].
    
    :return: Indices of NaN values.
    '''
    
    boundary = _get_boundary(boundary)
    
    return np.where((x < boundary[0]) | (x > boundary[1]))[0]


def process_nan(data: np.ndarray|u.Quantity, boundary: str|List[str|float]|Tuple[str|float]|None=None, time: List[datetime]|np.ndarray|None=None, method:str|None=None) -> List[np.ndarray]:
    '''
    Process NaN values in the input data.
    
    :param data: List of input data arrays.
    :param boundary: Boundary conditions for the input data arrays. Default is [-1E30, 1E30].
    
    :return: List of processed data arrays.
    '''
    
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
    '''
    Convert numpy datetime64 to datetime.
    '''
    
    return np.array([pd.to_datetime(date).to_pydatetime() for date in npdt64])
    
    
# @check_parameters
# def multi_dimensional_interpolate(x: List[datetime]|np.ndarray|u.Quantity, xp: List[datetime]|np.ndarray|u.Quantity, yp: np.ndarray|u.Quantity, **interp_kwargs) -> np.ndarray|u.Quantity:
#     '''
#     Perform multi-dimensional interpolation on the given data.

#     :param x: Array of x-coordinates at which to evaluate the interpolated values.
#     :param xp: Array of x-coordinates of the data points.
#     :param yp: Array of y-coordinates of the data points. Must be a Quantity with the same number of rows as xp.
#     :param interp_kwargs: Additional keyword arguments to pass to the numpy.interp function.
    
#     :return y: Interpolated values at the x, with the same number of columns as yp.
#     '''
    
#     if yp.shape[0] != xp.shape[0]:
#         raise ValueError(f"xp and yp must have the same number of rows. (But got xp.shape={xp.shape} and yp.shape={yp.shape})")
    
#     if isinstance(x[0], datetime):
#         x = np.array([i.timestamp() for i in x])
#     if isinstance(xp[0], datetime):
#         xp = np.array([i.timestamp() for i in xp])
        
#     y = np.zeros((x.shape[0], yp.shape[1]))
#     for i, col in enumerate(yp.T):
#         y[:, i] = np.interp(x, xp, col, **interp_kwargs)
        
#     if isinstance(yp, u.Quantity):
#         return y * yp.unit
#     else:
#         return y


@check_parameters
def interpolate(x: List[datetime]|np.ndarray, xp: List[datetime]|np.ndarray, yp: np.ndarray|u.Quantity, vector_norm_interp: bool =False) -> np.ndarray|u.Quantity:
    '''
    Perform interpolation on the given time series data.
    
    :param x: Array of datetime at which to evaluate the interpolated values.
    :param xp: Array of datetime of the data points.
    :param yp: Array of y-coordinates of the data points in shape (t,) or (t, dim).
    :param vector_norm_interp: Whether to interpolate the vector norm of the data points if yp is a vector time series. Default is False.
    
    :return y: Interpolated values at the x.
    '''
    
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(xp, list):
        xp = np.array(xp)
        
    if len(yp.shape) > 1 and yp.shape[1] > 1 and vector_norm_interp:
        yp_norm = np.linalg.norm(yp, axis=1, keepdims=True)
        y_interp_df = pd.DataFrame(np.concatenate((yp, yp_norm), axis=1), index=xp).reindex(np.concatenate((x, xp))).sort_index().interpolate(method='time').loc[x, :]
        y_interp_df.loc[y_interp_df.index > xp[-1], :] = np.nan
        y = y_interp_df.loc[~y_interp_df.index.duplicated(keep='first')].values
        y = y[:, :-1] / np.linalg.norm(y[:, :-1], axis=1, keepdims=True) * y[:, -1][:, None]
    else:
        y_interp_df = pd.DataFrame(yp, index=xp).reindex(np.concatenate((x, xp))).sort_index().interpolate(method='time').loc[x, :]
        y_interp_df.loc[y_interp_df.index > xp[-1], :] = np.nan
        y = y_interp_df.loc[~y_interp_df.index.duplicated(keep='first')].values
            
    if isinstance(yp, u.Quantity):
        y = y * yp.unit
    if len(yp.shape) == 1:
        y = y.flatten()
        
    return y
        
    