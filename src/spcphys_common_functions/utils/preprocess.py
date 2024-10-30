

from datetime import datetime
from typing import List, Tuple
import numpy as np
from astropy import units as u
import pandas as pd
import xarray

from . import config
from .utils import check_parameters


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


# def _npdt64_to_dt(npdt64: np.ndarray) -> List[datetime]:
#     '''
#     Convert numpy datetime64 to datetime.
#     '''
    
#     return [pd.to_datetime(date).to_pydatetime() for date in npdt64]


def _npdt64_to_dt(npdt64: np.ndarray) -> np.ndarray:
    '''
    Convert numpy datetime64 to datetime.
    '''
    
    return np.array([pd.to_datetime(date).to_pydatetime() for date in npdt64])