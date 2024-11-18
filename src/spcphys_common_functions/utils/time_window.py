from typing import List, Tuple
import warnings
from datetime import datetime, timedelta
from bisect import bisect_left
import numpy as np

from . import config
from .utils import check_parameters


def _time_indices(time: datetime, time_range: List[datetime]|Tuple[datetime]) -> np.ndarray:
    return np.arange(bisect_left(time, time_range[0]), bisect_left(time, time_range[1]))


@check_parameters
def slide_time_window(time: List[datetime]|np.ndarray, window_size: timedelta|int|None =None, step: timedelta|int|None =None, start_time: datetime|None =None, align_to: List[datetime]|np.ndarray =None) -> Tuple[List[Tuple[datetime, datetime]], List[np.ndarray]]:
    """
    Generate sliding time windows over a list of datetime objects.

    This function creates sliding windows of a specified size over a list of datetime objects. The windows can be defined by either a fixed number of elements (int) or a time duration (timedelta). The step size between windows can also be specified as either an integer or a timedelta.

    :param time: List of datetime objects to create windows from.
    :param window_size: Size of each window, specified as either an integer (number of elements) or a timedelta (duration).
    :param step: Step size between windows, specified as either an integer (number of elements) or a timedelta (duration)..
    :param start_time: Optional start time for the windows. If not provided, the first element of the time list is used.
    
    :return time_window_ranges: List of tuples representing the start and end times of each window.
    :return time_window_indices: List of numpy arrays containing the indices of the elements in each window.
    """
    
    if align_to is None:
        if type(window_size) != type(step) and window_size is not None:
            raise ValueError('window_size and step should have the same type (timedelta or int).')
    else:
        if len(align_to) > len(time):
            warnings.warn('The length of align_to is greater than the length of time.')
            if window_size is not None and  not isinstance(window_size, timedelta):
                raise ValueError('window_size should be a timedelta when align_to is provided.')
    
    if align_to is None:
        if isinstance(window_size, int):
            time_window_indices = [np.arange(i, i + window_size) for i in range(len(time) - window_size + 1, step)]
            time_window_ranges = [(time[i], time[i + window_size - 1]) for i in range(len(time) - window_size + 1, step)]
        elif isinstance(window_size, timedelta):
            if start_time is None:
                start_time = time[0]
            time_window_ranges = [(start_time + i * step, start_time + i * step + window_size) for i in range(int(((time[-1] - time[0]).total_seconds() - window_size.total_seconds()) / step.total_seconds() + 1))]
            time_window_indices = [_time_indices(time, time_window_range) for time_window_range in time_window_ranges]
        else:
            raise ValueError('window_size and step should be either int or timedelta.')
    else:
        if window_size is None:
            time_window_ranges = [(align_to[i], align_to[i+1]) if i < len(align_to) - 1 else (align_to[i], align_to[i] + (align_to[i] - align_to[i-1])) for i in range(len(align_to))]
        else:
            time_window_ranges = [(align_to[i], align_to[i] + window_size) for i in range(len(align_to))]
            
        time_window_indices = [_time_indices(time, time_window_range) for time_window_range in time_window_ranges]
        
    return time_window_ranges, time_window_indices
    