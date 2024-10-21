from typing import List
from datetime import datetime, timedelta
from bisect import bisect_left
import numpy as np

from . import config
from .utils import check_parameters


@check_parameters
def slide_time_window(time: List[datetime]|np.ndarray, window_size: timedelta|int, step: timedelta|int =1, start_time: datetime|None =None) -> List[np.ndarray]:
    """
    Generate sliding time windows over a list of datetime objects.

    This function creates sliding windows of a specified size over a list of datetime objects. The windows can be defined by either a fixed number of elements (int) or a time duration (timedelta). The step size between windows can also be specified as either an integer or a timedelta.

    :param time: List of datetime objects to create windows from.
    :param window_size: Size of each window, specified as either an integer (number of elements) or a timedelta (duration).
    :param step: Step size between windows, specified as either an integer (number of elements) or a timedelta (duration). Default is 1.
    :param start_time: Optional start time for the windows. If not provided, the first element of the time list is used.
    
    :return: List of numpy arrays, each containing the indices of the elements in the corresponding window.
    """
        
    if config._ENABLE_VALUE_CHECKING:
        if type(window_size) != type(step):
            raise ValueError('window_size and step should have the same type (timedelta or int).')
        
    if isinstance(window_size, int):
        time_window_indices = [np.arange(i, i + window_size) for i in range(len(time) - window_size + 1, step)]
    elif isinstance(window_size, timedelta):
        if start_time is None:
            start_time = time[0]
        time_window_indices = [np.arange(bisect_left(time, start_time + i * step), bisect_left(time, start_time + i * step + window_size)) for i in range(int(((time[-1] - time[0]).total_seconds() - window_size.total_seconds()) / step.total_seconds() + 1))]
    else:
        raise ValueError('window_size and step should be either int or timedelta.')
        
    return time_window_indices
    