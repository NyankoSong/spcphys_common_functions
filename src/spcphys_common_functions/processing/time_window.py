from typing import List, Tuple
import warnings
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
import numpy as np




def _time_indices(time: List[datetime] | np.ndarray, time_range: List[datetime]|Tuple[datetime]) -> np.ndarray:
    """Find indices of datetime objects within a specified time range.

    :param time: List of datetime objects to search in
    :type time: List[datetime] or numpy.ndarray
    :param time_range: Time range specified as [start_time, end_time]
    :type time_range: List[datetime] or Tuple[datetime]
    :return: Array of indices for datetime objects that fall within the specified time range
    :rtype: numpy.ndarray
    """
    return np.arange(bisect_left(time, time_range[0]), bisect_left(time, time_range[1]))



def slide_time_window(time: List[datetime]|np.ndarray, window_size: timedelta|int =None, step: timedelta|int =None, start_time: datetime|None =None, end_time: datetime|None =None, align_to: List[datetime]|np.ndarray =None) -> Tuple[List[Tuple[datetime, datetime]], List[np.ndarray]]:
    """Generate sliding time windows over a list of datetime objects.

    :param time: List of datetime objects to create windows from
    :type time: List[datetime] or numpy.ndarray
    :param window_size: Size of each window, specified as either an integer (number of elements) or a timedelta (duration), defaults to None
    :type window_size: timedelta or int, optional
    :param step: Step size between windows, specified as either an integer (number of elements) or a timedelta (duration), defaults to None
    :type step: timedelta or int, optional
    :param start_time: Optional start time for the windows, defaults to None
    :type start_time: datetime or None, optional
    :param end_time: Optional end time for the windows, defaults to None
    :type end_time: datetime or None, optional
    :param align_to: List of datetime objects to align the windows to, defaults to None
    :type align_to: List[datetime] or numpy.ndarray, optional
    :return: Tuple containing time_window_ranges (start and end times of each window) and time_window_indices (indices of elements in each window)
    :rtype: Tuple[List[Tuple[datetime, datetime]], List[numpy.ndarray]]
    
    This function creates sliding windows of a specified size over a list of datetime objects. 
    The windows can be defined by either a fixed number of elements (int) or a time duration 
    (timedelta). The step size between windows can also be specified as either an integer or 
    a timedelta.
    """
    
    if window_size is None and align_to is None:
        raise ValueError('Either window_size or align_to should be provided.')
    
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
        else:
            if start_time is None:
                start_time = time[0]
            time_window_ranges = [(start_time + i * step, start_time + i * step + window_size) for i in range(int(((time[-1] - time[0]).total_seconds() - window_size.total_seconds()) / step.total_seconds() + 1))]
            time_window_indices = [_time_indices(time, time_window_range) for time_window_range in time_window_ranges]
    else:
        if window_size is None:
            time_window_ranges = [(align_to[i], align_to[i+1]) if i < len(align_to) - 1 else (align_to[i], align_to[i] + (align_to[i] - align_to[i-1])) for i in range(len(align_to))]
        else:
            time_window_ranges = [(align_to[i], align_to[i] + window_size) for i in range(len(align_to))]
            
        time_window_indices = [_time_indices(time, time_window_range) for time_window_range in time_window_ranges]
        
    start_index = bisect_left([time_window_range[0] for time_window_range in time_window_ranges], start_time) if start_time is not None else 0
    end_index = bisect_right([time_window_range[1] for time_window_range in time_window_ranges], end_time) if end_time is not None else len(time_window_ranges)
        
    return time_window_ranges[start_index:end_index], time_window_indices[start_index:end_index]
