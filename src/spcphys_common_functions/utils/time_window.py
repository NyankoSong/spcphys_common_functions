from typing import List
from datetime import datetime, timedelta
from bisect import bisect_left
import numpy as np

from . import config
from .utils import check_parameters


@check_parameters
def slide_time_window(time: List[datetime], window_size: timedelta|int, step: timedelta|int =1, start_time: datetime|None =None) -> List[np.ndarray]:
    
    if config._ENABLE_VALUE_CHECKING:
        if type(window_size) != type(step):
            raise ValueError('window_size and step should have the same type (timedelta or int).')
        
    if isinstance(window_size, int):
        time_window_indices = [np.arange(i, i + window_size) for i in range(len(time) - window_size + 1, step)]
    elif isinstance(window_size, timedelta):
        if start_time is None:
            start_time = time[0]
        time_window_indices = [np.arange(bisect_left(time, start_time + i * step), bisect_left(time, start_time + i * step + window_size)) for i in range(((time[-1] - time[0]).total_seconds() - window_size.total_seconds()) / step.total_seconds() + 1)]
    else:
        raise ValueError('window_size and step should be either int or timedelta.')
        
    return time_window_indices
    