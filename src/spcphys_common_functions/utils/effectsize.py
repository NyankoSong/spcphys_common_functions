'''Modified from https://github.com/nbashir97/effectsize'''

from typing import Tuple
import numpy as np
from scipy import stats
from astropy import units as u

from . import config
from .utils import check_parameters
    

@check_parameters
def es_cohen(x1: np.ndarray|u.Quantity, x2:np.ndarray|u.Quantity, conf_interval: float=0.99, log_scale: bool=False, skewed: bool=False) -> Tuple[float, Tuple[float, float]]:
    '''
    Calculate Cohen's d (Hedges' g) effect size and confidence interval.
    
    :param x1: Data array 1.
    :param x2: Data array 2.
    :param conf_interval: Confidence interval, default 0.99.
    :param log_scale: If True, apply log scale to the data.
    :param skewed: If True, apply rank-based transformation to the data.
    
    :return es: Cohen's d effect size.
    :return ci: Confidence interval.
    '''
    
    if len(x1.shape) != 1 or len(x2.shape) != 1:
        raise ValueError("x1 and x2 must be 1D arrays.")
    if log_scale and skewed:
        raise ValueError("log_scale and skewed cannot be True at the same time.")
    if not 0 < conf_interval < 1:
        raise ValueError("conf_interval must be a float between 0 and 1.")
     
    if skewed:
        x_total = np.concatenate((x1, x2))
        ranks = np.argsort(x_total)
        x1 = ranks[:len(x1)]
        x2 = ranks[len(x1):]
        
    if log_scale:
        x1 = np.log10(x1)
        x2 = np.log10(x2)
        
    es = (x1.mean() - x2.mean()) / np.sqrt((x1.std(ddof=1)**2 + x2.std(ddof=1)**2) / 2)
    
    percentile = 1 - ((1 - conf_interval) / 2)
    zscore = stats.norm.ppf(percentile)
    deviation = np.sqrt((x1.size + x2.size) / (x1.size * x2.size) + es**2 / (2 * (x1.size + x2.size)))
    ci = zscore * deviation
    
    return es, ci