import numpy as np
from astropy import stats as astats
from astropy import units as u

from . import config
from .utils import check_parameters


@check_parameters
def most_probable_x(x: u.Quantity|np.ndarray, bins: np.ndarray|str='freedman') -> float:
    var_hist, hist_bins = astats.histogram(x, bins=bins)
    var_hist_max_ind = np.argmax(var_hist)
    return hist_bins[var_hist_max_ind] + (hist_bins[var_hist_max_ind] - hist_bins[var_hist_max_ind + 1]) / 2
    