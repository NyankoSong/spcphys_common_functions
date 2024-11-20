from datetime import timedelta
import numpy as np
from astropy import stats as astats
from astropy import units as u
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic
from sunpy.coordinates import HeliographicCarrington
from sunpy.sun.constants import sidereal_rotation_rate

from ..utils.utils import check_parameters


@check_parameters
def most_probable_x(x: u.Quantity|np.ndarray, bins: np.ndarray|str|int='freedman') -> float|u.Quantity:
    """
    Calculate the most probable value in the input array.

    :param x: Input one-dimensional array or quantity.
    :param bins: Binning method for the histogram, default is 'freedman'.
    
    :return x_mp: The most probable value in the input array.
    """
    
    if len(x.shape) > 1:
        raise ValueError('Input x must be one-dimensional.')
    
    x = x[~np.isnan(x)]
    
    var_hist, hist_bins = astats.histogram(x, bins=bins)
    var_hist_max_ind = np.argmax(var_hist)
    x_mp = hist_bins[var_hist_max_ind] + (hist_bins[var_hist_max_ind] - hist_bins[var_hist_max_ind + 1]) / 2
    
    return x_mp if not isinstance(x, u.Quantity) else x_mp * x.unit
    
    
@check_parameters
def ballistic_backmapping(pos_situ: SkyCoord, v_r: u.Quantity, r_target: u.Quantity|None =None, t_travel: timedelta|u.Quantity|None =None) -> SkyCoord:
    """
    Calculate the target position for ballistic backmapping.

    :param pos_situ: Current position, should be able to transform to HeliographicCarrington.
    :param v_r: Radial velocity, must have velocity units (u.m/u.s).
    :param r_target: Target radius, default is None.
    :param t_travel: Travel time, must have time units (u.s), default is None.
    
    :return pos_target: Target position in heliographic Carrington coordinates.
    """
    
    if (r_target is None and t_travel is None) or (r_target is not None and t_travel is not None):
        raise ValueError('Either r_target or t_travel must be specified.')
    if not v_r.unit.is_equivalent(u.m / u.s):
        raise ValueError('Radial velocity v_r must have units of velocity (u.m/u.s)')
    if isinstance(t_travel, u.Quantity) and not t_travel.unit.is_equivalent(u.s):
        raise ValueError('Travel time t_travel must have units of time (u.s)')
        
    if pos_situ.frame is not HeliographicCarrington:
        pos_situ = pos_situ.transform_to(HeliographicCarrington)
        
    v_r = v_r.si
    r_target = r_target.si if r_target is not None else r_target
    t_travel = t_travel.si if t_travel is not None and isinstance(t_travel, u.Quantity) else t_travel.total_seconds() * u.s if t_travel is not None else t_travel
        
    if t_travel is None:
        t_travel = ((pos_situ.radius - r_target) / v_r).si
    elif r_target is None:
        r_target = (pos_situ.radius - v_r * t_travel).si
        
    phi_target = (pos_situ.lon + t_travel * sidereal_rotation_rate.to(u.deg/u.s)).to(u.deg) % (360 * u.deg)
    time_target = [pos_situ.obstime.value[i] - timedelta(seconds=np.float64(t_travel.value[i])) for i in range(len(pos_situ.obstime))]
    
    return SkyCoord(lon=phi_target, lat=pos_situ.lat, radius=np.repeat(r_target, len(phi_target)), obstime=time_target, observer='earth', frame=HeliographicCarrington)