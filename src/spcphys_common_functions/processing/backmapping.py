'''
BALE S D, BADMAN S T, BONNELL J W, et al., 2019. Highly structured slow solar wind emerging from an equatorial coronal hole[J/OL]. Nature, 576(7786): 237-242. DOI:10.1038/s41586-019-1818-7.
'''
from typing import Tuple, Iterable
from datetime import timedelta
import warnings
from multiprocessing import Pool
import numpy as np
from astropy import stats as astats
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington
from sunpy.sun.constants import sidereal_rotation_rate
from scipy.optimize import minimize, Bounds
from tqdm import tqdm

from ..utils.utils import _determine_processes
from ..processing.preprocess import interpolate



def most_probable_x(x: u.Quantity|np.ndarray, bins: np.ndarray|str|int='freedman', least_length: float|int =4) -> float|u.Quantity:
    """Calculate the most probable value in the input array.

    :param x: Input one-dimensional array or quantity.
    :type x: astropy.units.Quantity or np.ndarray
    :param bins: Binning method for the histogram, defaults to 'freedman'.
    :type bins: np.ndarray or str or int
    :param least_length: Minimum number of data points required, defaults to 4.
    :type least_length: float or int
    :raises ValueError: If input x is not one-dimensional or if least_length is less than 4.
    :return: The most probable value in the input array.
    :rtype: float or u.Quantity
    """
    
    if len(x.shape) > 1:
        raise ValueError('Input x must be one-dimensional.')
    if least_length < 4:
        raise ValueError('least_length must be greater than 4.')
    
    x = x[~np.isnan(x)]
    
    if len(x) < int(least_length):
        warnings.warn(f'Input x has less than {int(least_length)} data points.')
        return np.nan
    x_value = x.to_value() if isinstance(x, u.Quantity) else x
    var_hist, hist_bins = astats.histogram(x_value, bins=bins)
    var_hist_max_ind = np.argmax(var_hist)
    x_mp = hist_bins[var_hist_max_ind] + (hist_bins[var_hist_max_ind + 1] - hist_bins[var_hist_max_ind]) / 2
    
    # if x_mp > np.max(x_value) or x_mp < np.min(x_value):
    #     warnings.warn('Most probable value is outside the range of input data. Setting to the boundary value.')
    #     x_mp = np.max(x_value) if x_mp > np.max(x_value) else np.min(x_value)

    return x_mp if not isinstance(x, u.Quantity) else x_mp * x.unit
    
    

def ballistic_backmapping(pos_insitu: SkyCoord|HeliographicCarrington, v_r: u.Quantity, r_target: u.Quantity|None =None, t_travel: timedelta|u.Quantity|None =None) -> SkyCoord:
    """Calculate the target position for ballistic backmapping.
    
    Either the target radius or the travel time must be specified.

    :param pos_insitu: Current position, should be able to transform to HeliographicCarrington.
    :type pos_insitu: SkyCoord or HeliographicCarrington
    :param v_r: Radial velocity, must have velocity units.
    :type v_r: astropy.units.Quantity
    :param r_target: Target radius, defaults to None.
    :type r_target: astropy.units.Quantity or None, optional
    :param t_travel: Travel time, defaults to None.
    :type t_travel: timedelta or u.Quantity or None, optional
    :raises ValueError: If neither or both r_target and t_travel are specified.
    :raises ValueError: If v_r does not have velocity units.
    :raises ValueError: If t_travel does not have time units.
    :return: Target position in heliographic Carrington coordinates.
    :rtype: SkyCoord
    """
    
    if (r_target is None and t_travel is None) or (r_target is not None and t_travel is not None):
        raise ValueError('Either r_target or t_travel must be specified.')
    if not v_r.unit.is_equivalent(u.m / u.s):
        raise ValueError('Radial velocity v_r must have units of velocity (u.m/u.s)')
    if isinstance(t_travel, u.Quantity) and not t_travel.unit.is_equivalent(u.s):
        raise ValueError('Travel time t_travel must have units of time (u.s)')
    
    if isinstance(pos_insitu, SkyCoord):
        if pos_insitu.frame is not HeliographicCarrington:
            pos_insitu = pos_insitu.transform_to(HeliographicCarrington)
        
    v_r = v_r.si
    r_target = r_target.si if r_target is not None else r_target
    t_travel = t_travel.si if t_travel is not None and isinstance(t_travel, u.Quantity) else t_travel.total_seconds() * u.s if t_travel is not None else t_travel
        
    if t_travel is None:
        t_travel = ((pos_insitu.radius - r_target) / v_r).si
    elif r_target is None:
        r_target = (pos_insitu.radius - v_r * t_travel).si
    elif len(r_target) == 0:
        r_target = np.repeat(r_target, len(phi_target))
        
    phi_target = (pos_insitu.lon + t_travel * sidereal_rotation_rate.to(u.deg/u.s)).to(u.deg) % (360 * u.deg)
    if isinstance(pos_insitu.obstime.value, Iterable):
        time_target = [pos_insitu.obstime.value[i] - timedelta(seconds=np.float64(t_travel.value[i])) for i in range(len(pos_insitu.obstime))]
    else:
        time_target = pos_insitu.obstime.value - timedelta(seconds=np.float64(t_travel.value))
    
    return SkyCoord(lon=phi_target, lat=pos_insitu.lat, radius=r_target, obstime=time_target, observer='earth', frame=HeliographicCarrington)


def _radius_diff(t_back, v_r, pos_sat1, r_sat2_interp_func, target_sc_date, pos_sat2):
    """Calculate the difference in radius between the backmapped position and the target spacecraft.

    :param t_back: Travel time for backmapping.
    :type t_back: float or np.ndarray
    :param v_r: Radial velocity.
    :type v_r: astropy.units.Quantity
    :param pos_sat1: Position of the first spacecraft.
    :type pos_sat1: SkyCoord or HeliographicCarrington
    :param r_sat2_interp_func: Function to interpolate the radius of the second spacecraft.
    :type r_sat2_interp_func: function
    :param target_sc_date: Observation date of the second spacecraft.
    :type target_sc_date: datetime or np.ndarray of datetime
    :param pos_sat2: Position of the second spacecraft.
    :type pos_sat2: SkyCoord or HeliographicCarrington
    :return: Absolute difference in radius.
    :rtype: float
    """
    if np.isnan(t_back):
        return np.inf
    try:
        pos_backmap = ballistic_backmapping(pos_insitu=pos_sat1, v_r=v_r, t_travel=t_back[0]*u.s)
    except ValueError:
        return np.inf
    else:
        return np.abs((pos_backmap.radius - r_sat2_interp_func(pos_backmap.obstime.value, target_sc_date, pos_sat2)).si.value)


def _target_sc_r_interp_func(target_time, target_sc_date, pos_sat2):
    """Interpolate the radius of the second spacecraft at the given time.

    :param target_time: Target time to interpolate for.
    :type target_time: datetime or np.ndarray of datetime
    :param target_sc_date: Time series of the second spacecraft.
    :type target_sc_date: list or np.ndarray of datetime
    :param pos_sat2: Position of the second spacecraft.
    :type pos_sat2: SkyCoord or HeliographicCarrington
    :return: Interpolated radius of the second spacecraft.
    :rtype: astropy.units.Quantity
    """
    return interpolate(target_time, target_sc_date, pos_sat2.radius)


def _optimize_t_back(args):
    """Optimize backmapping time to match the radius of two spacecraft.

    :param args: Tuple containing (base_sc_pos, base_sc_v_r, target_sc_date, pos_sat2, t_back_bounds).
    :type args: tuple
    :return: Optimized backmapping time or NaN if optimization fails.
    :rtype: float
    """
    base_sc_pos, base_sc_v_r, target_sc_date, pos_sat2, t_back_bounds = args
    base_sc_obs_date = base_sc_pos.obstime.value
    
    target_sc_r_init = _target_sc_r_interp_func(base_sc_obs_date, target_sc_date, pos_sat2)
    t_back_init = (base_sc_pos.radius - target_sc_r_init) / base_sc_v_r
    
    if t_back_init.si.to_value() < t_back_bounds.lb or t_back_init.si.to_value() > t_back_bounds.ub:
        return np.nan
    
    res = minimize(fun=_radius_diff, x0=t_back_init.si.to_value(), args=(base_sc_v_r, base_sc_pos, _target_sc_r_interp_func, target_sc_date, pos_sat2), bounds=t_back_bounds, method='Nelder-Mead')
    return res.x[0] if res.success else np.nan



def dual_spacecraft_obs_diff(pos_sat1: SkyCoord|HeliographicCarrington, pos_sat2: SkyCoord|HeliographicCarrington, v_r: u.Quantity, num_processes: float|int =1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, u.Quantity]:
    """Calculate the difference in the Carrington longitude and latitude between observations from two spacecraft.
    
    This function assumes that the radial velocity is oberved by the first spacecraft,
    and calculates the difference in the Carrington longitude and latitude between the parker spiral and the second spacecraft.
    
    :param pos_sat1: Position time series of the first spacecraft.
    :type pos_sat1: SkyCoord or HeliographicCarrington
    :param pos_sat2: Position time series of the second spacecraft.
    :type pos_sat2: SkyCoord or HeliographicCarrington
    :param v_r: Radial velocity time series or constant value, must have velocity units.
    :type v_r: astropy.units.Quantity
    :param num_processes: Number of processes to use for parallel processing, defaults to 1.
    :type num_processes: float or int, optional
    :raises ValueError: If v_r does not have velocity units.
    :return: A tuple containing (valid_indices, time_sat1, time_sat2, delta_phi) where:
             valid_indices are successfully backmapped indices,
             time_sat1 is the time of the first spacecraft observation,
             time_sat2 is the time of the second spacecraft observation,
             delta_phi is the difference in Carrington longitude at the radius of the second spacecraft.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, u.Quantity]
    """
    
    if pos_sat1.frame is not HeliographicCarrington:
        pos_sat1 = pos_sat1.transform_to(HeliographicCarrington)
    if pos_sat2.frame is not HeliographicCarrington:
        pos_sat2 = pos_sat2.transform_to(HeliographicCarrington)
    if not v_r.unit.is_equivalent(u.m / u.s):
        raise ValueError('Radial velocity v_r must have units of velocity (u.m/u.s)')
        
    v_r = v_r.si
    if not isinstance(v_r, Iterable):
        v_r = np.repeat(v_r, len(pos_sat1))
    t_back = np.full(v_r.shape, np.nan) * u.s
    delta_phi = np.full(v_r.shape, np.nan) * u.deg
    
    def _target_sc_lon_interp_func(target_time):
        # return (np.angle(np.interp(target_timestamp, target_sc_date_timestamp, np.exp(1j * pos_sat2.lon.to(u.rad).value)), deg=True) + 360) % 360 * u.deg
        return (interpolate(target_time, pos_sat2.obstime.value, pos_sat2.lon).to_value() + 360) % 360*u.deg

    t_back_lower_boundary = (pos_sat1.radius.min() - pos_sat2.radius.max()) / np.nanmin(v_r)
    t_back_upper_boundary = (pos_sat1.radius.max() - pos_sat2.radius.min()) / np.nanmin(v_r)
    t_back_bounds = Bounds(t_back_lower_boundary.si.to_value(), t_back_upper_boundary.si.to_value())
    
    num_processes = _determine_processes(num_processes)
    if num_processes == 1:
        for i, (base_sc_pos, base_sc_v_r) in tqdm(enumerate(zip(pos_sat1, v_r)), total=len(v_r), desc='Backmapping', unit='data'):
            t_back[i] = _optimize_t_back((base_sc_pos, base_sc_v_r, pos_sat2.obstime.value, pos_sat2, t_back_bounds)) * u.s
    else:
        args_list = [(base_sc_pos, base_sc_v_r, pos_sat2.obstime.value, pos_sat2, t_back_bounds) for base_sc_pos, base_sc_v_r in zip(pos_sat1, v_r)]
        with Pool(num_processes) as pool:
            t_back = np.array(list(tqdm(pool.imap(_optimize_t_back, args_list), total=len(v_r), desc='Backmapping', unit='data'))) * u.s
    
    failed_backmap = np.isnan(t_back)
    t_back[failed_backmap] = 0*u.s
    pos_backmap = ballistic_backmapping(pos_insitu=pos_sat1, v_r=v_r, t_travel=t_back)
    time_sat1 = pos_sat1.obstime.value[~failed_backmap]
    time_sat2 = pos_backmap.obstime.value[~failed_backmap]
    delta_phi = ((pos_backmap.lon - _target_sc_lon_interp_func(pos_backmap.obstime.value)).to(u.deg).to_value() + 360) % 360 * u.deg
    delta_phi[delta_phi > 180*u.deg] -= 360*u.deg
    delta_phi = delta_phi[~failed_backmap]
    
    if failed_backmap.any():
        warnings.warn(f'Failed to backmap {failed_backmap.sum()} data points.')
        
    return np.arange(len(t_back))[~failed_backmap], time_sat1, time_sat2, delta_phi