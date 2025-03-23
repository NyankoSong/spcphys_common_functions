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

from ..utils.utils import check_parameters, _determine_processes
from ..processing.preprocess import interpolate


@check_parameters
def most_probable_x(x: u.Quantity|np.ndarray, bins: np.ndarray|str|int='freedman', least_length: float|int =4) -> float|u.Quantity:
    """
    Calculate the most probable value in the input array.

    :param x: Input one-dimensional array or quantity.
    :param bins: Binning method for the histogram, default is 'freedman'.
    
    :return x_mp: The most probable value in the input array.
    """
    
    if len(x.shape) > 1:
        raise ValueError('Input x must be one-dimensional.')
    if least_length < 4:
        raise ValueError('least_length must be greater than 4.')
    
    x = x[~np.isnan(x)]
    
    if len(x) < int(least_length):
        warnings.warn(f'Input x has less than {int(least_length)} data points.')
        return np.nan
    
    var_hist, hist_bins = astats.histogram(x.to_value() if isinstance(x, u.Quantity) else x, bins=bins)
    var_hist_max_ind = np.argmax(var_hist)
    x_mp = hist_bins[var_hist_max_ind] + (hist_bins[var_hist_max_ind] - hist_bins[var_hist_max_ind + 1]) / 2
    
    return x_mp if not isinstance(x, u.Quantity) else x_mp * x.unit
    
    
@check_parameters
def ballistic_backmapping(pos_insitu: SkyCoord|HeliographicCarrington, v_r: u.Quantity, r_target: u.Quantity|None =None, t_travel: timedelta|u.Quantity|None =None) -> SkyCoord:
    """
    Calculate the target position for ballistic backmapping.
    Either the target radius or the travel time must be specified.

    :param pos_insitu: Current position, should be able to transform to HeliographicCarrington.
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
    if np.isnan(t_back):
        return np.inf
    try:
        pos_backmap = ballistic_backmapping(pos_insitu=pos_sat1, v_r=v_r, t_travel=t_back[0]*u.s)
    except ValueError:
        return np.inf
    else:
        return np.abs((pos_backmap.radius - r_sat2_interp_func(pos_backmap.obstime.value, target_sc_date, pos_sat2)).si.value)

def _target_sc_r_interp_func(target_time, target_sc_date, pos_sat2):
    return interpolate(target_time, target_sc_date, pos_sat2.radius)

# def _optimize_t_back(base_sc_pos, base_sc_v_r, _target_sc_r_interp_func, t_back_bounds):
def _optimize_t_back(args):
    base_sc_pos, base_sc_v_r, target_sc_date, pos_sat2, t_back_bounds = args
    base_sc_obs_date = base_sc_pos.obstime.value
    
    target_sc_r_init = _target_sc_r_interp_func(base_sc_obs_date, target_sc_date, pos_sat2)
    t_back_init = (base_sc_pos.radius - target_sc_r_init) / base_sc_v_r
    
    if t_back_init.si.to_value() < t_back_bounds.lb or t_back_init.si.to_value() > t_back_bounds.ub:
        return np.nan
    
    res = minimize(fun=_radius_diff, x0=t_back_init.si.to_value(), args=(base_sc_v_r, base_sc_pos, _target_sc_r_interp_func, target_sc_date, pos_sat2), bounds=t_back_bounds, method='Nelder-Mead')
    return res.x[0] if res.success else np.nan


@check_parameters
def dual_spacecraft_obs_diff(pos_sat1: SkyCoord|HeliographicCarrington, pos_sat2: SkyCoord|HeliographicCarrington, v_r: u.Quantity, num_processes: float|int =1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, u.Quantity]:
    '''
    Calculate the difference in the Carrington longitude and latitude between observations from two spacecraft.
    This function assumes that the radial velocity is oberved by the first spacecraft,
    and calculates the difference in the Carrington longitude and latitude between the parker spiral and the second spacecraft.
    
    :param pos_sat1: Position time series of the first spacecraft.
    :param pos_sat2: Position time series of the second spacecraft.
    :param v_r: Radial velocity time series or constant value, must have velocity units (u.m/u.s).
    :param num_processes: Number of processes to use for parallel processing, default is 1.
    
    :return valid_indices: Successfully backmapped indices.
    :return time_sat1: Time of the first spacecraft observation.
    :return time_sat2: Time of the second spacecraft observation.
    :return delta_phi: Difference in Carrington longitude at the radius of the second spacecraft.
    '''
    
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