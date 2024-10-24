from typing import List, Tuple
from datetime import datetime
import numpy as np
from astropy import units as u
from astropy.constants import mu0, m_p

from . import config
from .utils import check_parameters
from .time_window import _time_indices


@check_parameters
def calc_dx(x: u.Quantity|np.ndarray, axis=0, **mean_kwargs) -> u.Quantity|np.ndarray:
    '''
    Remove the mean value from the input array.
    
    :param x: Input array.
    :param axis: Axis along which to calculate the mean value. Default is 0.
    :param mean_kwargs: Additional keyword arguments to pass to the numpy.nanmean function.

    :return: Array with the mean value removed, as an astropy Quantity.
    '''
    
    return x - np.nanmean(x, axis=axis, **mean_kwargs)


@check_parameters
def calc_va(b: u.Quantity, n: u.Quantity, dva: bool = False) -> u.Quantity:
    '''
    Calculate the Alfven velocity.

    :param b: Magnetic field data in shape (time, 3).
    :param n: Proton number density data in shape (time).
    :param dva: Whether to remove mean value from magnetic field data. Default is False.
    
    :return va/dva: Alfven velocity or Alfven velocity with mean value removed.
    '''
    
    bottom = np.sqrt(mu0 * np.nanmean(n) * m_p)
    if dva:
        return (calc_dx(b) / bottom).si
    else:
        return (b / bottom).si
    
    
@check_parameters
def multi_dimensional_interpolate(x: List[datetime]|np.ndarray|u.Quantity, xp: List[datetime]|np.ndarray|u.Quantity, yp: np.ndarray|u.Quantity, **interp_kwargs) -> np.ndarray|u.Quantity:
    '''
    Perform multi-dimensional interpolation on the given data.

    :param x: Array of x-coordinates at which to evaluate the interpolated values.
    :param xp: Array of x-coordinates of the data points.
    :param yp: Array of y-coordinates of the data points. Must be a Quantity with the same number of rows as xp.
    :param interp_kwargs: Additional keyword arguments to pass to the numpy.interp function.
    
    :return y: Interpolated values at the x, with the same number of columns as yp.
    '''
    
    if config._ENABLE_VALUE_CHECKING:
        if yp.shape[0] != xp.shape[0]:
            raise ValueError(f"xp and yp must have the same number of rows. (But got xp.shape={xp.shape} and yp.shape={yp.shape})")
    
    if isinstance(x[0], datetime):
        x = np.array([i.timestamp() for i in x])
    if isinstance(xp[0], datetime):
        xp = np.array([i.timestamp() for i in xp])
        
    y = np.zeros((x.shape[0], yp.shape[1]))
    for i, col in enumerate(yp.T):
        y[:, i] = np.interp(x, xp, col, **interp_kwargs)
        
    if isinstance(yp, u.Quantity):
        return y * yp.unit
    else:
        return y


@check_parameters
def calc_alfven(p_date: List[datetime]|np.ndarray, v: u.Quantity, n: u.Quantity, b_date: List[datetime]|np.ndarray, b: u.Quantity, time_window_ranges: List[list]|List[tuple]|None =None) -> dict:
    '''
    Calculate the Alfvenic parameters (corrlation coefficient between velocity and magnetic field, residual energy, cross helicity, Alfven ratio, compressibility).
    
    :param p_date: List of datetime objects for proton velocity data.
    :param v: Proton velocity data in shape (time, 3).
    :param n: Proton number density data in shape (time).
    :param b_date: List of datetime objects for magnetic field data.
    :param b: Magnetic field data in shape (time, 3).
    
    :return r3: Correlation coefficient between velocity and magnetic field.
    :return residual_energy: Residual energy.
    :return cross_helicity: Cross helicity.
    :return alfven_ratio: Alfven ratio.
    :return compressibility: Compressibility. 
    '''
    
    if config._ENABLE_VALUE_CHECKING:
        if not v.unit.is_equivalent(u.m/u.s):
            raise ValueError("v must be a quantity with units of velocity (m/s).")
        if not n.unit.is_equivalent(u.m**-3):
            raise ValueError("n must be a quantity with units of number density (m^-3).")
        if not b.unit.is_equivalent(u.T):
            raise ValueError("b must be a quantity with units of magnetic field (T).")
        if v.shape[1] != 3 or b.shape[1] != 3:
            raise ValueError("v and b must have 3 columns.")
        if v.shape[0] != n.shape[0] or len(p_date) != n.shape[0]:
            raise ValueError("p_date, v and n must have the same number of rows.")
        if len(b_date) != b.shape[0]:
            raise ValueError("b_date and b must have the same number of rows.")
    
    if time_window_ranges is None:
        time_window_ranges = [(min(b_date[0], p_date[0]), max(b_date[-1], p_date[-1]))]
    if isinstance(p_date, list):
        p_date = np.array(p_date)
    if isinstance(b_date, list):
        b_date = np.array(b_date)
        
    v = v.si
    n = n.si
    b = b.si
    
    r3 = np.zeros(len(time_window_ranges)) * u.dimensionless_unscaled
    residual_energy = np.zeros(len(time_window_ranges)) * u.dimensionless_unscaled
    cross_helicity = np.zeros(len(time_window_ranges)) * u.dimensionless_unscaled
    alfven_ratio = np.zeros(len(time_window_ranges)) * u.dimensionless_unscaled
    compressibility = np.zeros(len(time_window_ranges)) * u.dimensionless_unscaled
    
    for i, time_range in enumerate(time_window_ranges):
        
        p_window_indices = _time_indices(p_date, time_range)
        b_window_indices = _time_indices(b_date, time_range)
        
        v_window = v[p_window_indices]
        n_window = n[p_window_indices]
        b_window = b[b_window_indices]
        
        dv = calc_dx(v_window) #dV
        dvA = multi_dimensional_interpolate(p_date[p_window_indices], b_date[b_window_indices], calc_va(b_window, n_window, dva=True)) # dV_A
        
        # dv2_mean = np.nansum(dv**2) / len(dv) # <dV^2>
        # dvA2_mean = np.nansum(dvA**2) / len(dvA) # <dV_A^2>
        # dv_dvA_mean = np.trace(np.dot(dv.T, dvA)) / len(dv) # <dV * dV_A>
        
        dv2_mean = np.nanmean(np.nansum(dv**2, axis=1))
        dvA2_mean = np.nanmean(np.nansum(dvA**2, axis=1))
        dv_dvA_mean = np.nanmean(np.einsum('ij,ij->i', dv, dvA))
        
        dn = calc_dx(n_window)
        
        b_magnitude = np.linalg.norm(b_window, axis=1)
        # 可压缩系数的磁场扰动是先做差、再取模
        db = calc_dx(b_window)
        # db_magnitude2_mean = np.nansum(db**2) / len(db)
        db_magnitude2_mean = np.nanmean(np.nansum(db**2, axis=1))
        
        r3_i = dv_dvA_mean / np.sqrt(dv2_mean * dvA2_mean) # Wu2021, Cvb
        residual_energy_i = (dv2_mean - dvA2_mean) / (dv2_mean + dvA2_mean) # Wu2021, R
        cross_helicity_i = 2 * dv_dvA_mean / (dv2_mean + dvA2_mean)
        alfven_ratio_i = dv2_mean / dvA2_mean
        compressibility_i = np.nanmean(dn**2) * np.nanmean(b_magnitude**2) / (np.nanmean(n_window)**2 * db_magnitude2_mean)
    
        r3[i], residual_energy[i], cross_helicity[i], alfven_ratio[i], compressibility[i] = r3_i.si, residual_energy_i.si, cross_helicity_i.si, alfven_ratio_i.si, compressibility_i.si
    
    return {'r3': r3, 'residual_energy': residual_energy, 'cross_helicity': cross_helicity, 'alfven_ratio': alfven_ratio, 'compressibility': compressibility}


@check_parameters
def vec_cart_to_sph(v: u.Quantity|np.ndarray, r: u.Quantity|np.ndarray, z: u.Quantity|np.ndarray|None =None) ->Tuple[u.Quantity|np.ndarray]:
    
    """
    Convert a vector from Cartesian coordinates to spherical coordinates.

    :param v: The vector to be converted. Shape should be (N, 3), where N is the number of vectors.
    :param r: The radial component of the vector. Shape should be (N, 3) or (3,).
    :param z: The z-component of the vector. Shape should be (N, 3) or (3,). If None, the function will only compute the magnitude and angle between v and r. Default is None.
    
    :return: If z is None, returns the magnitude and angle between v and r. Otherwise, returns the magnitude, azimuth, and elevation.
    """
    
    if config._ENABLE_VALUE_CHECKING:
        if type(v) != type(r):
            raise TypeError("v and r must have the same type.")
        if isinstance(v, u.Quantity) and not v.unit.is_equivalent(r.unit):
            raise ValueError("v and r must have the same units.")
        if z is not None:
            if type(v) != type(z):
                raise TypeError("v and z must have the same type.")
            if isinstance(v, u.Quantity) and not v.unit.is_equivalent(z.unit):
                raise ValueError("v and z must have the same units.")
        
    if len(r.shape) == 1 or r.shape[0] == 1:
        r = np.tile(r, (v.shape[0], 1))
    r = r / np.tile(np.linalg.norm(r, axis=1), (3, 1)).T
    
    v_mag = np.linalg.norm(v, axis=1)
    
    if z is None:
        theta = np.arccos(np.einsum('ij,ij->i', v, r) / v_mag)
        
        if isinstance(v, u.Quantity):
            theta = theta.to(u.deg)
        else:
            theta = np.rad2deg(theta)
            
        return v_mag, theta
    
    else:
        if len(z.shape) == 1 or z.shape[0] == 1:
            z = np.tile(z, (v.shape[0], 1))
        z = z / np.tile(np.linalg.norm(z, axis=1), (3, 1)).T
        
        y = np.cross(z, r)
        
        v_r, v_y, v_z = np.einsum('ij,ij->i', v, r), np.einsum('ij,ij->i', v, y), np.einsum('ij,ij->i', v, z)
        azimuth = np.arctan2(v_y, v_r)
        elevation = np.arcsin(v_z / v_mag)
        
        if isinstance(v, u.Quantity):
            azimuth = azimuth.to(u.deg)
            elevation = elevation.to(u.deg)
        else:
            azimuth = np.rad2deg(azimuth)
            elevation = np.rad2deg(elevation)
            
        return v_mag, azimuth, elevation