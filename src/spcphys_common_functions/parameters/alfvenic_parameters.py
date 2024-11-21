'''
YAO S, HE J S, TU C Y, et al., 2013. SMALL-SCALE PRESSURE-BALANCED STRUCTURES DRIVEN BY MIRROR-MODE WAVES IN THE SOLAR WIND[J/OL]. The Astrophysical Journal, 776(2): 94. DOI:10.1088/0004-637X/776/2/94.
WU H, TU C, WANG X, et al., 2021. Magnetic and Velocity Fluctuations in the Near-Sun Region from 0.1−0.3 au Observed by Parker Solar Probe[J/OL]. The Astrophysical Journal, 922(2): 92. DOI:10.3847/1538-4357/ac3331.

'''

from typing import List, Tuple
from datetime import datetime
import numpy as np
from astropy import units as u
from astropy.constants import mu0, m_p

from ..utils.utils import check_parameters
from ..processing.time_window import _time_indices
from ..processing.preprocess import multi_dimensional_interpolate


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
def calc_alfven(p_date: List[datetime]|np.ndarray, v: u.Quantity, n: u.Quantity, b_date: List[datetime]|np.ndarray, b: u.Quantity, time_window_ranges: List[list]|List[tuple]|None =None) -> dict:
    '''
    Calculate the Alfvenic parameters (corrlation coefficient between velocity and magnetic field, residual energy, cross helicity, Alfven ratio, compressibility).
    
    :param p_date: List of datetime objects for proton velocity data.
    :param v: Proton velocity data in shape (time, 3).
    :param n: Proton number density data in shape (time).
    :param b_date: List of datetime objects for magnetic field data.
    :param b: Magnetic field data in shape (time, 3).
    
    :return {'r3': r3, 'residual_energy': residual_energy, 'cross_helicity': cross_helicity, 'alfven_ratio': alfven_ratio, 'compressibility': compressibility}: Dictionary of Alfvenic parameters.
    '''
    
    if not v.unit.is_equivalent(u.m/u.s):
        raise ValueError("v must be a quantity with unit equivalent to velocity (m/s).")
    if not n.unit.is_equivalent(u.m**-3):
        raise ValueError("n must be a quantity with unit equivalent to number density (m^-3).")
    if not b.unit.is_equivalent(u.T):
        raise ValueError("b must be a quantity with unit equivalent to magnetic field (T).")
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