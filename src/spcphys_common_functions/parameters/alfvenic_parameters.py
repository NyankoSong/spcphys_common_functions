'''
YAO S, HE J S, TU C Y, et al., 2013. SMALL-SCALE PRESSURE-BALANCED STRUCTURES DRIVEN BY MIRROR-MODE WAVES IN THE SOLAR WIND[J/OL]. The Astrophysical Journal, 776(2): 94. DOI:10.1088/0004-637X/776/2/94.
WU H, TU C, WANG X, et al., 2021. Magnetic and Velocity Fluctuations in the Near-Sun Region from 0.1−0.3 au Observed by Parker Solar Probe[J/OL]. The Astrophysical Journal, 922(2): 92. DOI:10.3847/1538-4357/ac3331.

'''

from typing import List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import mu0, m_p
from scipy import stats as sstats


from ..processing.time_window import slide_time_window
from ..processing.preprocess import interpolate



def calc_dx(x: u.Quantity|np.ndarray, axis=0, **mean_kwargs) -> u.Quantity|np.ndarray:
    '''Remove the mean value from the input array.
    
    :param x: Input array
    :type x: astropy.units.Quantity or numpy.ndarray
    :param axis: Axis along which to calculate the mean value, defaults to 0
    :type axis: int, optional
    :param mean_kwargs: Additional keyword arguments to pass to the numpy.nanmean function
    :type mean_kwargs: dict
    :return: Array with the mean value removed
    :rtype: astropy.units.Quantity or numpy.ndarray
    '''
    
    return x - np.nanmean(x, axis=axis, **mean_kwargs)



def calc_va(b: u.Quantity, n: u.Quantity, dva: bool = False) -> u.Quantity:
    '''Calculate the Alfven velocity.

    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param dva: Whether to remove mean value from magnetic field data, defaults to False
    :type dva: bool, optional
    :return: Alfven velocity or Alfven velocity with mean value removed
    :rtype: astropy.units.Quantity
    '''
    
    bottom = np.sqrt(mu0 * np.nanmean(n) * m_p)
    if dva:
        return (calc_dx(b) / bottom).si
    else:
        return (b / bottom).si



def calc_alfven(p_date: List[datetime]|np.ndarray, v: u.Quantity, n: u.Quantity, b_date: List[datetime]|np.ndarray, b: u.Quantity, least_data_in_window: int|float =20):
    '''Calculate the Alfvenic parameters.
    
    :param p_date: List or array of datetime objects for proton velocity data
    :type p_date: List[datetime] or numpy.ndarray
    :param v: Proton velocity data in shape (time, 3)
    :type v: astropy.units.Quantity
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param b_date: List or array of datetime objects for magnetic field data
    :type b_date: List[datetime] or numpy.ndarray
    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :param least_data_in_window: Least number of valid data points, defaults to 20
    :type least_data_in_window: int or float, optional
    :return: Dictionary containing Alfvenic parameters (r3, p3, residual_energy, cross_helicity, alfven_ratio, compressibility, vA, num_valid_p_points, num_valid_b_points)
    :rtype: dict
    
    This function calculates the correlation coefficient between velocity and magnetic field, 
    residual energy, cross helicity, Alfven ratio, and compressibility.
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
    
    valid_p_indices = np.isfinite(np.concatenate((v.to_value(), n[:, np.newaxis].to_value()), axis=1)).all(axis=1)
    valid_b_indices = np.isfinite(b.to_value()).all(axis=1)
    num_valid_p_points, num_valid_b_points = np.sum(valid_p_indices), np.sum(valid_b_indices)
    
    if num_valid_p_points < least_data_in_window or num_valid_b_points < least_data_in_window:
        return {'r3': np.nan, 'p3': np.nan,
                'residual_energy': np.nan, 'cross_helicity': np.nan, 
                'alfven_ratio': np.nan, 'compressibility': np.nan, 
                'vA': np.nan, 'num_valid_p_points': num_valid_p_points, 
                'num_valid_b_points': num_valid_b_points}
    
    if isinstance(p_date, list):
        p_date = np.array(p_date)
    if isinstance(b_date, list):
        b_date = np.array(b_date)
    if isinstance(least_data_in_window, float):
        least_data_in_window = int(least_data_in_window)

    # Delete invalid data parts to avoid more valid magnetic field data than valid proton data after interpolation
    p_date = p_date[valid_p_indices]
    b_date = b_date[valid_b_indices]

    v = v[valid_p_indices].si
    n = n[valid_p_indices].si
    b = b[valid_b_indices].si
    
    dv = calc_dx(v) #dV
    dvA = interpolate(p_date, b_date, calc_va(b, n, dva=True), vector_norm_interp=True) #dV_A
    
    # dv2_mean = np.nansum(dv**2) / len(dv) # <dV^2>
    # dvA2_mean = np.nansum(dvA**2) / len(dvA) # <dV_A^2>
    # dv_dvA_mean = np.trace(np.dot(dv.T, dvA)) / len(dv) # <dV * dV_A>
    
    dv2_mean = np.nanmean(np.nansum(dv**2, axis=1)) # <dV^2>
    dvA2_mean = np.nanmean(np.nansum(dvA**2, axis=1)) # <dV_A^2>
    dv_dvA_mean = np.nanmean(np.einsum('ij,ij->i', dv, dvA)) # <dV * dV_A>
    
    dn = calc_dx(n)
    
    b_magnitude = np.linalg.norm(b, axis=1)
    # Compressibility of magnetic field perturbation is calculated by subtracting first and then taking the modulus
    db = calc_dx(b)
    # db_magnitude2_mean = np.nansum(db**2) / len(db)
    db_magnitude2_mean = np.nanmean(np.nansum(db**2, axis=1))
    
    r3_i = dv_dvA_mean / np.sqrt(dv2_mean * dvA2_mean) # Wu2021, Cvb
    p3_i = (1 - sstats.t.cdf(np.abs(r3_i) * np.sqrt((num_valid_p_points - 2) / (1 - r3_i**2)), df=num_valid_p_points - 2)) * 2 * u.dimensionless_unscaled
    residual_energy_i = (dv2_mean - dvA2_mean) / (dv2_mean + dvA2_mean) # Wu2021, R
    cross_helicity_i = 2 * dv_dvA_mean / (dv2_mean + dvA2_mean)
    alfven_ratio_i = dv2_mean / dvA2_mean
    compressibility_i = np.nanmean(dn**2) * np.nanmean(b_magnitude)**2 / (np.nanmean(n)**2 * db_magnitude2_mean)
    vA_i = np.nanmean(np.linalg.norm(calc_va(b, n, dva=False), axis=1))

    return {'r3': r3_i.si, 'p3': p3_i.si,
            'residual_energy': residual_energy_i.si, 'cross_helicity': cross_helicity_i.si, 
            'alfven_ratio': alfven_ratio_i.si, 'compressibility': compressibility_i.si, 
            'vA': vA_i.si, 'num_valid_p_points': num_valid_p_points, 
            'num_valid_b_points': num_valid_b_points}


def calc_alfven_t(p_date: List[datetime]|np.ndarray, v: u.Quantity, n: u.Quantity, b_date: List[datetime]|np.ndarray, b: u.Quantity, least_data_in_window: int|float =20, **slide_time_window_kwargs) -> dict:
    '''Calculate the Alfvenic parameters over time windows.
    
    :param p_date: List or array of datetime objects for proton velocity data
    :type p_date: List[datetime] or numpy.ndarray
    :param v: Proton velocity data in shape (time, 3)
    :type v: astropy.units.Quantity
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param b_date: List or array of datetime objects for magnetic field data
    :type b_date: List[datetime] or numpy.ndarray
    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :param least_data_in_window: Least number of valid data points in each time window, defaults to 20
    :type least_data_in_window: int or float, optional
    :param slide_time_window_kwargs: Additional keyword arguments to pass to the slide_time_window function
    :type slide_time_window_kwargs: dict
    :return: Dictionary of Alfvenic parameters for each time window (time, r3, p3, residual_energy, cross_helicity, alfven_ratio, compressibility, vA, time_window, num_valid_p_points, num_valid_b_points)
    :rtype: dict
    
    This function calculates time-dependent Alfvenic parameters including correlation coefficient
    between velocity and magnetic field, residual energy, cross helicity, Alfven ratio, and compressibility.
    '''
    
    if 'start_time' not in slide_time_window_kwargs:
        slide_time_window_kwargs['start_time'] = p_date[0]
    if 'end_time' not in slide_time_window_kwargs:
        slide_time_window_kwargs['end_time'] = p_date[-1]
    if 'window_size' not in slide_time_window_kwargs:
        slide_time_window_kwargs['window_size'] = slide_time_window_kwargs['end_time'] - slide_time_window_kwargs['start_time']
    if 'step' not in slide_time_window_kwargs:
        slide_time_window_kwargs['step'] = slide_time_window_kwargs['window_size']
    time_windows, p_time_window_indices = slide_time_window(p_date, **slide_time_window_kwargs)
    if 'align_to' not in slide_time_window_kwargs:
        _, b_time_window_indices = slide_time_window(b_date, align_to=[t[0] for t in time_windows], **slide_time_window_kwargs)
    else:
        _, b_time_window_indices = slide_time_window(b_date, **slide_time_window_kwargs)

    num_window = len(time_windows)
    
    r3 = np.zeros(num_window) * u.dimensionless_unscaled
    p3 = np.zeros(num_window) * u.dimensionless_unscaled
    residual_energy = np.zeros(num_window) * u.dimensionless_unscaled
    cross_helicity = np.zeros(num_window) * u.dimensionless_unscaled
    alfven_ratio = np.zeros(num_window) * u.dimensionless_unscaled
    compressibility = np.zeros(num_window) * u.dimensionless_unscaled
    vA = np.zeros(num_window) * u.m/u.s
    
    num_valid_p_points = np.zeros(num_window)
    num_valid_b_points = np.zeros(num_window)
    
    for i, (p_window_indices, b_window_indices) in enumerate(zip(p_time_window_indices, b_time_window_indices)):

        alfven_params_window = calc_alfven(p_date=p_date[p_window_indices], 
                                           v=v[p_window_indices], 
                                           n=n[p_window_indices], 
                                           b_date=b_date[b_window_indices], 
                                           b=b[b_window_indices], 
                                           least_data_in_window=least_data_in_window)
        
        num_valid_p_points[i], num_valid_b_points[i] = alfven_params_window['num_valid_p_points'], alfven_params_window['num_valid_b_points']
        r3[i], p3[i], residual_energy[i], cross_helicity[i], alfven_ratio[i], compressibility[i], vA[i] = alfven_params_window['r3'], alfven_params_window['p3'], alfven_params_window['residual_energy'], alfven_params_window['cross_helicity'], alfven_params_window['alfven_ratio'], alfven_params_window['compressibility'], alfven_params_window['vA']
    
    return {'time': [t[0] + (t[1] - t[0])/2 for t in time_windows], 'r3': r3, 'p3': p3, 'residual_energy': residual_energy, 'cross_helicity': cross_helicity, 'alfven_ratio': alfven_ratio, 'compressibility': compressibility, 'vA': vA,
            'time_window': time_windows, 'num_valid_p_points': num_valid_p_points, 'num_valid_b_points': num_valid_b_points}
    # return {'time': [t[0] for t in time_windows], 'r3': r3, 'residual_energy': residual_energy, 'cross_helicity': cross_helicity, 'alfven_ratio': alfven_ratio, 'compressibility': compressibility, 'vA': vA,
    #         'time_window': time_windows, 'num_valid_p_points': num_valid_p_points, 'num_valid_b_points': num_valid_b_points}
