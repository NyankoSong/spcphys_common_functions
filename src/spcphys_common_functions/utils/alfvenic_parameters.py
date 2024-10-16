from typing import List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from astropy import units as u
# from scipy.constants import mu_0, m_p
from astropy.constants import mu0, m_p

from . import config
from .utils import check_parameters


@check_parameters
def calc_dx(x: u.Quantity, axis=0, **mean_kwargs):
    return x - np.nanmean(x, axis=axis, **mean_kwargs)


@check_parameters
def calc_va(b: u.Quantity, n: u.Quantity, dva: bool = False):
    bottom = np.sqrt(mu0 * np.nanmean(n) * m_p)
    if dva:
        return calc_dx(b) / bottom
    else:
        return b / bottom
    
    
@check_parameters
def multi_dimensional_interpolate(x: List[datetime]|np.ndarray|u.Quantity, xp: List[datetime]|np.ndarray|u.Quantity, yp: u.Quantity, **interp_kwargs):
    
    if config._ENABLE_VALUE_CHECKING:
        if yp.shape[0] != xp.shape[0]:
            raise ValueError(f"xp and yp must have the same number of rows. (But got xp.shape={xp.shape} and yp.shape={yp.shape})")
    
    if isinstance(x[0], datetime):
        x = np.array([i.timestamp() for i in x])
    if isinstance(xp[0], datetime):
        xp = np.array([i.timestamp() for i in xp])
        
    y = np.zeros((x.shape(0), yp.shape[1]))
    for i, col in enumerate(yp.T):
        y[:, i] = np.interp(x, xp, col, **interp_kwargs)
        
    return y


@check_parameters
def calc_alfven(p_date: List[datetime], v: u.Quantity, n: u.Quantity, b_date: List[datetime], b: u.Quantity) -> Tuple[float, float, float]:
    '''
    Calculate the Alfvenic parameters (corrlation coefficient between velocity and magnetic field, residual energy, cross helicity, Alfven ratio, compressibility).
    
    :param p_date: List of datetime objects for proton velocity data.
    :param v: Proton velocity data in shape (time, 3).
    :param n: Proton number density data in shape (time).
    :param b_date: List of datetime objects for magnetic field data.
    :param b: Magnetic field data in shape (time, 3).
    
    :return: Tuple of Alfvenic parameters (correlation coefficient between velocity and magnetic field, residual energy, cross helicity, Alfven ratio, compressibility).
    '''
    
    if config._ENABLE_VALUE_CHECKING:
        if not v.unit.is_equivalent(u.m/u.s):
            raise ValueError("v must be a quantity with units of velocity (m/s).")
        if not n.unit.is_equivalent(u.m**-3):
            raise ValueError("n must be a quantity with units of number density (m^-3).")
        if not b.unit.is_equivalent(u.T):
            raise ValueError("b must be a quantity with units of magnetic field (T).")
        
    v = v.si
    n = n.si
    b = b.si
    
    dv = calc_dx(v) #dV
    dvA = multi_dimensional_interpolate(p_date, b_date, calc_va(b, n, dva=True)) # dV_A
    dv2_mean = np.nansum(dv**2) / len(dv) # <dV^2>
    dvA2_mean = np.nansum(dvA**2) / len(dvA) # <dV_A^2>
    dv_dvA_mean = np.trace(np.dot(dv.T, dvA)) / len(dv) # <dV * dV_A>
    
    dn = calc_dx(n)
    
    b_magnitude = np.linalg.norm(b, axis=1)
    # 可压缩系数的磁场扰动是先做差、再取模
    db = calc_dx(b)
    db_magnitude2_mean = np.nansum(db**2) / len(db)
    
    r3 = dv_dvA_mean / np.sqrt(dv2_mean * dvA2_mean) # Wu2021, Cvb
    residual_energy = (dv2_mean - dvA2_mean) / (dv2_mean + dvA2_mean) # Wu2021, R
    cross_helicity = 2 * dv_dvA_mean / (dv2_mean + dvA2_mean)
    alfven_ratio = dv2_mean / dvA2_mean
    compressibility = np.nanmean(dn**2) * np.nanmean(b_magnitude**2) / (np.nanmean(n)**2 * db_magnitude2_mean)
    
    return r3, residual_energy, cross_helicity, alfven_ratio, compressibility



# if __name__ == "__main__":
#     # Test data
#     from datetime import timedelta
#     p_date = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
#     v = np.random.rand(10, 3)
#     n = np.random.rand(10)
#     b_date = [datetime(2023, 1, 1, 0, 30) + timedelta(hours=i) for i in range(240)]
#     b = np.random.rand(240, 3)

#     # Call the function
#     cross_helicity, alfven_ratio, compressibility = calc_alfven(p_date, v, n, b_date, b)

#     # Print the results
#     print("Cross Helicity:", cross_helicity)
#     print("Alfven Ratio:", alfven_ratio)
#     print("Compressibility:", compressibility)