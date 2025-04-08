from astropy import units as u
from astropy.constants import k_B, mu0
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

from ..utils.utils import check_parameters
from ..processing.preprocess import interpolate


@check_parameters
def pressure_thermal(n: u.Quantity, T: u.Quantity):    
    '''Calculate thermal pressure.
    
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param T: Proton temperature data in shape (time)
    :type T: astropy.units.Quantity
    :return: Thermal pressure
    :rtype: astropy.units.Quantity
    '''
    
    if not n.unit.is_equivalent(u.m**-3):
        raise TypeError("n must be a quantity with units of number density (m^-3)")
    if not T.unit.is_equivalent(u.K):
        raise TypeError("T must be a quantity with units of temperature (K)")
    
    n = n.si
    T = T.si
        
    return (n * k_B * T).si


@check_parameters
def pressure_magnetic(b: u.Quantity):
    '''Calculate magnetic pressure.
    
    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :return: Magnetic pressure
    :rtype: astropy.units.Quantity
    '''
    
    if not b.unit.is_equivalent(u.T):
        raise TypeError("b must be a quantity with units of magnetic field (T)")
    
    b = b.si
        
    return (np.linalg.norm(b, axis=1)**2 / (2 * mu0)).si


@check_parameters
def calc_beta(p_date: List[datetime]|np.ndarray, n: u.Quantity, b_date: List[datetime]|np.ndarray, b: u.Quantity, T: u.Quantity):
    '''Calculate plasma beta.
    
    :param p_date: List or array of datetime objects for proton number density data
    :type p_date: List[datetime] or numpy.ndarray
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param b_date: List or array of datetime objects for magnetic field data
    :type b_date: List[datetime] or numpy.ndarray
    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :param T: Proton temperature data in shape (time)
    :type T: astropy.units.Quantity
    :return: Plasma beta
    :rtype: astropy.units.Quantity
    '''
    
    if not n.unit.is_equivalent(u.m**-3):
        raise TypeError("n must be a quantity with units of number density (m^-3)")
    if not b.unit.is_equivalent(u.T):
        raise TypeError("b must be a quantity with units of magnetic field (T)")
    if not T.unit.is_equivalent(u.K):
        raise TypeError("T must be a quantity with units of temperature (K)")
    
    n = n.si
    b = b.si
    T = T.si
    
    if isinstance(p_date, list):
        p_date = np.array(p_date)
    if isinstance(b_date, list):
        b_date = np.array(b_date)
    
    pth = pressure_thermal(n, T)
    pb = pressure_magnetic(b)
    
    pb = interpolate(p_date, b_date, pb, vector_norm_interp=True)
    
    return pth / pb



# if __name__ == "__main__":
#     # Test data
#     from datetime import timedelta
#     p_date = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
#     n = (np.random.rand(10)*10+10) * u.m**-3
#     b_date = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
#     b = np.random.rand(10, 3)*20 * u.T
#     T = (np.random.rand(10)*10000+100000) * u.K
    
#     print(calc_beta(p_date, n, b_date, b, T))