import numpy as np
import pandas as pd
from astropy import units as u
from scipy.constants import mu_0, m_p
from datetime import datetime
from typing import List, Tuple

from . import config
from .utils import check_parameters


@check_parameters
def calc_alfven(p_date: List[datetime], v: u.Quantity, n: u.Quantity, b_date: List[datetime], b: u.Quantity) -> Tuple[float, float, float]:
    '''
    Calculate the Alfvenic parameters (cross helicity, Alfven ratio, and compressibility) for a given time window.
    
    :param p_date: List of datetime objects for proton velocity data.
    :param v: Proton velocity data in shape (time, 3).
    :param n: Proton number density data in shape (time).
    :param b_date: List of datetime objects for magnetic field data.
    :param b: Magnetic field data in shape (time, 3).
    
    :return: Tuple of cross helicity, Alfven ratio, and compressibility.
    '''
    
    if config.ENABLE_VALUE_CHECKING:
        if not v.unit.is_equivalent(u.m/u.s):
            raise ValueError("v must be a quantity with units of velocity (m/s).")
        if not n.unit.is_equivalent(u.m**-3):
            raise ValueError("n must be a quantity with units of number density (m^-3).")
        if not b.unit.is_equivalent(u.T):
            raise ValueError("b must be a quantity with units of magnetic field (T).")
        
    v = v.si.to_value()
    n = n.si.to_value()
    b = b.si.to_value()
    
    dv = v - np.nanmean(v, axis=0) #dV
    dv2_mean = np.nanmean(np.linalg.norm(dv, axis=1)**2) #<dV^2>
    dv_A = (b - np.nanmean(b, axis=0)) / (mu_0*np.nanmean(n)*m_p)**0.5 # dV_A
    dv_A = pd.DataFrame(dv_A, index=b_date, columns=['Bx', 'By', 'Bz']).reindex(p_date + b_date).sort_index().interpolate().loc[p_date].values # 插值对齐
    dv_A2_mean = np.nanmean(np.linalg.norm(dv_A, axis=1)**2) # <dV_A^2>
    dv_dv_A_mean = np.trace(np.dot(dv[~(np.isnan(dv[:, 0]) | np.isnan(dv_A[:, 0])), :].T, dv_A[~(np.isnan(dv[:, 0]) | np.isnan(dv_A[:, 0])), :])) / len(p_date) # <dV·dV_A>
    dn = n - np.nanmean(n)
    
    # 可压缩系数的磁场扰动是先做差、再取模
    b_magnitude = np.linalg.norm(b, axis=1)
    db = b - np.nanmean(b, axis=0)
    db_magnitude = np.linalg.norm(db, axis=1)
    
    cross_helicity = 2*dv_dv_A_mean / (dv2_mean + dv_A2_mean)
    alfven_ratio = dv2_mean / dv_A2_mean
    compressibility = np.nanmean(dn**2) * np.nanmean(b_magnitude**2) / (np.nanmean(n)**2 * np.nanmean(db_magnitude**2))
    
    return cross_helicity, alfven_ratio, compressibility



if __name__ == "__main__":
    # Test data
    from datetime import timedelta
    p_date = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    v = np.random.rand(10, 3)
    n = np.random.rand(10)
    b_date = [datetime(2023, 1, 1, 0, 30) + timedelta(hours=i) for i in range(240)]
    b = np.random.rand(240, 3)

    # Call the function
    cross_helicity, alfven_ratio, compressibility = calc_alfven(p_date, v, n, b_date, b)

    # Print the results
    print("Cross Helicity:", cross_helicity)
    print("Alfven Ratio:", alfven_ratio)
    print("Compressibility:", compressibility)