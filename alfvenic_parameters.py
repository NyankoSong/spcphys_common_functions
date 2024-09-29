import numpy as np
import pandas as pd
from scipy.constants import mu_0, m_p
from datetime import datetime
from typing import List

def calc_alfven_window(p_date_window: List[datetime], v_window: np.ndarray, n_window: np.ndarray, b_date_window: List[datetime], b_window: np.ndarray):
    '''
    Calculate the Alfvenic parameters (cross helicity, Alfven ratio, and compressibility) for a given time window.
    
    :param p_date_window: Time window for proton data.
    :param v_window: Proton velocity data in shape (time, 3).
    :param n_window: Proton number density data in shape (time).
    :param b_date_window: Time window for magnetic field data.
    :param b_window: Magnetic field data in shape (time, 3).
    
    :return: Tuple of cross helicity, Alfven ratio, and compressibility.
    '''
    
    dv = v_window - np.nanmean(v_window, axis=0) #dV
    dv2_mean = np.nanmean(np.linalg.norm(dv, axis=1)**2) #<dV^2>
    dv_A = (b_window - np.nanmean(b_window, axis=0)) / (mu_0*np.nanmean(n_window)*m_p)**0.5 # dV_A
    dv_A = pd.DataFrame(dv_A, index=b_date_window, columns=['Bx', 'By', 'Bz']).reindex(p_date_window + b_date_window).sort_index().interpolate().loc[p_date_window].values # 插值对齐
    dv_A2_mean = np.nanmean(np.linalg.norm(dv_A, axis=1)**2) # <dV_A^2>
    dv_dv_A_mean = np.trace(np.dot(dv[~(np.isnan(dv[:, 0]) | np.isnan(dv_A[:, 0])), :].T, dv_A[~(np.isnan(dv[:, 0]) | np.isnan(dv_A[:, 0])), :])) / len(p_date_window) # <dV·dV_A>
    dn = n_window - np.nanmean(n_window)
    
    # 可压缩系数的磁场扰动是先做差、再取模
    b_magnitude = np.linalg.norm(b_window, axis=1)
    db = b_window - np.nanmean(b_window, axis=0)
    db_magnitude = np.linalg.norm(db, axis=1)
    
    cross_helicity = 2*dv_dv_A_mean / (dv2_mean + dv_A2_mean)
    alfven_ratio = dv2_mean / dv_A2_mean
    compressibility = np.nanmean(dn**2) * np.nanmean(b_magnitude**2) / (np.nanmean(n_window)**2 * np.nanmean(db_magnitude**2))
    
    return cross_helicity, alfven_ratio, compressibility



if __name__ == "__main__":
    # Test data
    from datetime import timedelta
    p_date_window = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    v_window = np.random.rand(10, 3)
    n_window = np.random.rand(10)
    b_date_window = [datetime(2023, 1, 1, 0, 30) + timedelta(hours=i) for i in range(240)]
    b_window = np.random.rand(240, 3)

    # Call the function
    cross_helicity, alfven_ratio, compressibility = calc_alfven_window(p_date_window, v_window, n_window, b_date_window, b_window)

    # Print the results
    print("Cross Helicity:", cross_helicity)
    print("Alfven Ratio:", alfven_ratio)
    print("Compressibility:", compressibility)