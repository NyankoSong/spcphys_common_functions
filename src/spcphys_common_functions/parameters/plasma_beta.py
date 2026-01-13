from typing import Literal
from astropy import units as u
from astropy.constants import k_B, mu0
import numpy as np



def pressure_thermal(n: u.Quantity, T: u.Quantity) -> u.Quantity:    
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
    if not T.unit.is_equivalent(u.K) and not T.unit.is_equivalent(u.J):
        raise TypeError("T must be a quantity with units of temperature (K)")
    
    n = n.si
    T = T.si
    T = T.to(u.K, equivalencies=u.temperature_energy())
        
    return (n * k_B * T).si



def pressure_magnetic(b: u.Quantity) -> u.Quantity:
    '''Calculate magnetic pressure.
    
    :param b: Magnetic field data in shape (time, 3) or magnitude in shape (time,)
    :type b: astropy.units.Quantity
    :return: Magnetic pressure
    :rtype: astropy.units.Quantity
    '''
    
    if not b.unit.is_equivalent(u.T):
        raise TypeError("b must be a quantity with units of magnetic field (T)")
    
    b = b.si
    if b.ndim == 1:
        return (b**2 / (2 * mu0)).si
    elif b.ndim == 2 and b.shape[1] == 3:
        return (np.linalg.norm(b, axis=1)**2 / (2 * mu0)).si
    else:
        raise ValueError("b must be a 1D or 2D array with shape (time,) or (time, 3) respectively.")



def calc_beta(n: u.Quantity, b: u.Quantity, T: u.Quantity) -> u.Quantity:
    '''Calculate plasma beta.
    
    :param n: Proton number density data in shape (time)
    :type n: astropy.units.Quantity
    :param b: Magnetic field data in shape (time, 3)
    :type b: astropy.units.Quantity
    :param T: Proton temperature data in shape (time)
    :type T: astropy.units.Quantity
    :return: Plasma beta
    :rtype: astropy.units.Quantity
    '''
    
    n = n.si
    b = b.si
    T = T.si
    
    pth = pressure_thermal(n, T)
    pb = pressure_magnetic(b)
    
    return pth / pb


def instability_func(beta: u.Quantity | np.ndarray | float, S: float, alpha: float, beta_0: float) -> u.Quantity | np.ndarray | float:
    '''Calculate temperature anisotropy from instability function.
    Hellinger, P., Trávníček, P., Kasper, J. C., & Lazarus, A. J. (2006). Solar wind proton temperature anisotropy: Linear theory and WIND/SWE observations. Geophysical Research Letters, 33(9), 2006GL025925. https://doi.org/10.1029/2006GL025925
    
    :param beta: Plasma beta data in shape (time)
    :type beta: astropy.units.Quantity or np.ndarray
    :param S: S parameter
    :type S: float
    :param alpha: Alpha parameter
    :type alpha: float
    :param beta_0: Beta_0 parameter
    :type beta_0: float
    :return: Temperature anisotropy
    :rtype: astropy.units.Quantity or np.ndarray or float
    '''
    return 1 + S / (beta + beta_0) ** alpha


def fitted_instability(beta: u.Quantity | np.ndarray, gamma_max: Literal['1e-4', '1e-3', '1e-2']) -> dict:
    '''Calculate fitted instability.
    Verscharen, D., Klein, K. G., & Maruca, B. A. (2019). The multi-scale nature of the solar wind. Living Reviews in Solar Physics, 16(1), 5. https://doi.org/10.1007/s41116-019-0021-0
    
    :param beta: Plasma beta data in shape (time)
    :type beta: astropy.units.Quantity or np.ndarray
    :param gamma_max: Maximum growth rate in unit of proton gyrofrequency
    :type gamma_max: str
    
    :return: Fitted temperature anisotropy
    :rtype: dict or astropy.units.Quantity or np.ndarray
    '''
    
    if isinstance(beta, u.Quantity) and not beta.unit.is_equivalent(u.dimensionless_unscaled):
        raise TypeError("beta must be a dimensionless quantity or a numpy array.")
    
    params = {}
    if gamma_max == '1e-2':
        params['Ion-Cyclotron'] = {'S': 0.649, 'alpha': 0.400, 'beta_0': -0.000}
        params['Mirror-Mode'] = {'S': 1.040, 'alpha': 0.633, 'beta_0': 0.012}
        params['Parallel-Firehose'] = {'S': -0.647, 'alpha': 0.583, 'beta_0': -0.713}
        params['Oblique-Firehose'] = {'S': -1.447, 'alpha': 1.000, 'beta_0': 0.148}
        
    elif gamma_max == '1e-3':
        params['Ion-Cyclotron'] = {'S': 0.437, 'alpha': 0.428, 'beta_0': 0.003}
        params['Mirror-Mode'] = {'S': 0.801, 'alpha': 0.763, 'beta_0': 0.063}
        params['Parallel-Firehose'] = {'S': -0.497, 'alpha': 0.566, 'beta_0': -0.543}
        params['Oblique-Firehose'] = {'S': -1.390, 'alpha': 1.005, 'beta_0': 0.111}
        
    elif gamma_max == '1e-4':
        params['Ion-Cyclotron'] = {'S': 0.367, 'alpha': 0.364, 'beta_0': -0.011}
        params['Mirror-Mode'] = {'S': 0.702, 'alpha': 0.674, 'beta_0': 0.009}
        params['Parallel-Firehose'] = {'S': -0.408, 'alpha': 0.529, 'beta_0': -0.410}
        params['Oblique-Firehose'] = {'S': -1.454, 'alpha': 1.023, 'beta_0': 0.178}

    results = {}
    for key in params:
        p = params[key]
        results[key] = instability_func(beta, p['S'], p['alpha'], p['beta_0'])
        
    return results
    




# if __name__ == "__main__":
#     # Test data
#     from datetime import timedelta
#     p_date = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
#     n = (np.random.rand(10)*10+10) * u.m**-3
#     b_date = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
#     b = np.random.rand(10, 3)*20 * u.T
#     T = (np.random.rand(10)*10000+100000) * u.K
    
#     print(calc_beta(p_date, n, b_date, b, T))