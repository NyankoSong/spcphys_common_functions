'''Transformations between thermal velocity, energy, and temperature.'''

from typing import Tuple
import warnings
import numpy as np
from astropy import units as u
from astropy.constants import k_B, m_p


def T_tensor_to_T(
    T_tensor: u.Quantity,
    b: u.Quantity,
    replace_negative: bool = True
    ) -> Tuple[u.Quantity, u.Quantity]:
    '''Calculate parallel and perpendicular components of temperature.
    
    :param T_tensor: Energy tensor with units, shape (n, 3), (n, 6) or (n, 3, 3)
    :type T_tensor: astropy.units.Quantity
    :param b: Magnetic field vector with units, shape (n, 3)
    :type b: astropy.units.Quantity
    :param replace_negative: Whether to replace negative temperatures with np.nan, defaults to True
    :type replace_negative: bool, optional
    :return: Parallel and perpendicular components of temperature
    :rtype: Tuple[astropy.units.Quantity, astropy.units.Quantity]
    '''
    if not T_tensor.unit.is_equivalent(u.J) or not T_tensor.unit.is_equivalent(u.K):
        raise ValueError("Energy tensor T_tensor must have units of energy (u.J) and temperature (u.K)")
    if not b.unit.is_equivalent(u.T):
        raise ValueError("Magnetic field b must have units of magnetic field (u.T)")
    
    shape_error = False
    if len(T_tensor.shape) == 2:
        T_tensor_mat = np.zeros((T_tensor.shape[0], 3, 3)) * u.J
        if T_tensor.shape[1] == 3:
            warnings.warn('Assuming T_tensor is Txx, Tyy, and Tzz of the temperature tensor and Txy=Txz=Tyz=0.', UserWarning)
            T_tensor_mat[:, 0, 0] = T_tensor[:, 0]
            T_tensor_mat[:, 1, 1] = T_tensor[:, 1]
            T_tensor_mat[:, 2, 2] = T_tensor[:, 2]
        elif T_tensor.shape[1] == 6:
            warnings.warn('Assuming T_tensor is Txx, Tyy, Tzz, Txy, Txz, and Tyz of the temperature tensor.', UserWarning)
            T_tensor_mat[:, 0, 0] = T_tensor[:, 0]
            T_tensor_mat[:, 1, 1] = T_tensor[:, 1]
            T_tensor_mat[:, 2, 2] = T_tensor[:, 2]
            T_tensor_mat[:, 0, 1] = T_tensor[:, 3]
            T_tensor_mat[:, 1, 0] = T_tensor[:, 3]
            T_tensor_mat[:, 0, 2] = T_tensor[:, 4]
            T_tensor_mat[:, 2, 0] = T_tensor[:, 4]
            T_tensor_mat[:, 1, 2] = T_tensor[:, 5]
            T_tensor_mat[:, 2, 1] = T_tensor[:, 5]
        else:
            shape_error = True
    elif len(T_tensor.shape) == 3 and T_tensor.shape[1] == 3 and T_tensor.shape[2] == 3:
        T_tensor_mat = T_tensor
    else:
        shape_error = True

    if shape_error:
        raise ValueError("T_tensor must have shape (n, 3), (n, 6) or (n, 3, 3) for 3D tensor components.")
    
    b_unit_vector = b / np.linalg.norm(b, axis=1, keepdims=True)
    T_para = np.einsum('ki,kij,kj->k', b_unit_vector, T_tensor_mat, b_unit_vector)
    T_perp = (np.trace(T_tensor_mat, axis1=1, axis2=2) - T_para) / 2

    if replace_negative:
        T_para = np.where(T_para <= 0, np.nan, T_para)
        T_perp = np.where(T_perp <= 0, np.nan, T_perp)

    return T_para, T_perp


def T_to_vth(T: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''Calculate thermal velocity from temperature.
    
    :param T: Temperature with units
    :type T: astropy.units.Quantity
    :param mass: Mass with units, defaults to proton mass
    :type mass: astropy.units.Quantity, optional
    :param n: Factor determining the type of thermal velocity, defaults to 2
    :type n: int or float, optional
              n=2 for most probable speed,
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation)
    :return: Thermal velocity with units
    :rtype: astropy.units.Quantity
    '''
    
    if not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of temperature (u.K)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    T = T.si
    mass = mass.si
    vth = (n * k_B * T / mass)**0.5
    return vth.si



def vth_to_T(vth: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''Calculate temperature from thermal velocity.
    
    :param vth: Thermal velocity with units
    :type vth: astropy.units.Quantity
    :param mass: Mass with units, defaults to proton mass
    :type mass: astropy.units.Quantity, optional
    :param n: Factor determining the type of thermal velocity, defaults to 2
    :type n: int or float, optional
              n=2 for most probable speed,
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation)
    :return: Temperature with units
    :rtype: astropy.units.Quantity
    '''
    
    if not vth.unit.is_equivalent(u.m / u.s):
        raise ValueError("Thermal velocity vth must have units of velocity (u.m/u.s)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    vth = vth.si
    mass = mass.si
    T = mass * vth**2 / (n * k_B)
    return T.si



def E_to_T(E: u.Quantity) -> u.Quantity:
    '''Calculate temperature from energy.
    
    :param E: Energy with units
    :type E: astropy.units.Quantity
    :return: Temperature with units
    :rtype: astropy.units.Quantity
    '''
    
    if not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of energy (u.J)")
    
    E = E.si
    T = E / k_B
    return T.si



def T_to_E(T: u.Quantity) -> u.Quantity:
    '''Calculate energy from temperature.
    
    :param T: Temperature with units
    :type T: astropy.units.Quantity
    :return: Energy with units
    :rtype: astropy.units.Quantity
    '''
    
    if not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of temperature (u.K)")
    
    T = T.si
    E = k_B * T
    return E.si



def E_to_vth(E: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''Calculate thermal velocity from energy.
    
    :param E: Energy with units
    :type E: astropy.units.Quantity
    :param mass: Mass with units, defaults to proton mass
    :type mass: astropy.units.Quantity, optional
    :param n: Factor determining the type of thermal velocity, defaults to 2
    :type n: int or float, optional
              n=2 for most probable speed,
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation)
    :return: Thermal velocity with units
    :rtype: astropy.units.Quantity
    '''
    
    if not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of energy (u.J)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    E = E.si
    mass = mass.si
    vth = (2 * E / mass)**0.5
    return vth.si



def vth_to_E(vth: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''Calculate energy from thermal velocity.
    
    :param vth: Thermal velocity with units
    :type vth: astropy.units.Quantity
    :param mass: Mass with units, defaults to proton mass
    :type mass: astropy.units.Quantity, optional
    :param n: Factor determining the type of thermal velocity, defaults to 2
    :type n: int or float, optional
              n=2 for most probable speed,
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation)
    :return: Energy with units
    :rtype: astropy.units.Quantity
    '''
    
    if not vth.unit.is_equivalent(u.m / u.s):
        raise ValueError("Thermal velocity vth must have units of velocity (u.m/u.s)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    vth = vth.si
    mass = mass.si
    E = 0.5 * mass * vth**2
    return E.si




# if __name__ == "__main__":
#     T = 1e6 * u.K  # Temperature
#     vth = 1e5 * u.m / u.s  # Thermal velocity
#     E = 1e-17 * u.J  # Energy

#     # Test T_to_vth
#     vth_calculated = T_to_vth(T)
#     print(f"T_to_vth({T}) = {vth_calculated}")

#     # Test vth_to_T
#     T_calculated = vth_to_T(vth)
#     print(f"vth_to_T({vth}) = {T_calculated}")

#     # Test E_to_T
#     T_from_E = E_to_T(E)
#     print(f"E_to_T({E}) = {T_from_E}")
    
#     # Test T_to_E
#     E_from_T = T_to_E(T)
#     print(f"T_to_E({T}) = {E_from_T}")

#     # Test E_to_vth
#     vth_from_E = E_to_vth(E)
#     print(f"E_to_vth({E}) = {vth_from_E}")

#     # Test vth_to_E
#     E_from_vth = vth_to_E(vth)
#     print(f"vth_to_E({vth}) = {E_from_vth}")
    
#     pass