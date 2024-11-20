'''Transformations between thermal velocity, energy, and temperature.'''

from astropy import units as u
from astropy.constants import k_B, m_p

from ..utils.utils import check_parameters


@check_parameters
def T_to_vth(T: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''
    Calculate thermal velocity from temperature.
    
    :param T: Temperature with units (e.g., T * u.K).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
              
    :return vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    '''
    
    if not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of temperature (u.K)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    T = T.si
    mass = mass.si
    vth = (n * k_B * T / mass)**0.5
    return vth.si


@check_parameters
def vth_to_T(vth: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''
    Calculate temperature from thermal velocity.
    
    :param vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
              
    :return T: Temperature with units (e.g., T * u.K).
    '''
    
    if not vth.unit.is_equivalent(u.m / u.s):
        raise ValueError("Thermal velocity vth must have units of velocity (u.m/u.s)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    vth = vth.si
    mass = mass.si
    T = mass * vth**2 / (n * k_B)
    return T.si


@check_parameters
def E_to_T(E: u.Quantity) -> u.Quantity:
    '''
    Calculate temperature from energy.
    
    :param E: Energy with units (e.g., E * u.J).
    
    :return T: Temperature with units (e.g., T * u.K).
    '''
    
    if not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of energy (u.J)")
    
    E = E.si
    T = E / k_B
    return T.si


@check_parameters
def T_to_E(T: u.Quantity) -> u.Quantity:
    '''
    Calculate energy from temperature.
    
    :param T: Temperature with units (e.g., T * u.K).
    
    :return E: Energy with units (e.g., E * u.J).
    '''
    
    if not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of temperature (u.K)")
    
    T = T.si
    E = k_B * T
    return E.si


@check_parameters
def E_to_vth(E: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''
    Calculate thermal velocity from energy.
    
    :param E: Energy with units (e.g., E * u.J).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
              
    :return vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    '''
    
    if not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of energy (u.J)")
    if not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of mass (u.kg)")
    
    E = E.si
    mass = mass.si
    vth = (2 * E / mass)**0.5
    return vth.si


@check_parameters
def vth_to_E(vth: u.Quantity, mass: u.Quantity=m_p, n: int|float=2) -> u.Quantity:
    '''
    Calculate energy from thermal velocity.
    
    :param vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
              
    :return E: Energy with units (e.g., E * u.J).
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