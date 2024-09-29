import numpy as np
from astropy import units as u
from astropy.constants import k_B, m_p

def T_to_vth(T, mass=m_p, n=2):
    '''
    Calculate thermal velocity from temperature.
    
    :param T: Temperature with units (e.g., T * u.K).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
    :return: Thermal velocity with units (e.g., vth * u.m/u.s).
    '''
    if not isinstance(T, u.Quantity) or not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of Kelvin (u.K)")
    if not isinstance(mass, u.Quantity) or not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of kilogram (u.kg)")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a number")
    
    T = T.to(u.K)
    mass = mass.to(u.kg)
    vth = (n * k_B * T / mass)**0.5
    return vth.to(u.m / u.s)


def vth_to_T(vth, mass=m_p, n=2):
    '''
    Calculate temperature from thermal velocity.
    
    :param vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
    :return: Temperature with units (e.g., T * u.K).
    '''
    if not isinstance(vth, u.Quantity) or not vth.unit.is_equivalent(u.m / u.s):
        raise ValueError("Thermal velocity vth must have units of meters per second (u.m/u.s)")
    if not isinstance(mass, u.Quantity) or not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of kilogram (u.kg)")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a number")
    
    vth = vth.to(u.m / u.s)
    mass = mass.to(u.kg)
    T = mass * vth**2 / (n * k_B)
    return T.to(u.K)


def E_to_T(E):
    '''
    Calculate temperature from energy.
    
    :param E: Energy with units (e.g., E * u.J).
    :return: Temperature with units (e.g., T * u.K).
    '''
    if not isinstance(E, u.Quantity) or not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of Joules (u.J)")
    
    E = E.to(u.J)
    T = E / k_B
    return T.to(u.K)


def T_to_E(T):
    '''
    Calculate energy from temperature.
    
    :param T: Temperature with units (e.g., T * u.K).
    :return: Energy with units (e.g., E * u.J).
    '''
    if not isinstance(T, u.Quantity) or not T.unit.is_equivalent(u.K):
        raise ValueError("Temperature T must have units of Kelvin (u.K)")
    
    T = T.to(u.K)
    E = k_B * T
    return E.to(u.J)


def E_to_vth(E, mass=m_p, n=2):
    '''
    Calculate thermal velocity from energy.
    
    :param E: Energy with units (e.g., E * u.J).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
    :return: Thermal velocity with units (e.g., vth * u.m/u.s).
    '''
    if not isinstance(E, u.Quantity) or not E.unit.is_equivalent(u.J):
        raise ValueError("Energy E must have units of Joules (u.J)")
    if not isinstance(mass, u.Quantity) or not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of kilogram (u.kg)")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a number")
    
    E = E.to(u.J)
    mass = mass.to(u.kg)
    vth = (2 * E / mass)**0.5
    return vth.to(u.m / u.s)

def vth_to_E(vth, mass=m_p, n=2):
    '''
    Calculate energy from thermal velocity.
    
    :param vth: Thermal velocity with units (e.g., vth * u.m/u.s).
    :param mass: Mass with units (e.g., mass * u.kg). Default is proton mass.
    :param n: Factor determining the type of thermal velocity.
              n=2 for most probable speed (Default),
              n=3 for root mean square speed,
              n=8/π for mean speed,
              n=1 for one-dimensional root mean square speed (for single direction temperature calculation).
    :return: Energy with units (e.g., E * u.J).
    '''
    if not isinstance(vth, u.Quantity) or not vth.unit.is_equivalent(u.m / u.s):
        raise ValueError("Thermal velocity vth must have units of meters per second (u.m/u.s)")
    if not isinstance(mass, u.Quantity) or not mass.unit.is_equivalent(u.kg):
        raise ValueError("Mass must have units of kilogram (u.kg)")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a number")
    
    vth = vth.to(u.m / u.s)
    mass = mass.to(u.kg)
    E = 0.5 * mass * vth**2
    return E.to(u.J)




if __name__ == "__main__":
    T = 1e6 * u.K  # Temperature
    vth = 1e5 * u.m / u.s  # Thermal velocity
    E = 1e-17 * u.J  # Energy

    # Test T_to_vth
    vth_calculated = T_to_vth(T)
    print(f"T_to_vth({T}) = {vth_calculated}")

    # Test vth_to_T
    T_calculated = vth_to_T(vth)
    print(f"vth_to_T({vth}) = {T_calculated}")

    # Test E_to_T
    T_from_E = E_to_T(E)
    print(f"E_to_T({E}) = {T_from_E}")
    
    # Test T_to_E
    E_from_T = T_to_E(T)
    print(f"T_to_E({T}) = {E_from_T}")

    # Test E_to_vth
    vth_from_E = E_to_vth(E)
    print(f"E_to_vth({E}) = {vth_from_E}")

    # Test vth_to_E
    E_from_vth = vth_to_E(vth)
    print(f"vth_to_E({vth}) = {E_from_vth}")
    
    pass