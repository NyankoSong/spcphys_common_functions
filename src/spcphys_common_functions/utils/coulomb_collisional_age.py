import numpy as np
from astropy.constants import e, m_p, au, eps0, u as u_const
from scipy import integrate
from astropy import units as u

from . import config
from .utils import check_parameters
from .vth_E_T import T_to_E, T_to_vth


@check_parameters
def calc_Ac(v_j: u.Quantity,
            T_j: u.Quantity,
            n_j: u.Quantity,
            v_i: u.Quantity,
            T_i: u.Quantity,
            n_i: u.Quantity,
            charge_number_i: int,
            mass_number_i: int,
            m_i: u.Quantity|None = None,
            charge_number_j: int = 1,
            mass_number_j: int = 1,
            m_j: u.Quantity|None = None,
            distance: u.Quantity = au,
            ) -> float:

    '''
    Calculate Coulomb collisional age, reproduced according to the algorithm of Tracy et al. (2015)
    
    :param v_j: Field particle velocity, m/s
    :param T_j: Field particle Temperature, K
    :param n_j: Field particle number density, m^-3
    :param v_i: Test particle velocity, m/s
    :param T_i: Test particle Temperature, K
    :param n_i: Test particle number density, m^-3
    :param charge_number_i: Test particle charge number
    :param mass_number_i: Test particle mass number
    :param m_i: Test particle mass, kg, default None, calculated from mass_number_i with atomic mass constant
    :param charge_number_j: Field particle charge number, default 1, proton
    :param mass_number_j: Field particle mass number, default 1, proton
    :param m_j: Field particle mass, kg, default None, calculated from mass_number_j with atomic mass constant
    :param distance: heliocentric distance, AU, default 1 AU
    
    :return Ac: Coulomb Collisional Age
    '''
    
    if not all(x.unit.is_equivalent(u.m/u.s) for x in [v_j, v_i]):
        raise ValueError("v_j, vth_j, v_i, and vth_i must be quantities with units of velocity (m/s).")
    if not all(x.unit.is_equivalent(u.K) for x in [T_j, T_i]):
        raise ValueError("T_j and T_i must be quantities with units of temperature (K).")
    if not all(x.unit.is_equivalent(u.m**-3) for x in [n_j, n_i]):
        raise ValueError("n_j and n_i must be quantities with units of number density (m^-3).")
    if not distance.unit.is_equivalent(u.au):
        raise ValueError("distance must be a quantity with units of astronomical units (AU).")
    if not all(x > 0 for x in [charge_number_i, mass_number_i, charge_number_j, mass_number_j]):
        raise ValueError("charge_number_i, mass_number_i, charge_number_j, mass_number_j, and n_vth must be positive integers.")
    if m_i is not None and (not m_i.unit.is_equivalent(u.kg) or m_i <= 0):
        raise ValueError("m_i must be a positive quantity with units of mass (kg) or None.")
    if m_j is not None and (not m_j.unit.is_equivalent(u.kg) or m_j <= 0):
        raise ValueError("m_j must be a positive quantity with units of mass (kg) or None.")
    
    v_j = v_j.si
    v_i = v_i.si
    T_j = T_j.si
    T_i = T_i.si
    n_j = n_j.si
    n_i = n_i.si
    distance = distance.si
    if m_i is not None:
        m_i = m_i.si
    if m_j is not None:
        m_j = m_j.si
    
    q_e = e.si
    q_j = - charge_number_j * q_e
    q_i = - charge_number_i * q_e
    
    mass_number_to_mass = lambda m: m_p if m == 1 else m * u_const
    if m_i is None:
        m_i = mass_number_to_mass(mass_number_i)
    if m_j is None:
        m_j = mass_number_to_mass(mass_number_j)
    
    n_vth = 2 # most probable speed
    Te_j = T_to_E(T_j).to(u.eV) # in eV
    Te_i = T_to_E(T_i).to(u.eV) # in eV
    vth2_j = T_to_vth(T_j, mass=m_j, n=n_vth)**2
    vth2_i = T_to_vth(T_i, mass=m_i, n=n_vth)**2
    
    ln_lambda = 29.9 - np.log(((charge_number_i*charge_number_j*(mass_number_i + mass_number_j) / (mass_number_i*Te_j + mass_number_j*Te_i)) * np.sqrt(n_i*charge_number_i**2 / Te_i + n_j*charge_number_j**2 / Te_j)).to_value()) # 无量纲化
    x = np.abs(v_i - v_j) / np.sqrt(vth2_i + vth2_j)
    phi_x = np.array([(2 / np.sqrt(np.pi)) * integrate.quad(lambda z: np.exp(-z**2), 0, xi)[0] for xi in x]) if x.size > 1 else (2 / np.sqrt(np.pi)) * integrate.quad(lambda z: np.exp(-z**2), 0, x)[0]
    nu_th = ((1 / (3*np.pi*eps0**2)) * (q_i**2 * q_j**2 * ln_lambda * n_j / (m_i * m_j * (vth2_i + vth2_j)**(3/2))) * (phi_x / x))
    t_travel = distance / v_j
    Ac = nu_th * t_travel
    
    return Ac.si.to_value()


# if __name__ == "__main__":
#     # Example parameters for oxygen ion (O) and proton (H)
#     v_j = 400 * u.km / u.s
#     T_j = 1e6 * u.K
#     n_j = 20 / u.cm**3
#     v_i = 500 * u.km / u.s
#     T_i = 1e7 * u.K
#     n_i = 0.02 / u.cm**3
#     charge_number_i = 6  # Oxygen ion
#     mass_number_i = 16  # Oxygen ion
#     # charge_number_j = 1  # Proton, default field particle
#     # mass_number_j = 1  # Proton, default field particle
#     distance = 1 * au

#     # Call the function
#     result = calc_Ac(v_j, T_j, n_j, v_i, T_i, n_i, charge_number_i, mass_number_i, distance=distance, 
#                                     #  charge_number_j=charge_number_j, mass_number_j=mass_number_j
#                                      )

#     print(f'v_j = {v_j}, T_j = {T_j}, n_j = {n_j}, v_i = {v_i}, T_i = {T_i}, n_i = {n_i}, charge_number_i = {charge_number_i}, mass_number_i = {mass_number_i}, distance = {distance}')
#     print(f'Ac = {result}')
    
#     pass