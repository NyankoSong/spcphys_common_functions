'''
TRACY P J, KASPER J C, ZURBUCHEN T H, et al., 2015. THERMALIZATION OF HEAVY IONS IN THE SOLAR WIND[J/OL]. The Astrophysical Journal, 812(2): 170. DOI:10.1088/0004-637X/812/2/170.
Hellinger, P. (2016). ION COLLISIONAL TRANSPORT COEFFICIENTS IN THE SOLAR WIND AT 1 au. The Astrophysical Journal, 825(2), 120. https://doi.org/10.3847/0004-637X/825/2/120
'''

import warnings
from typing import TypedDict, Required, NotRequired
import numpy as np
from astropy.constants import e, m_p, au, eps0, u as u_const
from scipy.integrate import quad
from scipy.special import erf, hyp2f1, gamma
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u


from .vth_E_T import T_to_vth


def coloumb_logarithm(
    charge_number_j: int,
    mass_number_j: int,
    n_j: u.Quantity,
    T_j: u.Quantity,
    charge_number_i: int | None = None,
    mass_number_i: int | None = None,
    n_i: u.Quantity | None = None,
    T_i: u.Quantity | None = None,
    self_collision: bool = False
) -> float:
    '''
    Calculate the Coulomb logarithm for given plasma parameters.
    
    Hellinger (2016) Eq. 14 / Tracy (2015) Eq. 2:
        ln Λ_st = 29.9 - ln[(Z_s Z_t (A_s + A_t))/(A_s T̃_t + A_t T̃_s) × sqrt(n_s Z_s²/T̃_s + n_t Z_t²/T̃_t)]
    
    where T̃ denotes temperature in electronvolts (eV).
    
    The Coulomb logarithm characterizes the ratio of maximum to minimum impact parameters
    in Coulomb collisions. Typical values in the solar wind are 15-25.
    
    :param charge_number_j: Field particle charge number (Z_j)
    :param mass_number_j: Field particle mass number (A_j)
    :param n_j: Field particle number density, m^-3
    :param T_j: Field particle temperature, K or J
    :param charge_number_i: Test particle charge number (Z_i)
    :param mass_number_i: Test particle mass number (A_i)
    :param n_i: Test particle number density, m^-3
    :param T_i: Test particle temperature, K or J
    :param self_collision: If True, use one-species self-collision mode and only provide field-particle parameters.
    
    :return ln_lambda: Coulomb logarithm (dimensionless)
    '''
    if self_collision:
        if not all(x is None for x in [T_i, n_i, charge_number_i, mass_number_i]):
            raise ValueError("For self-collision, only field particle parameters should be provided.")
        T_i = T_j
        n_i = n_j
        charge_number_i = charge_number_j
        mass_number_i = mass_number_j
        
    if not all(x.unit.is_equivalent(u.K) or x.unit.is_equivalent(u.J) for x in [T_j, T_i]):
        raise ValueError("T_j and T_i must be quantities with units of temperature (K) or energy (J).")
    if not all(x.unit.is_equivalent(u.m**-3) for x in [n_j, n_i]):
        raise ValueError("n_j and n_i must be quantities with units of number density (m^-3).")
    
    # Convert temperatures to eV as required by the formula
    Te_j = T_j.to(u.eV, equivalencies=u.temperature_energy())
    Te_i = T_i.to(u.eV, equivalencies=u.temperature_energy())
    
    # Tracy (2015) Eq. 2: Coulomb logarithm
    ln_lambda = 29.9 - np.log(((charge_number_i*charge_number_j*(mass_number_i + mass_number_j) / (mass_number_i*Te_j + mass_number_j*Te_i)) * np.sqrt(n_i*charge_number_i**2 / Te_i + n_j*charge_number_j**2 / Te_j)).to_value())
    
    return ln_lambda

def calc_Ac(v_j: u.Quantity,
            T_j: u.Quantity,
            n_j: u.Quantity,
            v_i: u.Quantity|None = None,
            T_i: u.Quantity|None = None,
            n_i: u.Quantity|None = None,
            charge_number_i: int|None = None,
            mass_number_i: int|None = None,
            m_i: u.Quantity|None = None,
            charge_number_j: int = 1,
            mass_number_j: int = 1,
            m_j: u.Quantity|None = m_p,
            distance: u.Quantity = au,
            x_assumption: u.Quantity|float|None = None,
            self_collision: bool = False
            ) -> u.Quantity:

    '''
    Calculate Coulomb collisional age according to Tracy et al. (2015).
    
    Tracy (2015) Eq. 6: Collisional age definition
        A_c = ν_th,ij × R / V_SW
    
    where ν_th,ij is the thermalization rate and R/V_SW is the solar wind transit time.
    A_c represents the cumulative effect of Coulomb collisions during propagation.
    A_c >> 1 indicates strong collisional coupling; A_c << 1 indicates collisionless regime.
    
    Tracy (2015) Eq. 1: Thermalization rate
        ν_th,ij = [q_i² q_j² ln Λ n_j / (3πε₀² m_i m_j (v_th,i² + v_th,j²)^(3/2))] × φ(x)/x
    
    Tracy (2015) Eq. 3: Normalized differential streaming
        x = |U_i - U_j| / sqrt(v_th,i² + v_th,j²)
    
    Tracy (2015) Eq. 4: Error function correction
        φ(x) = erf(x)
    
    :param v_j: Field particle velocity, m/s
    :param T_j: Field particle Temperature, K
    :param n_j: Field particle number density, m^-3
    :param v_i: Test particle velocity, m/s, default None. Required when self_collision is False.
    :param T_i: Test particle temperature, K or J, default None. Required when self_collision is False.
    :param n_i: Test particle number density, m^-3, default None. Required when self_collision is False.
    :param charge_number_i: Test particle charge number, default None. Required when self_collision is False.
    :param mass_number_i: Test particle mass number, default None. Required when self_collision is False.
    :param m_i: Test particle mass, kg, default None. If None, it is calculated from mass_number_i with atomic mass constant.
    :param charge_number_j: Field particle charge number, default 1, proton
    :param mass_number_j: Field particle mass number, default 1, proton
    :param m_j: Field particle mass, kg, default m_p (proton mass). If set to None, it is calculated from mass_number_j with atomic mass constant.
    :param distance: heliocentric distance, AU, default 1 AU
    :param x_assumption: Assumed normalized differential streaming value. Tracy et al. (2015) suggest 0.5. Default None, in which case it is calculated from velocities and thermal speeds.
    :param self_collision: If True, use one-species self-collision mode and only provide field-particle parameters.
    
    :return Ac: Coulomb Collisional Age (dimensionless)
    '''
    
    if self_collision:
        if not all(x is None for x in [v_i, T_i, n_i, charge_number_i, mass_number_i]):
            raise ValueError("For self-collision, only field particle parameters should be provided.")
        v_i = v_j
        T_i = T_j
        n_i = n_j
        charge_number_i = charge_number_j
        mass_number_i = mass_number_j
        m_i = m_j
    else:
        if not all(x is not None for x in [v_i, T_i, n_i, charge_number_i, mass_number_i]):
            raise ValueError("v_i, T_i, n_i, charge_number_i, and mass_number_i are required when self_collision is False.")

    mass_number_to_mass = lambda m: m_p if m == 1 else m * u_const
    if m_i is None:
        m_i = mass_number_to_mass(mass_number_i)
    if m_j is None:
        m_j = mass_number_to_mass(mass_number_j)

    species_j: SpeciesParamsTracy = {
        'charge': charge_number_j * e.si,
        'mass': m_j,
        'n': n_j,
        'v': v_j,
        'T': T_j,
        'distance': distance,
    }
    species_i_comp: SpeciesParamsTracy | None
    if self_collision:
        species_i_comp = None
    else:
        species_i_comp = {
            'charge': charge_number_i * e.si,
            'mass': m_i,
            'n': n_i,
            'v': v_i,
            'T': T_i,
            'distance': distance,
        }

    return calc_collisional_age(
        species_j=species_j,
        species_i=species_i_comp,
        x_assumption=x_assumption,
        self_collision=self_collision,
    )


# ==================== Type Definitions ====================

class SpeciesParamsBase(TypedDict, total=False):
    """
    Parameters for a plasma species used in Hellinger 2016 collision calculations.
    
    All quantities should have appropriate astropy units.
    """
    charge: Required[u.Quantity]
    """Particle charge with units of C"""
    mass: Required[u.Quantity]
    """Particle mass with units of kg"""
    n: Required[u.Quantity]
    """Number density with units of m^-3"""
    
class SpeciesParamsComp(SpeciesParamsBase):
    """
    Extended species parameters with optional fields for more flexible input.
    """
    v_drift: Required[u.Quantity]
    """Parallel drift velocity with units of m/s"""
    T_para: Required[u.Quantity]
    """Parallel temperature with units of K or J"""
    T_perp: Required[u.Quantity]
    """Perpendicular temperature with units of K or J"""
    T: NotRequired[u.Quantity]
    """Mean temperature with units of K or J"""
    
class SpeciesParamsTracy(SpeciesParamsBase):
    """
    Species parameters specifically for Tracy (2015) collisional age calculations.
    """
    v: Required[u.Quantity]
    """Bulk velocity with units of m/s"""
    T: Required[u.Quantity]
    """Temperature with units of K or J"""
    distance: Required[u.Quantity]
    """Heliocentric distance with units of m"""
    

# ==================== Heating Rate Functions (Hellinger 2016) ====================

def _coloumb_logarithm_hellinger(
    species_s: SpeciesParamsComp | SpeciesParamsTracy,
    species_t: SpeciesParamsComp | SpeciesParamsTracy | None = None,
    self_collision: bool = False
) -> float:
    '''
    Calculate the Coulomb logarithm for Hellinger 2016 collision calculations, 
    which is equivalent to Tracy (2015) Eq. 2 but without dimensionless charge and mass numbers.
    
    Hellinger (2016) Eq. 14:
        ln Λ_st = 29.9 - ln[(q_s q_t (m_s + m_t)) / (e³ (m_s T̃_t + m_t T̃_s)) 
                           × sqrt(n_s q_s²/T̃_s + n_t q_t²/T̃_t)]
    
    where T̃ denotes temperature in electronvolts (eV), q is charge, m is mass.
    
    :param species_s: Species s parameters dictionary
    :param species_t: Species t parameters dictionary (or None for self-collision)
    :param self_collision: If True, use species_s for both s and t
    :return: Coulomb logarithm (dimensionless float)
    '''
    q_s = species_s['charge'].si
    m_s = species_s['mass'].si
    n_s = species_s['n'].si
    if 'T' in species_s:
        T_s = species_s['T'].to(u.eV, equivalencies=u.temperature_energy())
    else:
        T_s = ((species_s['T_para'] + 2 * species_s['T_perp']) / 3).to(u.eV, equivalencies=u.temperature_energy())
    
    if self_collision:
        q_t, m_t, n_t, T_t = q_s, m_s, n_s, T_s
    else:
        if species_t is None:
            raise ValueError("species_t must be provided for two-species collision")
        q_t = species_t['charge'].si
        m_t = species_t['mass'].si
        n_t = species_t['n'].si
        if 'T' in species_t:
            T_t = species_t['T'].to(u.eV, equivalencies=u.temperature_energy())
        else:
            T_t = ((species_t['T_para'] + 2 * species_t['T_perp']) / 3).to(u.eV, equivalencies=u.temperature_energy())
    
    # Hellinger (2016) Eq. 14:
    # ln Λ_st = 29.9 - ln[(q_s q_t (m_s + m_t)) / (e³ (m_s T̃_t + m_t T̃_s)) × sqrt(n_s q_s²/T̃_s + n_t q_t²/T̃_t)]
    numerator = q_s * q_t * (m_s + m_t)
    denominator = e.si**3 * (m_s * T_t + m_t * T_s)
    sqrt_term = np.sqrt(n_s * q_s**2 / T_s + n_t * q_t**2 / T_t)
    
    ln_Lambda = 29.9 - np.log((numerator / denominator * sqrt_term).value)
    
    return ln_Lambda


def calc_collisional_age(
    species_j: SpeciesParamsTracy,
    species_i: SpeciesParamsTracy | None = None,
    x_assumption: u.Quantity | float | None = None,
    self_collision: bool = False
) -> u.Quantity:
    '''
    Calculate Coulomb collisional age according to Tracy et al. (2015).

    Tracy (2015) Eq. 6: Collisional age definition
        A_c = ν_th,ij × R / V_SW

    Tracy (2015) Eq. 1: Thermalization rate
        ν_th,ij = [q_i² q_j² ln Λ n_j / (3πε₀² m_i m_j (v_th,i² + v_th,j²)^(3/2))] × φ(x)/x

    Tracy (2015) Eq. 3: Normalized differential streaming
        x = |U_i - U_j| / sqrt(v_th,i² + v_th,j²)

    Tracy (2015) Eq. 4: Error function correction
        φ(x) = erf(x)

    :param species_j: Field-particle species dictionary with keys charge, mass, n, v, T, distance.
    :param species_i: Test-particle species dictionary with keys charge, mass, n, v, T, distance.
                      It can be omitted when self_collision=True.
    :param x_assumption: Optional assumed normalized differential streaming value.
    :param self_collision: If True, use one-species self-collision mode and only provide species_j.
    :return: Coulomb collisional age (dimensionless)
    '''
    if self_collision:
        if species_i is not None:
            raise ValueError("For self-collision, only species_j should be provided.")
        species_i = species_j
    elif species_i is None:
        raise ValueError("species_i must be provided when self_collision is False.")

    q_j = species_j['charge'].si
    m_j = species_j['mass'].si
    n_j = species_j['n'].si
    v_j = species_j['v'].si
    T_j = species_j['T'].to(u.K, equivalencies=u.temperature_energy())
    distance = species_j['distance'].si

    q_i = species_i['charge'].si
    m_i = species_i['mass'].si
    # n_i = species_i['n'].si
    v_i = species_i['v'].si
    T_i = species_i['T'].to(u.K, equivalencies=u.temperature_energy())

    # Tracy (2015) Eq. 5: Most probable speed v_th = sqrt(2kT/m)
    n_vth = 2
    vth2_j = T_to_vth(T_j, mass=m_j, n=n_vth)**2
    vth2_i = T_to_vth(T_i, mass=m_i, n=n_vth)**2
    ln_lambda = _coloumb_logarithm_hellinger(
        species_s=species_j,
        species_t=None if self_collision else species_i,
        self_collision=self_collision,
    )

    # Tracy (2015) Eq. 3: Normalized differential streaming
    if x_assumption is not None:
        if isinstance(x_assumption, u.Quantity):
            if not x_assumption.unit.is_equivalent(u.dimensionless_unscaled):
                raise ValueError("x_assumption must be dimensionless when provided as a quantity.")
            x = x_assumption.to_value(u.dimensionless_unscaled)
        else:
            x = x_assumption
    else:
        x = np.abs(v_i - v_j) / np.sqrt(vth2_i + vth2_j)

    # Tracy (2015) Eq. 4: Error function correction φ(x)/x
    if self_collision:
        phi_x_over_x = 2 / np.sqrt(np.pi)
    else:
        phi_x_over_x = erf(x) / x

    # Tracy (2015) Eq. 1: Thermalization rate
    nu_th = ((1 / (3 * np.pi * eps0**2)) *
             (q_i**2 * q_j**2 * ln_lambda * n_j / (m_i * m_j * (vth2_i + vth2_j)**(3 / 2))) *
             phi_x_over_x)

    # Tracy (2015) Eq. 6: Collisional age = ν_th × transit_time
    t_travel = distance / v_j
    return (nu_th * t_travel).si


def _kampe_de_feriet(a: float, b: float, c: float, x: float, y: float) -> float:
    '''
    Calculate the generalized Kampé de Fériet double hypergeometric function F^{2··}_{1·1}.
    
    Hellinger (2016) Eq. 13 - Integral representation for b=d case:
        F^{2··}_{1·1}(a,b; c;b | x,y) = Γ(c)/[Γ(a)Γ(c-a)] × ∫₀¹ t^{a-1}(1-t)^{c-a-1}/(1-tx)^b × exp(ty/(1-tx)) dt
    
    This special function appears in the collisional transport coefficients
    for bi-Maxwellian distributions (Hellinger & Trávníček 2009).
    
    :param a: First parameter of the function
    :param b: Second parameter of the function  
    :param c: Third parameter of the function
    :param x: First argument (typically 1 - A_st, where A_st is combined temperature anisotropy)
    :param y: Second argument (typically A_st × v_st²/(4v_st∥²), drift-related term)
    :return: Value of the Kampé de Fériet function
    '''
    coef = gamma(c) / (gamma(a) * gamma(c - a))

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    def _integrate_scalar(xi: float, yi: float) -> float:
        def integrand(t):
            num = (t**(a - 1)) * ((1 - t)**(c - a - 1))
            den = (1 - t * xi)**b
            # Hellinger (2016) Eq. 13: exp(+ty/(1-tx)), NOT exp(-ty/(1-tx))
            exp_term = np.exp((yi * t) / (1 - t * xi))
            return num / den * exp_term

        integral, _ = quad(integrand, 0, 1)
        return coef * integral

    if x_arr.ndim == 0 and y_arr.ndim == 0:
        return _integrate_scalar(float(x_arr), float(y_arr))

    x_b, y_b = np.broadcast_arrays(x_arr, y_arr)
    out = np.empty_like(x_b, dtype=float)
    for idx in np.ndindex(x_b.shape):
        out[idx] = _integrate_scalar(float(x_b[idx]), float(y_b[idx]))
    return out


def _calc_F_st(
    a: float,
    b: float,
    c: float,
    A_st: float | np.ndarray,
    v_drift_ratio: float | np.ndarray,
) -> float | np.ndarray:
    '''
    Calculate F_abc^(st) from reduced dimensionless parameters.

    Hellinger (2016) Eq. 10:
        F_{abc}^(st) = exp(-v_st²/(4v_st∥²)) × F^{2··}_{1·1}(a,b; c;b | 1-A_st, A_st×v_st²/(4v_st∥²))
                     = exp(-v_drift_ratio²/4) × F^{2··}_{1·1}(a,b; c;b | 1-A_st, A_st×v_drift_ratio²/4)

    where v_drift_ratio = v_st / v_st∥.
    
    This function encapsulates the drift and anisotropy effects in the collisional
    transport coefficients for bi-Maxwellian distributions.
    
    :param a, b, c: Parameters of the Kampé de Fériet function
    :param A_st: Combined temperature anisotropy
    :param v_drift_ratio: Ratio of relative drift velocity to combined parallel thermal velocity
    :return: Value of F_abc^(st)
    '''
    x = 1 - A_st
    y = A_st * v_drift_ratio**2 / 4
    kampe_val = _kampe_de_feriet(a, b, c, x, y)
    exp_factor = np.exp(-(v_drift_ratio**2) / 4)
    return exp_factor * kampe_val


def build_F_abc_lookup_table(
    A_st_range: tuple[float, float] | None = None,
    v_drift_ratio_range: tuple[float, float] | None = None,
    n_A_st: int = 50,
    n_v_ratio: int = 50,
    species_s: SpeciesParamsComp | None = None,
    species_t: SpeciesParamsComp | None = None,
) -> dict:
    '''
    Build a lookup table for the three F_abc functions used in calc_heating_rates.
    
    This precomputes F_{1,1/2,5/2}, F_{2,1/2,5/2}, and F_{1,3/2,5/2} over a grid
    of (A_st, v_drift_ratio) values, where v_drift_ratio = v_st / v_st_para.
    
    The lookup table can significantly speed up calc_heating_rates when processing
    large datasets, as it avoids repeated numerical integration.
    
    :param A_st_range: Range of combined temperature anisotropy (T_perp/T_para).
                       If None and species_s/species_t are provided, auto-estimate from data.
                       The final range is clipped to the convergence domain |1-A_st| < 1.
    :param v_drift_ratio_range: Range of normalized drift velocity (v_st/v_st_para).
                                If None and species_s/species_t are provided, auto-estimate from data.
    :param n_A_st: Number of grid points for A_st
    :param n_v_ratio: Number of grid points for v_drift_ratio
    :param species_s: Parameters for species s
    :param species_t: Parameters for species t
    :return: Dictionary containing interpolators for each F_abc function
    '''  
    
    default_A_st_range = (1e-2, 2 - 1e-2)  # Default range for A_st to ensure |1-A_st| < 1
    default_v_drift_ratio_range = (0.0, 5.0)  # Default range for v_drift_ratio based on typical solar wind conditions
    
    # Auto-estimate ranges from data if requested
    if A_st_range is None or v_drift_ratio_range is None:
        if species_s is not None and species_t is not None:
            # Calculate A_st and v_drift_ratio for all data points
            v_th_para_s = T_to_vth(species_s['T_para'].to(u.K, equivalencies=u.temperature_energy()), mass=species_s['mass'], n=1)
            v_th_perp_s = T_to_vth(species_s['T_perp'].to(u.K, equivalencies=u.temperature_energy()), mass=species_s['mass'], n=1)
            v_th_para_t = T_to_vth(species_t['T_para'].to(u.K, equivalencies=u.temperature_energy()), mass=species_t['mass'], n=1)
            v_th_perp_t = T_to_vth(species_t['T_perp'].to(u.K, equivalencies=u.temperature_energy()), mass=species_t['mass'], n=1)
            
            v_st_para = np.sqrt((v_th_para_s**2 + v_th_para_t**2) / 2)
            v_st_perp = np.sqrt((v_th_perp_s**2 + v_th_perp_t**2) / 2)
            v_st = np.abs(species_s['v_drift'] - species_t['v_drift'])
            
            A_st_data = (v_st_perp**2 / v_st_para**2).decompose().value
            v_drift_ratio_data = (v_st / v_st_para).decompose().value
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(A_st_data) & np.isfinite(v_drift_ratio_data)
            A_st_data = A_st_data[valid_mask]
            v_drift_ratio_data = v_drift_ratio_data[valid_mask]
            
            if A_st_range is None:
                A_st_min = np.percentile(A_st_data, 5)
                A_st_max = np.percentile(A_st_data, 95)
                margin = (A_st_max - A_st_min) * 0.2  # 20% margin
                A_st_range = (max(default_A_st_range[0], A_st_min - margin), min(default_A_st_range[1], A_st_max + margin))
            
            if v_drift_ratio_range is None:
                v_ratio_min = np.percentile(v_drift_ratio_data, 5)
                v_ratio_max = np.percentile(v_drift_ratio_data, 95)
                margin = (v_ratio_max - v_ratio_min) * 0.2
                v_drift_ratio_range = (max(default_v_drift_ratio_range[0], v_ratio_min - margin), min(default_v_drift_ratio_range[1], v_ratio_max + margin))
        else:
            # Default ranges if no data provided
            A_st_range = default_A_st_range
            v_drift_ratio_range = default_v_drift_ratio_range
    
    # Create grid
    A_st_grid = np.logspace(np.log10(A_st_range[0]), np.log10(A_st_range[1]), n_A_st)
    v_ratio_grid = np.linspace(v_drift_ratio_range[0], v_drift_ratio_range[1], n_v_ratio)
    
    # Initialize result arrays
    F_1_05_25 = np.zeros((n_A_st, n_v_ratio))
    F_2_05_25 = np.zeros((n_A_st, n_v_ratio))
    F_1_15_25 = np.zeros((n_A_st, n_v_ratio))
    
    # Compute F_abc values
    # F_abc depends on x = 1 - A_st and y = A_st * v_st^2 / (4 * v_st_para^2)
    # where v_drift_ratio = v_st / v_st_para, so y = A_st * v_drift_ratio^2 / 4
    
    for i, A_st in enumerate(A_st_grid):
        for j, v_ratio in enumerate(v_ratio_grid):
            F_1_05_25[i, j] = _calc_F_st(1.0, 0.5, 2.5, A_st, v_ratio)
            F_2_05_25[i, j] = _calc_F_st(2.0, 0.5, 2.5, A_st, v_ratio)
            F_1_15_25[i, j] = _calc_F_st(1.0, 1.5, 2.5, A_st, v_ratio)
    
    # Create interpolators (using log scale for A_st for better accuracy)
    interpolator_1_05_25 = RegularGridInterpolator(
        (np.log10(A_st_grid), v_ratio_grid),
        F_1_05_25,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    interpolator_2_05_25 = RegularGridInterpolator(
        (np.log10(A_st_grid), v_ratio_grid),
        F_2_05_25,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    interpolator_1_15_25 = RegularGridInterpolator(
        (np.log10(A_st_grid), v_ratio_grid),
        F_1_15_25,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    lookup_table = {
        'F_1_05_25': interpolator_1_05_25,
        'F_2_05_25': interpolator_2_05_25,
        'F_1_15_25': interpolator_1_15_25,
        'A_st_range': A_st_range,
        'v_drift_ratio_range': v_drift_ratio_range,
        'n_A_st': n_A_st,
        'n_v_ratio': n_v_ratio
    }
    
    return lookup_table


def calc_nu_st(
    species_s: SpeciesParamsComp,
    species_t: SpeciesParamsComp
) -> u.Quantity:
    '''
    Calculate interspecies collision frequency ν_st.
    
    Hellinger (2016) Eq. 9:
        ν_st = (q_s² q_t² n_t ln Λ_st) / (12π^(3/2) ε₀² m_s m_st v_st∥³)
    
    where m_st = m_s×m_t/(m_s+m_t) is the reduced mass and
    v_st∥ = sqrt((v_s∥² + v_t∥²)/2) is the combined parallel thermal velocity.
    
    This frequency characterizes the rate at which species s exchanges
    momentum and energy with species t through Coulomb collisions.
    
    :param species_s: Dictionary with species s parameters (see SpeciesParams TypedDict)
    :param species_t: Dictionary with species t parameters (see SpeciesParams TypedDict)
    :return: Collision frequency with units of 1/s
    
    Example:
        >>> species_s = {'charge': 1*e.si, 'mass': 1*u.u, 'n': 10/u.cm**3,
        ...              'T_para': 1e5*u.K, 'T_perp': 1.2e5*u.K, 'v_drift': 400*u.km/u.s}
        >>> species_t = {'charge': 2*e.si, 'mass': 4*u.u, 'n': 0.4/u.cm**3,
        ...              'T_para': 3e5*u.K, 'T_perp': 3.5e5*u.K, 'v_drift': 430*u.km/u.s}
        >>> nu_st = calc_nu_st(species_s, species_t)
    '''
    m_s = species_s['mass']
    m_t = species_t['mass']
    q_s = species_s['charge']
    q_t = species_t['charge']
    T_para_s = species_s['T_para'].to(u.K, equivalencies=u.temperature_energy())
    T_para_t = species_t['T_para'].to(u.K, equivalencies=u.temperature_energy())
    
    # Thermal velocities: v_s∥ = sqrt(k_B*T_s∥/m_s)
    v_th_para_s = T_to_vth(T_para_s, mass=m_s, n=1)
    v_th_para_t = T_to_vth(T_para_t, mass=m_t, n=1)


    # Hellinger (2016) Eq. 7: Combined parallel thermal velocity
    v_st_para = np.sqrt((v_th_para_s**2 + v_th_para_t**2) / 2)
    # Reduced mass
    m_st = (m_s * m_t) / (m_s + m_t)
    # Coulomb logarithm (Eq. 14)
    ln_Lambda = _coloumb_logarithm_hellinger(species_s, species_t)
    
    # Hellinger (2016) Eq. 9: Interspecies collision frequency
    num = q_s**2 * q_t**2 * species_t['n'] * ln_Lambda
    den = 12 * np.pi**(1.5) * eps0**2 * m_s * m_st * v_st_para**3
    return (num / den).si


def calc_nu_Ts(
    species_s: SpeciesParamsComp
) -> u.Quantity:
    '''
    Calculate intraspecies isotropization frequency ν_Ts.
    
    Hellinger (2016) Eq. 4:
        ν_Ts = (q_s⁴ n_s ln Λ_ss) / (30π^(3/2) ε₀² m_s² v_s∥³) × ₂F₁(2, 3/2; 7/2; 1-A_s)
    
    where A_s = T_s⊥/T_s∥ is the temperature anisotropy and
    ₂F₁ is the Gauss hypergeometric function.
    
    This frequency characterizes the rate at which temperature anisotropy
    is reduced through self-collisions within species s.
    
    :param species_s: Dictionary with species s parameters (see SpeciesParams TypedDict)
    :return: Self-thermalization frequency with units of 1/s
    
    Example:
        >>> species_s = {'charge': 1*e.si, 'mass': 1*u.u, 'n': 10/u.cm**3,
        ...              'T_para': 1e5*u.K, 'T_perp': 1.2e5*u.K, 'v_drift': 400*u.km/u.s}
        >>> nu_Ts = calc_nu_Ts(species_s)
    '''
    m_s = species_s['mass']
    q_s = species_s['charge']
    T_para_s = species_s['T_para'].to(u.K, equivalencies=u.temperature_energy())
    
    # Coulomb logarithm for self-collision (Eq. 14)
    ln_Lambda = _coloumb_logarithm_hellinger(species_s, self_collision=True)
    
    # Thermal velocity: v_s∥ = sqrt(k_B*T_s∥/m_s)
    v_th_para_s = T_to_vth(T_para_s, mass=m_s, n=1)
    # Temperature anisotropy
    A_s = (species_s['T_perp'] / species_s['T_para']).decompose().value
    
    # Hellinger (2016) Eq. 4: Isotropization frequency
    num = q_s**4 * species_s['n'] * ln_Lambda
    den = 30 * np.pi**(1.5) * eps0**2 * m_s**2 * v_th_para_s**3
    
    # Gauss hypergeometric function ₂F₁(2, 3/2; 7/2; 1-A_s)
    hyper_val = hyp2f1(2, 1.5, 3.5, 1 - A_s)
    
    nu_Ts = (num / den * hyper_val).to(1 / u.s)
    return nu_Ts


def calc_heating_rates(
    species_s: SpeciesParamsComp,
    other_species: list[SpeciesParamsComp],
    F_abc_lookup_table: dict | None = None
) -> dict:
    '''
    Calculate temperature time derivatives dT_para/dt and dT_perp/dt.
    
    Hellinger (2016) Eq. 2 (perpendicular temperature evolution):
        (dT_s⊥/dt)_c = -ν_Ts(T_s⊥ - T_s∥) + T_s⊥ Σ_t ν_Hs⊥^(t)
    
    Hellinger (2016) Eq. 3 (parallel temperature evolution):
        (dT_s∥/dt)_c = 2ν_Ts(T_s⊥ - T_s∥) + T_s∥ Σ_t ν_Hs∥^(t)
    
    The first term represents intraspecies isotropization (self-collisions),
    the second term represents interspecies heating/cooling.
    
    :param species_s: Dictionary with species s parameters (see SpeciesParams TypedDict).
                      Required keys: charge, mass, n, T_para, T_perp, v_drift.
    :param other_species: List of dictionaries with field species parameters.
                          Each dict should have: charge, mass, n, T_para, T_perp, v_drift.
    :param F_abc_lookup_table: Optional precomputed lookup table for F_abc functions (from build_F_abc_lookup_table).
                               Using this can significantly speed up calculations for large datasets.
    :return: Dictionary with keys:
             - dT_para_dt, dT_perp_dt: Total temperature derivatives (K/s)
             - dT_para_dt_self, dT_perp_dt_self: Self-collision heating components (K/s)
             - dT_para_dt_inter, dT_perp_dt_inter: Inter-collision heating components (K/s)
             - nu_Ts: Isotropization frequency (1/s)
             - nu_Hs_perp, nu_Hs_para: Inter-species heating frequencies (1/s)
             - nu_Hs: Mean heating frequency (1/s) - Hellinger (2016) Eq. 12
             - nu_V_st: Deceleration frequency for each species pair (list, 1/s) - Hellinger (2016) Eq. 15-16
             - species_labels: Labels for nu_V_st entries (list of str)
    
    Example:
        >>> species_s = {'charge': 1*e.si, 'mass': 1*u.u, 'n': 10/u.cm**3,
        ...              'T_para': 1e5*u.K, 'T_perp': 1.2e5*u.K, 'v_drift': 400*u.km/u.s}
        >>> species_t = {'charge': 2*e.si, 'mass': 4*u.u, 'n': 0.4/u.cm**3,
        ...              'T_para': 3e5*u.K, 'T_perp': 3.5e5*u.K, 'v_drift': 430*u.km/u.s}
        >>> result = calc_heating_rates(species_s, [species_t])
    '''
    
    m_s = species_s['mass']
    q_s = species_s['charge']
    n_s = species_s['n']
    T_para_s = species_s['T_para'].to(u.K, equivalencies=u.temperature_energy())
    T_perp_s = species_s['T_perp'].to(u.K, equivalencies=u.temperature_energy())
    v_drift_s = species_s['v_drift']
    
    # Hellinger (2016) Eq. 4: Intraspecies isotropization frequency
    nu_Ts = calc_nu_Ts(species_s)
    
    nu_shape = np.shape(np.asarray(nu_Ts.to_value(1 / u.s)))
    sum_nu_H_perp = np.zeros(nu_shape, dtype=float) * (1 / u.s)
    sum_nu_H_para = np.zeros(nu_shape, dtype=float) * (1 / u.s)
    
    # Lists to store per-species deceleration frequencies and labels
    nu_V_st_list = []
    species_labels = []
    
    for species_t in other_species:
        m_t = species_t['mass']
        q_t = species_t['charge']
        n_t = species_t['n']
        T_para_t = species_t['T_para'].to(u.K, equivalencies=u.temperature_energy())
        T_perp_t = species_t['T_perp'].to(u.K, equivalencies=u.temperature_energy())
        v_drift_t = species_t['v_drift']
        
        # Hellinger (2016) Eq. 9: Interspecies collision frequency
        nu_st = calc_nu_st(species_s, species_t)
        
        # Hellinger (2016) Eq. 7: Combined thermal velocities
        v_th_para_s = T_to_vth(T_para_s, mass=m_s, n=1)
        v_th_perp_s = T_to_vth(T_perp_s, mass=m_s, n=1)
        v_th_para_t = T_to_vth(T_para_t, mass=m_t, n=1)
        v_th_perp_t = T_to_vth(T_perp_t, mass=m_t, n=1)
        
        v_st_para = np.sqrt((v_th_para_s**2 + v_th_para_t**2) / 2)
        v_st_perp = np.sqrt((v_th_perp_s**2 + v_th_perp_t**2) / 2)
        v_st = np.abs(v_drift_s - v_drift_t)
        
        # Hellinger (2016) Eq. 8: Combined temperature anisotropy
        A_st = (v_st_perp**2 / v_st_para**2).decompose().value
        v_drift_ratio = (v_st / v_st_para).decompose().value

        invalid_mask = np.abs(1.0 - np.asarray(A_st, dtype=float)) >= 1.0
        if np.any(invalid_mask):
            warnings.warn(
                "Some points violate |1-A_st|<1; corresponding values are set to NaN.",
                RuntimeWarning,
            )
        # Keep shape/type behavior and let NaN propagate through subsequent equations.
        A_st = np.where(invalid_mask, np.nan, A_st)

        # Combined (reduced) mass
        m_st = (m_s * m_t) / (m_s + m_t)
        
        # Hellinger (2016) Eq. 10: F_abc^(st) functions for heating rates
        # Use lookup table if provided, otherwise compute directly
        if F_abc_lookup_table is not None:
            if np.ndim(A_st) == 0:
                points = np.array([[np.log10(A_st), v_drift_ratio]], dtype=float)
            else:
                points = np.column_stack([np.log10(A_st), v_drift_ratio])

            F_st_1_05_25 = F_abc_lookup_table['F_1_05_25'](points)
            F_st_2_05_25 = F_abc_lookup_table['F_2_05_25'](points)
            F_st_1_15_25 = F_abc_lookup_table['F_1_15_25'](points)

            if np.ndim(A_st) == 0:
                F_st_1_05_25 = float(F_st_1_05_25)
                F_st_2_05_25 = float(F_st_2_05_25)
                F_st_1_15_25 = float(F_st_1_15_25)
        else:
            # Compute directly using numerical integration
            F_st_1_05_25 = _calc_F_st(1.0, 0.5, 2.5, A_st, v_drift_ratio)
            F_st_2_05_25 = _calc_F_st(2.0, 0.5, 2.5, A_st, v_drift_ratio)
            F_st_1_15_25 = _calc_F_st(1.0, 1.5, 2.5, A_st, v_drift_ratio)
        
        # Hellinger (2016) Eq. 5: Perpendicular heating rate
        # ν_Hs⊥^(t) = (ν_st/A_st)[(m_st/m_t)(T_t⊥/T_s⊥ - 1)F_{2,1/2,5/2}^(st) + F_{2,1/2,5/2}^(st) - F_{1,1/2,5/2}^(st)]
        term_perp = (m_st / m_t) * ((T_perp_t / T_perp_s).decompose().value - 1) * F_st_2_05_25
        nu_H_perp = (nu_st / A_st) * (term_perp + F_st_2_05_25 - F_st_1_05_25)
        
        # Hellinger (2016) Eq. 6: Parallel heating rate
        # ν_Hs∥^(t) = ν_st[(m_st/m_t)(T_t∥/T_s∥ - 1)F_{1,1/2,5/2}^(st) - 2(F_{2,1/2,5/2}^(st) - F_{1,1/2,5/2}^(st)) + (v_st²/2v_st∥²)F_{1,3/2,5/2}^(st)]
        term_para = (m_st / m_t) * ((T_para_t / T_para_s).decompose().value - 1) * F_st_1_05_25
        drift_term = ((v_st**2) / (2 * v_st_para**2)).decompose().value * F_st_1_15_25
        nu_H_para = nu_st * (term_para - 2 * (F_st_2_05_25 - F_st_1_05_25) + drift_term)
        
        sum_nu_H_perp += nu_H_perp
        sum_nu_H_para += nu_H_para
        
        # Hellinger (2016) Eq. 15-16: Deceleration frequency
        # ν_V^(st) = (q_s² q_t² n_st)/(24π^(3/2) ε₀² m_st² v_st∥³) ln Λ_st F_{1,3/2,5/2}^(st)
        # where n_st = (n_s m_s + n_t m_t)/(m_s + m_t)
        n_st = (n_s * m_s + n_t * m_t) / (m_s + m_t)
        
        ln_Lambda_st = _coloumb_logarithm_hellinger(species_s, species_t)
        
        nu_V_st = (q_s**2 * q_t**2 * n_st * ln_Lambda_st / 
                   (24 * np.pi**(1.5) * eps0**2 * m_st**2 * v_st_para**3) * 
                   F_st_1_15_25).to(1 / u.s)
        
        nu_V_st_list.append(nu_V_st)
        
        # Create label for this species pair using mass (in amu)
        mass_amu = m_t.to(u.u).value
        charge_e = (q_t / e.si).decompose().value
        species_label = f"q{charge_e:.0f}_m{mass_amu:.1f}"
        species_labels.append(species_label)
    
    # Hellinger (2016) Eq. 2 & 3: Final temperature derivatives
    # Self-collision (isotropization) components
    dT_perp_dt_self = -nu_Ts * (T_perp_s - T_para_s)
    dT_para_dt_self = 2 * nu_Ts * (T_perp_s - T_para_s)
    
    # Inter-collision (heating) components  
    dT_perp_dt_inter = T_perp_s * sum_nu_H_perp
    dT_para_dt_inter = T_para_s * sum_nu_H_para
    
    # Total temperature derivatives
    dT_perp_dt = dT_perp_dt_self + dT_perp_dt_inter
    dT_para_dt = dT_para_dt_self + dT_para_dt_inter
    
    # Hellinger (2016) Eq. 12: Mean heating rate
    # ν_Hs^(t) = (1/3)(T_s∥/T_s)ν_Hs∥^(t) + (2/3)(T_s⊥/T_s)ν_Hs⊥^(t)
    # where T_s = (2T_s⊥ + T_s∥)/3 is the mean temperature
    T_mean_s = (2 * T_perp_s + T_para_s) / 3
    nu_Hs = ((1/3) * (T_para_s / T_mean_s) * sum_nu_H_para + 
             (2/3) * (T_perp_s / T_mean_s) * sum_nu_H_perp).decompose()
    
    # Return dictionary with all relevant quantities
    return {
        'dT_para_dt': dT_para_dt.to(u.K / u.s),
        'dT_perp_dt': dT_perp_dt.to(u.K / u.s),
        'dT_para_dt_self': dT_para_dt_self.to(u.K / u.s),
        'dT_perp_dt_self': dT_perp_dt_self.to(u.K / u.s),
        'dT_para_dt_inter': dT_para_dt_inter.to(u.K / u.s),
        'dT_perp_dt_inter': dT_perp_dt_inter.to(u.K / u.s),
        'nu_Ts': nu_Ts.to(1 / u.s),
        'nu_Hs_perp': sum_nu_H_perp.to(1 / u.s),
        'nu_Hs_para': sum_nu_H_para.to(1 / u.s),
        'nu_Hs': nu_Hs.to(1 / u.s),
        'nu_V_st': [nu_v.to(1 / u.s) for nu_v in nu_V_st_list],
        'species_labels': species_labels
    }


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