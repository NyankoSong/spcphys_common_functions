"""
Comprehensive test suite for spcphys-common-functions package.

This module contains unit tests for all main functionalities of the package.
Run with: pytest tests/test_all.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from astropy.constants import m_p, k_B
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington


# ============================================================================
# Tests for parameters module
# ============================================================================

class TestAlfvenicParameters:
    """Tests for alfvenic_parameters module."""
    
    def test_calc_dx(self):
        """Test mean removal function."""
        from spcphys_common_functions.parameters.alfvenic_parameters import calc_dx
        
        # Test with numpy array
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = calc_dx(x)
        assert np.isclose(np.nanmean(dx), 0.0, atol=1e-10)
        
        # Test with astropy Quantity
        x_q = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.m/u.s
        dx_q = calc_dx(x_q)
        assert np.isclose(np.nanmean(dx_q).value, 0.0, atol=1e-10)
        assert dx_q.unit == u.m/u.s
    
    def test_calc_va(self):
        """Test Alfvén velocity calculation."""
        from spcphys_common_functions.parameters.alfvenic_parameters import calc_va
        
        # Create test data
        b = np.array([[1e-9, 0, 0], [1e-9, 0, 0], [1e-9, 0, 0]]) * u.T
        n = np.array([5e6, 5e6, 5e6]) * u.m**-3
        
        va = calc_va(b, n)
        
        # Check units and shape
        assert va.unit.is_equivalent(u.m/u.s)
        assert va.shape == b.shape
        
        # Alfvén velocity should be positive
        assert np.all(va.value >= 0)
    
    def test_calc_alfven(self):
        """Test Alfvénic parameters calculation."""
        from spcphys_common_functions.parameters.alfvenic_parameters import calc_alfven
        
        # Create synthetic test data
        np.random.seed(42)
        n_points = 50
        
        p_date = [datetime(2023, 1, 1) + timedelta(seconds=i) for i in range(n_points)]
        v = (np.random.randn(n_points, 3) * 10 + 400) * u.km/u.s
        b = (np.random.randn(n_points, 3) * 1 + 5) * u.nT
        n = (np.random.rand(n_points) * 5 + 5) * u.cm**-3
        
        result = calc_alfven(p_date, v, b, n=n)
        
        # Check all expected keys are present
        expected_keys = ['r3', 'p3', 'rvB', 'pvB', 'residual_energy', 
                        'cross_helicity', 'alfven_ratio', 'compressibility', 
                        'vA', 'num_valid_p_points', 'num_valid_b_points']
        for key in expected_keys:
            assert key in result
        
        # Check value ranges
        assert -1 <= result['r3'].value <= 1 or np.isnan(result['r3'].value)
        assert -1 <= result['cross_helicity'].value <= 1 or np.isnan(result['cross_helicity'].value)


class TestPlasmaBeta:
    """Tests for plasma_beta module."""
    
    def test_pressure_thermal(self):
        """Test thermal pressure calculation."""
        from spcphys_common_functions.parameters.plasma_beta import pressure_thermal
        
        n = 5e6 * u.m**-3
        T = 1e5 * u.K
        
        p_th = pressure_thermal(n, T)
        
        # Check units
        assert p_th.unit.is_equivalent(u.Pa)
        
        # Pressure should be positive
        assert p_th.value > 0
        
        # Check approximate value: P = n * k_B * T
        expected = (n * k_B * T).si
        assert np.isclose(p_th.value, expected.value, rtol=1e-5)
    
    def test_pressure_magnetic(self):
        """Test magnetic pressure calculation."""
        from spcphys_common_functions.parameters.plasma_beta import pressure_magnetic
        
        # Test with 1D magnitude array
        b_mag = np.array([5e-9]) * u.T
        p_b = pressure_magnetic(b_mag)
        assert p_b.unit.is_equivalent(u.Pa)
        assert p_b.value[0] > 0
        
        # Test with 3D vector
        b_vec = np.array([[3e-9, 4e-9, 0]]) * u.T
        p_b_vec = pressure_magnetic(b_vec)
        assert p_b_vec.unit.is_equivalent(u.Pa)
    
    def test_calc_beta(self):
        """Test plasma beta calculation."""
        from spcphys_common_functions.parameters.plasma_beta import calc_beta
        
        n = 5e6 * u.m**-3
        b = np.array([[5e-9, 0, 0]]) * u.T
        T = 1e5 * u.K
        
        beta = calc_beta(n, b, T)
        
        # Beta should be dimensionless
        assert beta.unit.is_equivalent(u.dimensionless_unscaled)
        
        # Beta should be positive
        assert np.all(beta.value > 0)


class TestVthET:
    """Tests for vth_E_T module."""
    
    def test_T_to_vth_and_back(self):
        """Test temperature to thermal velocity conversion and inverse."""
        from spcphys_common_functions.parameters.vth_E_T import T_to_vth, vth_to_T
        
        T_original = 1e6 * u.K
        
        vth = T_to_vth(T_original)
        T_recovered = vth_to_T(vth)
        
        assert vth.unit.is_equivalent(u.m/u.s)
        assert T_recovered.unit.is_equivalent(u.K)
        assert np.isclose(T_original.value, T_recovered.value, rtol=1e-10)
    
    def test_E_to_T_and_back(self):
        """Test energy to temperature conversion and inverse."""
        from spcphys_common_functions.parameters.vth_E_T import E_to_T, T_to_E
        
        E_original = 1e-17 * u.J
        
        T = E_to_T(E_original)
        E_recovered = T_to_E(T)
        
        assert T.unit.is_equivalent(u.K)
        assert E_recovered.unit.is_equivalent(u.J)
        assert np.isclose(E_original.value, E_recovered.value, rtol=1e-10)
    
    def test_E_to_vth_and_back(self):
        """Test energy to thermal velocity conversion and inverse."""
        from spcphys_common_functions.parameters.vth_E_T import E_to_vth, vth_to_E
        
        E_original = 1e-17 * u.J
        
        vth = E_to_vth(E_original)
        E_recovered = vth_to_E(vth)
        
        assert vth.unit.is_equivalent(u.m/u.s)
        assert E_recovered.unit.is_equivalent(u.J)
        assert np.isclose(E_original.value, E_recovered.value, rtol=1e-10)
    
    def test_T_tensor_to_T(self):
        """Test temperature tensor decomposition."""
        from spcphys_common_functions.parameters.vth_E_T import T_tensor_to_T
        
        # Create diagonal temperature tensor (isotropic case)
        T_val = 1e5
        T_tensor = np.array([[T_val, T_val, T_val]]) * u.K
        b = np.array([[1, 0, 0]]) * u.nT
        
        T_para, T_perp = T_tensor_to_T(T_tensor, b)
        
        # For isotropic case, T_para ≈ T_perp
        assert T_para.unit.is_equivalent(u.K)
        assert T_perp.unit.is_equivalent(u.K)
    
    def test_vth_types(self):
        """Test different thermal velocity types (n parameter)."""
        from spcphys_common_functions.parameters.vth_E_T import T_to_vth
        
        T = 1e6 * u.K
        
        # Most probable speed (n=2)
        vth_mp = T_to_vth(T, n=2)
        
        # RMS speed (n=3)
        vth_rms = T_to_vth(T, n=3)
        
        # RMS > most probable
        assert vth_rms.value > vth_mp.value


class TestCoulombCollision:
    """Tests for coulomb_collision module."""
    
    def test_calc_Ac(self):
        """Test Coulomb collisional age calculation."""
        from spcphys_common_functions.parameters.coulomb_collision import calc_Ac
        
        # Proton-proton collision parameters
        v_j = 400 * u.km/u.s
        T_j = 1e5 * u.K
        n_j = 5 * u.cm**-3
        v_i = 400 * u.km/u.s
        T_i = 1e5 * u.K
        n_i = 5 * u.cm**-3
        
        Ac = calc_Ac(v_j, T_j, n_j, v_i, T_i, n_i, 
                     charge_number_i=1, mass_number_i=1)
        
        # Collisional age should be dimensionless
        assert Ac.unit.is_equivalent(u.dimensionless_unscaled)


class TestEffectSize:
    """Tests for effectsize module."""
    
    def test_es_cohen(self):
        """Test Cohen's d effect size calculation."""
        from spcphys_common_functions.parameters.effectsize import es_cohen
        
        np.random.seed(42)
        
        # Two samples with known difference
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(1, 1, 100)  # Mean difference of 1
        
        es, ci = es_cohen(x1, x2)
        
        # Effect size should be around -1 (x1 < x2)
        assert -2 < es < 0
        
        # Confidence interval should be positive
        assert ci > 0
    
    def test_es_cohen_identical(self):
        """Test Cohen's d for identical distributions."""
        from spcphys_common_functions.parameters.effectsize import es_cohen
        
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(0, 1, 100)
        
        es, ci = es_cohen(x1, x2)
        
        # Effect size should be close to 0
        assert abs(es) < 0.5


class TestMinimumVariance:
    """Tests for minimum_variance module."""
    
    def test_min_var(self):
        """Test Minimum Variance Analysis."""
        from spcphys_common_functions.parameters.minimum_variance import min_var
        
        np.random.seed(42)
        
        # Create data with known variance structure
        n_points = 100
        data = np.random.randn(n_points, 3)
        data[:, 0] *= 10  # High variance in x
        data[:, 1] *= 5   # Medium variance in y
        data[:, 2] *= 1   # Low variance in z
        
        vrot, v, w = min_var(data)
        
        # Check shapes
        assert vrot.shape == data.shape
        assert v.shape == (3, 3)
        assert w.shape == (3,)
        
        # Eigenvalues should be in descending order
        assert w[0] >= w[1] >= w[2]
        
        # Eigenvectors should be orthonormal
        for i in range(3):
            assert np.isclose(np.linalg.norm(v[:, i]), 1.0, rtol=1e-5)


# ============================================================================
# Tests for processing module
# ============================================================================

class TestBackmapping:
    """Tests for backmapping module."""
    
    def test_most_probable_x(self):
        """Test most probable value calculation."""
        from spcphys_common_functions.processing.backmapping import most_probable_x
        
        np.random.seed(42)
        
        # Create distribution with known mode
        x = np.random.normal(5.0, 1.0, 1000)
        
        x_mp = most_probable_x(x)
        
        # Most probable should be close to mean for normal distribution
        assert 4.0 < x_mp < 6.0
    
    def test_ballistic_backmapping(self):
        """Test ballistic backmapping."""
        from spcphys_common_functions.processing.backmapping import ballistic_backmapping
        
        # Create test position
        pos = SkyCoord(
            lon=0 * u.deg,
            lat=0 * u.deg,
            radius=1 * u.AU,
            obstime=datetime(2023, 1, 1),
            frame=HeliographicCarrington
        )
        
        v_r = 400 * u.km/u.s
        r_target = 0.1 * u.AU
        
        pos_target = ballistic_backmapping(pos, v_r, r_target=r_target)
        
        # Target should be at specified radius
        assert np.isclose(pos_target.radius.to(u.AU).value, 0.1, rtol=1e-5)
        
        # Longitude should have changed due to solar rotation
        assert pos_target.lon.value != pos.lon.value


class TestTimeWindow:
    """Tests for time_window module."""
    
    def test_time_indices(self):
        """Test time indices finding."""
        from spcphys_common_functions.processing.time_window import _time_indices
        
        times = [datetime(2023, 1, 1, 0, i) for i in range(60)]
        time_range = [datetime(2023, 1, 1, 0, 10), datetime(2023, 1, 1, 0, 20)]
        
        indices = _time_indices(times, time_range)
        
        assert len(indices) == 10
        assert indices[0] == 10
        assert indices[-1] == 19
    
    def test_slide_time_window(self):
        """Test sliding time window generation."""
        from spcphys_common_functions.processing.time_window import slide_time_window
        
        times = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(120)]
        
        windows, indices = slide_time_window(
            times,
            window_size=timedelta(minutes=30),
            step=timedelta(minutes=10)
        )
        
        # Check window structure
        assert len(windows) == len(indices)
        assert len(windows) > 0
        
        # Each window should span 30 minutes
        for w in windows:
            assert (w[1] - w[0]).total_seconds() == 30 * 60


class TestPreprocess:
    """Tests for preprocess module."""
    
    def test_find_argnan(self):
        """Test out-of-bound detection."""
        from spcphys_common_functions.processing.preprocess import find_argnan
        
        x = np.array([1, 2, 3, 4, 1e31, 6])
        
        nan_indices = find_argnan(x)
        
        assert 4 in nan_indices  # 1e31 > default boundary
        assert 0 not in nan_indices  # normal value
    
    def test_process_nan(self):
        """Test NaN processing."""
        from spcphys_common_functions.processing.preprocess import process_nan
        
        x = np.array([1, 2, 1e31, 4, -1e31, 6])
        
        processed = process_nan(x.copy())
        
        assert np.isnan(processed[2])
        assert np.isnan(processed[4])
        assert processed[0] == 1
    
    def test_npdt64_to_dt(self):
        """Test numpy datetime64 to datetime conversion."""
        from spcphys_common_functions.processing.preprocess import npdt64_to_dt
        
        npdt = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64')
        
        dt = npdt64_to_dt(npdt)
        
        assert isinstance(dt[0], datetime)
        assert dt[0].year == 2023
        assert dt[0].month == 1
        assert dt[0].day == 1
    
    def test_interpolate(self):
        """Test interpolation function."""
        from spcphys_common_functions.processing.preprocess import interpolate
        
        xp = [datetime(2023, 1, 1, 0, i) for i in range(10)]
        yp = np.arange(10).astype(float)
        x = datetime(2023, 1, 1, 0, 4, 30)  # Between index 4 and 5
        
        y = interpolate(x, xp, yp)
        
        # Should interpolate to ~4.5
        assert 4.0 < y < 5.0
    
    def test_down_sample(self):
        """Test down-sampling function."""
        from spcphys_common_functions.processing.preprocess import down_sample
        
        # High-resolution data
        tp = [datetime(2023, 1, 1, 0, 0, i) for i in range(60)]
        xp = np.sin(np.linspace(0, 2*np.pi, 60))
        
        # Low-resolution target times
        t = [datetime(2023, 1, 1, 0, 0, i*10) for i in range(6)]
        
        # Test interpolate method (works for 1D)
        x_interp = down_sample(t, tp, xp, method='interpolate')
        assert len(x_interp) == 6
        
        # Test mean method with 2D data
        xp_2d = np.column_stack([np.sin(np.linspace(0, 2*np.pi, 60)), 
                                  np.cos(np.linspace(0, 2*np.pi, 60))])
        x_mean = down_sample(t, tp, xp_2d, method='mean')
        assert x_mean.shape == (6, 2)


class TestVecCartSph:
    """Tests for vec_cart_sph module."""
    
    def test_vec_cart_to_sph(self):
        """Test Cartesian to spherical conversion."""
        from spcphys_common_functions.processing.vec_cart_sph import vec_cart_to_sph
        
        # Vector pointing in +x direction
        v = np.array([[1, 0, 0]]) * u.km/u.s
        r = np.array([[1, 0, 0]])
        
        v_mag, theta = vec_cart_to_sph(v, r)
        
        assert np.isclose(v_mag[0].value, 1.0)
        assert np.isclose(theta[0], 0.0)  # Aligned with r
    
    def test_vec_sph_to_cart(self):
        """Test spherical to Cartesian conversion."""
        from spcphys_common_functions.processing.vec_cart_sph import vec_sph_to_cart
        
        v_mag = np.array([1.0]) * u.km/u.s
        azimuth = np.array([0.0]) * u.deg
        elevation = np.array([0.0]) * u.deg
        
        x, y, z = vec_sph_to_cart(v_mag, azimuth, elevation)
        
        assert np.isclose(x[0].value, 1.0)
        assert np.isclose(y[0].value, 0.0, atol=1e-10)
        assert np.isclose(z[0].value, 0.0, atol=1e-10)
    
    def test_quat_rot_vec(self):
        """Test quaternion rotation of vectors."""
        from spcphys_common_functions.processing.vec_cart_sph import quat_rot_vec
        
        # Identity quaternion (no rotation)
        q = np.array([[1, 0, 0, 0]])
        v = np.array([[1, 0, 0]])
        
        v_rot = quat_rot_vec(q, v)
        
        assert np.allclose(v_rot, v, atol=1e-10)
        
        # 90-degree rotation around z-axis
        # q = cos(45°) + sin(45°)*k = [cos(45°), 0, 0, sin(45°)]
        angle = np.pi / 2
        q_90z = np.array([[np.cos(angle/2), 0, 0, np.sin(angle/2)]])
        v_rot_90 = quat_rot_vec(q_90z, v)
        
        # [1,0,0] rotated 90° around z should give [0,1,0]
        assert np.isclose(v_rot_90[0, 0], 0, atol=1e-10)
        assert np.isclose(v_rot_90[0, 1], 1, atol=1e-10)
        assert np.isclose(v_rot_90[0, 2], 0, atol=1e-10)


class TestPlotTools:
    """Tests for plot_tools module."""
    
    def test_box_stats(self):
        """Test box plot statistics calculation."""
        from spcphys_common_functions.processing.plot_tools import box_stats
        
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        stats = box_stats(data)
        
        assert 'whislo' in stats
        assert 'q1' in stats
        assert 'med' in stats
        assert 'q3' in stats
        assert 'whishi' in stats
        assert 'mean' in stats
        
        # Quartiles should be ordered
        assert stats['whislo'] <= stats['q1'] <= stats['med'] <= stats['q3'] <= stats['whishi']
    
    def test_box_stats_log(self):
        """Test box plot statistics with log scale."""
        from spcphys_common_functions.processing.plot_tools import box_stats
        
        np.random.seed(42)
        data = np.abs(np.random.normal(10, 2, 1000))  # Positive values
        
        stats = box_stats(data, scale='log')
        
        assert stats['whislo'] > 0
        assert stats['q1'] > 0
    
    def test_histogram(self):
        """Test histogram function."""
        from spcphys_common_functions.processing.plot_tools import histogram
        
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        
        result = histogram([x, y], bins=20, scales='linear')
        
        assert 'hist' in result
        assert 'edges' in result
        assert 'mids' in result
        assert result['hist'].shape[0] == 20
        assert result['hist'].shape[1] == 20


class TestUtils:
    """Tests for utils module."""
    
    def test_determine_processes(self):
        """Test process number determination."""
        from spcphys_common_functions.utils.utils import _determine_processes
        import os
        
        # Test with explicit number
        assert _determine_processes(4) == 4
        
        # Test with fraction
        result = _determine_processes(0.5)
        assert result == int(os.cpu_count() * 0.5) or result == 1
        
        # Test with 1
        assert _determine_processes(1) == 1
        
        # Test minimum is 1
        assert _determine_processes(0.001) >= 1


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_full_alfven_analysis_pipeline(self):
        """Test complete Alfvénic analysis workflow."""
        from spcphys_common_functions.parameters.alfvenic_parameters import calc_alfven_t
        from spcphys_common_functions.parameters.plasma_beta import calc_beta
        
        np.random.seed(42)
        
        # Create realistic solar wind data
        n_points = 200
        p_date = np.array([datetime(2023, 1, 1) + timedelta(seconds=i*4) for i in range(n_points)])
        
        v = (np.random.randn(n_points, 3) * 20 + np.array([400, 0, 0])) * u.km/u.s
        b = (np.random.randn(n_points, 3) * 2 + np.array([5, 0, 0])) * u.nT
        n = (np.random.rand(n_points) * 5 + 5) * u.cm**-3
        T = (np.random.rand(n_points) * 5e4 + 1e5) * u.K
        
        # Calculate time-dependent Alfvénic parameters
        result = calc_alfven_t(
            p_date, v, b, n=n,
            window_size=timedelta(minutes=5),
            step=timedelta(minutes=2)
        )
        
        assert len(result['time']) > 0
        assert len(result['cross_helicity']) == len(result['time'])
        
        # Calculate plasma beta
        beta = calc_beta(n, b, T)
        assert len(beta) == n_points
    
    def test_thermal_velocity_consistency(self):
        """Test consistency between different thermal velocity conversions."""
        from spcphys_common_functions.parameters.vth_E_T import (
            T_to_vth, vth_to_T, T_to_E, E_to_T, E_to_vth, vth_to_E
        )
        
        T_start = 1e6 * u.K
        
        # Path 1: T -> vth -> T
        vth1 = T_to_vth(T_start)
        T1 = vth_to_T(vth1)
        
        # Path 2: T -> E -> T
        E1 = T_to_E(T_start)
        T2 = E_to_T(E1)
        
        # Path 3: T -> E -> vth -> E -> T
        E2 = T_to_E(T_start)
        vth2 = E_to_vth(E2)
        E3 = vth_to_E(vth2)
        T3 = E_to_T(E3)
        
        # All paths should give same result
        assert np.isclose(T_start.value, T1.value, rtol=1e-10)
        assert np.isclose(T_start.value, T2.value, rtol=1e-10)
        assert np.isclose(T_start.value, T3.value, rtol=1e-10)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
