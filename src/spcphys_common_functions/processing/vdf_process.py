'''
OWEN C J, BRUNO R, LIVI S, et al., 2020. The Solar Orbiter Solar Wind Analyser (SWA) suite[J/OL]. Astronomy & Astrophysics, 642: A16. DOI:10.1051/0004-6361/201937259.
DE MARCO R, BRUNO R, JAGARLAMUDI V K, et al., 2023. Innovative technique for separating proton core, proton beam, and alpha particles in solar wind 3D velocity distribution functions[J/OL]. Astronomy & Astrophysics, 669: A108. DOI:10.1051/0004-6361/202243719.
'''

import warnings
from typing import List, Tuple
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm
import numpy as np
from astropy.constants import m_p
from scipy import interpolate as sinterp
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import QuadMesh

from ..utils.utils import _determine_processes
from .time_window import slide_time_window
from .plot_tools import plot_mesh1d_ts


def generate_vdf_cart_unit(vdf_time: List[datetime]|np.ndarray, imf_time: List[datetime]|np.ndarray, imf: u.Quantity, vp_time: List[datetime]|np.ndarray|None =None, vp: u.Quantity|None =None) -> np.ndarray:
    
    '''
    This function generates the base vectors of the new coordinate system.
    
    :param vdf_time: The time array of VDF.
    :param imf_time: The time array of IMF.
    :param imf: The IMF data in shape (imf_time, 3).
    :param vp_time: The time array of velocity. Default is None.
    :param vp: The velocity data in shape (vp_time, 3). Default is None.
    
    :return v_unit_new: The base vectors of the new coordinate system in shape (time, 3, 3), where the last dimension is [e_ix, e_iy, e_iz].
    :return e_b: The IMF unit vector.
    '''
    
    if not imf.unit.is_equivalent(u.T):
        warnings.warn("IMF does not have units equivalent to T. Are you sure to not use magnetic field?", UserWarning)
    if imf.shape[1] != 3 or len(imf.shape) != 2 or imf.shape[0] != len(imf_time):
        raise ValueError("IMF must be a 2-dimensional array with shape (imf_time, 3).")
    if vp is not None:
        if not vp.unit.is_equivalent(u.m/u.s):
            warnings.warn("vp does not have units equivalent to m/s. Are you sure to not use velocity?", UserWarning)
        if vp.shape[1] != 3 or len(vp.shape) != 2 or vp.shape[0] != len(vp_time):
            raise ValueError("vp must be a 2-dimensional array with shape (vp_time, 3).")
        
    
    _, imf_indices_align_to_vdf = slide_time_window(imf_time, align_to=vdf_time)
    imf_vec_mean = np.array([np.mean(imf[imf_indices_align_to_vdf[i]], axis=0) for i in range(len(imf_indices_align_to_vdf))])
    
    e_b = imf_vec_mean / np.tile(np.linalg.norm(imf_vec_mean, axis=1), (3, 1)).T
    
    if vp_time is not None and vp is not None:
        _, vp_indices_align_to_vdf = slide_time_window(vp_time, align_to=vdf_time)
        vp_vec_mean = np.array([np.mean(vp[vp_indices_align_to_vdf[i]], axis=0) for i in range(len(vp_indices_align_to_vdf))])
        
        e_v = vp_vec_mean / np.tile(np.linalg.norm(vp_vec_mean, axis=1), (3, 1)).T
    else:
        e_v = np.tile(np.array([1, 0, 0]), (len(vdf_time), 1))
        
    e_3 = np.cross(e_v, e_b)
    e_2 = np.cross(e_3, e_v)
    
    return np.array([e_v, e_2, e_3]).transpose(1, 0, 2), e_b



def vdf_sph_to_cart(azimuth: u.Quantity, elevation: u.Quantity, energy: u.Quantity, vdf: u.Quantity, v_unit_new: np.ndarray) -> Tuple[u.Quantity, u.Quantity]:
    '''
    This function calculates the 3D scatters of VDF in the new coordinate system. (Only tested for SolO data)
    
    :param azimuth: Azimuth angles in degrees.
    :param elevation: Elevation angles in degrees.
    :param energy: Energy in J.
    :param vdf: VDF data in shape (time, azimuth, elevation, energy).
    :param v_unit_new: Base vectors of the new coordinate system in shape (time, 3, 3), where the last dimension is [e_ix, e_iy, e_iz]. Default is None, which means the original base vectors [[1,0,0],[0,1,0],[0,0,1]] is used.
    
    :return vdf_vec: The 3D scatters of VDF in the new coordinate system in shape (time, azimuth*elevation*energy, 3).
    :return vdf_value: The values of the 3D scatters of VDF in shape (time, azimuth*elevation*energy).
    '''
    
    if azimuth.ndim != 1 or elevation.ndim != 1 or energy.ndim != 1:
        raise ValueError("Azimuth, elevation, and energy must be 1-dimensional arrays.")
    if vdf.ndim != 4 or vdf.shape[1] != len(azimuth) or vdf.shape[2] != len(elevation) or vdf.shape[3] != len(energy):
        raise ValueError("VDF must be a 4-dimensional array with shape (time, azimuth, elevation, energy).")
    if v_unit_new.shape != (vdf.shape[0], 3, 3):
        raise ValueError("v_unit_new must have shape (time, 3, 3).")
    if not azimuth.unit.is_equivalent(u.deg) or not elevation.unit.is_equivalent(u.deg) or not energy.unit.is_equivalent(u.J):
        raise ValueError("Azimuth, elevation, and energy must have units equivalent to deg, deg, and J.")
    if not vdf.unit.is_equivalent(u.s**3 / u.m**6):
        warnings.warn("VDF does not have units equivalent to s^3/m^6. Is this phase space density instead of counts?", UserWarning)
    
    energy = energy.si
    
    v = np.sqrt(2*energy / m_p).si
    vdf_vec_t = np.zeros((vdf.shape[0], len(azimuth)*len(elevation)*len(energy), 3)) * v.unit
    vdf_value_t = np.zeros((vdf.shape[0], len(azimuth)*len(elevation)*len(energy))) * vdf.unit
    
    for i in range(vdf.shape[0]):
        v_unit_tmp = np.array([np.tile(np.cos(np.deg2rad(azimuth)).reshape(-1, 1), (1, len(elevation))) * np.tile(np.cos(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1)), 
                               np.tile(np.sin(np.deg2rad(azimuth)).reshape(-1, 1), (1, len(elevation))) * np.tile(np.cos(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1)),
                               np.tile(np.sin(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1))])
        v_unit_tmp = np.transpose(v_unit_tmp, (1, 2, 0))
        
        v_tmp = np.zeros((len(azimuth)*len(elevation)*len(energy), v_unit_tmp.shape[-1]))
        for j in range(v_unit_tmp.shape[-1]):
            v_tmp[:, j] = (np.tile(v.reshape(1, 1, -1), (len(azimuth), len(elevation), 1)) * np.tile((np.dot(v_unit_tmp, v_unit_new[i, j, :])/np.linalg.norm(v_unit_new[i, j, :])).reshape(len(azimuth), len(elevation), -1), (1, 1, len(energy)))).reshape(-1)
            
        vdf_vec_t[i, :, :] = v_tmp * v.unit
        vdf_value_t[i, :] = vdf[i, :, :, :].reshape(-1)
    
    return vdf_vec_t, vdf_value_t


def _imf_in_vdf(mag_vector_3d: np.ndarray, v_unit: np.ndarray):
    '''
    Calculate the magnetic vector in the new coordinate system.
    
    If the N-component of z-axis unit vector (e_z[-1]) is less than 0,
    the y and z components of the magnetic field will be reversed.
    This logic is derived from a Fortran program with unknown reasoning.
    
    :param mag_vector_3d: The 3D magnetic vector.
    :type mag_vector_3d: np.ndarray
    :param v_unit: The base vectors of the coordinate system, shape (3,3).
    :type v_unit: np.ndarray
    
    :return: The magnetic vector in the new coordinate system.
    :rtype: np.ndarray
    '''
    mag_vector = np.zeros(3)
    mag_vector[0] = np.dot(mag_vector_3d, v_unit[0]) / np.linalg.norm(v_unit[0])
    mag_vector[1] = np.dot(mag_vector_3d, v_unit[1]) / np.linalg.norm(v_unit[1])
    mag_vector[2] = np.dot(mag_vector_3d, v_unit[2]) / np.linalg.norm(v_unit[2])
    
    # Logic for sign reversal (uncertain reasoning, from Fortran code):
    if v_unit[2, -1] < 0:
        mag_vector[1] = -mag_vector[1]
        mag_vector[2] = -mag_vector[2] # ?
        
    return mag_vector
    
    

def vdf_griddata(vdf_vec: u.Quantity, vdf_value: u.Quantity, grid_lim: np.ndarray|list =None, grid_size: int =25, normalize: bool =True, device: str ='cpu') -> Tuple[u.Quantity, u.Quantity]:
    
    '''
    This function interpolates the 3D scatters of VDF to a 3D grid.
    
    :param vdf_vec: The 3D scatters of VDF in shape (azimuth*elevation*energy, 3).
    :param vdf_value: The values of the 3D scatters of VDF in shape (azimuth*elevation*energy).
    :param grid_lim: The limits of the grid in shape (3, 2). Default is None, which means the limits will be calculated automatically.
    :param grid_size: The size of the grid.
    :param device: The device to use. Default is 'cpu'.
    
    :return pdf_3d: The 3D grid of VDF in shape (grid_size, grid_size, grid_size).
    :return v_grid: The 3D grid of velocity in shape (3, grid_size).
    '''
    
    if not vdf_vec.unit.is_equivalent(u.m/u.s):
        raise ValueError("VDF vectors must have units equivalent to m/s.")
    if vdf_vec.shape[0] != vdf_value.shape[0] or vdf_vec.shape[-1] != 3 or vdf_value.ndim != 1 or vdf_vec.ndim != 2:
        raise ValueError("The shapes of VDF vectors must be (azimuth*elevation*energy, 3) and VDF values must be (azimuth*elevation*energy).")
    if device not in ['cpu', 'gpu']:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")
        
    v_grid = []
    valid_vdf_indices = vdf_value > 0
    valid_vdf_vec = vdf_vec[valid_vdf_indices, :]
    valid_vdf_value = vdf_value[valid_vdf_indices]
    for i in range(3):
        if grid_lim is not None:
            v_grid.append(np.linspace(grid_lim[i][0], grid_lim[i][1], grid_size))
        else:
            v_grid.append(np.linspace(np.min(valid_vdf_vec[:, i]), np.max(valid_vdf_vec[:, i]), grid_size))
        
    v_gridmesh = np.meshgrid(*v_grid)
    v_gridmesh_points = np.hstack((v_gridmesh[0].reshape(-1, 1), v_gridmesh[1].reshape(-1, 1), v_gridmesh[2].reshape(-1, 1)))
    
    for i in range(v_gridmesh_points.shape[-1]):
        v_gridmesh_points = v_gridmesh_points[v_gridmesh_points[:, v_gridmesh_points.shape[1] - i - 1].argsort(kind='mergesort'), :]
    
    if device == 'cpu':
        pdf = sinterp.griddata(vdf_vec.to_value(), vdf_value.to_value(), v_gridmesh_points.to_value(), method='linear', fill_value=0) * valid_vdf_value.unit
        pdf_3d = pdf.reshape(grid_size, grid_size, grid_size)
        if normalize:
            pdf_3d /= pdf.sum()
        
    else:
        raise ValueError('Currently only support cpu.')
    
    return pdf_3d, np.concatenate([v_axis[:, np.newaxis] for v_axis in v_grid], axis=1).T


def _unpack_params_griddata(vdf_input):
    return vdf_griddata(vdf_vec=vdf_input[0], vdf_value=vdf_input[1], grid_lim=vdf_input[2], grid_size=vdf_input[3])



def vdf_griddata_t(vdf_vec_t: u.Quantity, vdf_value_t: u.Quantity, grid_lim: np.ndarray|list|None =None, grid_size: int =25, device: str ='cpu', multiprocess: int|float =1) -> Tuple[u.Quantity, u.Quantity]:
    
    '''
    This function packs the `vdf_griddata` function to calculate the 3D grid of VDFs.
    
    :param vdf_vec_t: The 3D scatters of VDF in the new coordinate system in shape (time, azimuth*elevation*energy, 3).
    :param vdf_value_t: The values of the 3D scatters of VDF in shape (time, azimuth*elevation*energy).
    :param grid_lim: The limits of the grid in shape (3, 2) or (t, 3, 2). Default is None, which means the limits will be calculated automatically.
    :param grid_size: The size of the grid.
    :param device: The device to use. Default is 'cpu'.
    :param multiprocess: Whether to use multiprocess. Default is False.
    
    :return pdf_t: The 3D grid of VDF in shape (time, grid_size, grid_size, grid_size).
    :return v_grid_t: The 3D grid of velocity in shape (time, 3, grid_size).
    '''
    
    if isinstance(grid_lim, list):
        grid_lim = np.array(grid_lim)
    
    if vdf_vec_t.shape != (vdf_value_t.shape[0], vdf_value_t.shape[1], 3) or vdf_value_t.ndim != 2:
        raise ValueError("The shapes of VDF vectors must be (time, azimuth*elevation*energy, 3) and VDF values must be (time, azimuth*elevation*energy).")
    if grid_lim is not None and (grid_lim.shape != (vdf_value_t.shape[0], 3, 2) or grid_lim.shape != (3, 2)):
        raise ValueError("grid_lim must have shape (time, 3, 2) or (3, 2).")
    if device not in ['cpu', 'gpu']:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")

    pdf_t = np.zeros((vdf_value_t.shape[0], grid_size, grid_size, grid_size)) * u.dimensionless_unscaled
    v_grid_t = np.zeros((vdf_value_t.shape[0], 3, grid_size)) * vdf_vec_t.unit
    
    if grid_lim is None:
        valid_vdf_t_indices = vdf_value_t.reshape(-1) > 0
        valid_vdf_vec_t = vdf_vec_t.reshape(-1, 3)[valid_vdf_t_indices]
        grid_lim = np.array([[np.min(valid_vdf_vec_t[:, i].to_value()), np.max(valid_vdf_vec_t[:, i]).to_value()] for i in range(3)]) * vdf_vec_t.unit
    
    if multiprocess != 1:
        if len(grid_lim.shape) == 2:
            vdf_input_t = [(vdf_vec_t[i, :, :], vdf_value_t[i, :], grid_lim, grid_size) for i in range(vdf_value_t.shape[0])]
        else:
            vdf_input_t = [(vdf_vec_t[i, :, :], vdf_value_t[i, :], grid_lim[i], grid_size) for i in range(vdf_value_t.shape[0])]
        with Pool(processes=_determine_processes(multiprocess)) as pool:
            result = list(tqdm(pool.imap(_unpack_params_griddata, vdf_input_t), total=len(vdf_input_t)))
        for i in range(vdf_value_t.shape[0]):
            pdf_t[i], v_grid_t[i] = result[i]
    else:
        for i in tqdm(range(vdf_value_t.shape[0])):
            pdf_t[i], v_grid_t[i] = vdf_griddata(vdf_vec_t[i, :, :], vdf_value_t[i, :], grid_size, device=device)
        
    return pdf_t, v_grid_t


def vdf_core(pdf_t: u.Quantity|np.ndarray, v_grid_t: u.Quantity|np.ndarray):
    
    v_t_max = np.zeros((v_grid_t.shape[0], 3))
        
    for (i, n) in enumerate(range(3)):
        v_grid_index = [j for j in range(len(pdf_t.shape) - 1)]
        v_grid_index.remove(n)
        pdf_1d_t = np.sum(pdf_t, axis=(v_grid_index[0] + 1, v_grid_index[1] + 1))
        pdf_1d_t_max_index = np.argmax(pdf_1d_t, axis=1)
        v_t_max[:, i] = np.array([v_grid_t[j, n, pdf_1d_t_max_index[j]] for j in range(v_grid_t.shape[0])]).reshape(-1)
        
    return v_t_max if isinstance(v_grid_t, np.ndarray) else v_t_max * v_grid_t.unit


def _plot_arrow(axes: plt.Axes, mag_vector_3d: np.ndarray, point_3d: np.ndarray, v_unit: np.ndarray, **arrow_kwargs):
    
    v_unit_index = [0, 1]
    point = point_3d[v_unit_index]
    mag_vector = _imf_in_vdf(mag_vector_3d, v_unit)[v_unit_index]
    
    x_lim = axes.get_xlim()
    y_lim = axes.get_ylim()
    
    x_width = x_lim[1] - x_lim[0]
    y_height = y_lim[1] - y_lim[0]
    scale_factor = 0.1
    
    x_lim = [x_lim[0] + x_width * scale_factor, x_lim[1] - x_width * scale_factor]
    y_lim = [y_lim[0] + y_height * scale_factor, y_lim[1] - y_height * scale_factor]
    
    arrow_k = mag_vector[-1] / mag_vector[0]
    # y - y0 = k * (x - x0)
    y_0 = (x_lim[0] - point[0]) * arrow_k + point[1]
    if y_0 > y_lim[1]:
        y_0 = y_lim[1]
        x_0 = (y_lim[1] - point[1]) / arrow_k + point[0]
    elif y_0 < y_lim[0]:
        y_0 = y_lim[0]
        x_0 = (y_lim[0] - point[1]) / arrow_k + point[0]
    else:
        x_0 = x_lim[0]
    
    y_1 = (x_lim[1] - point[0]) * arrow_k + point[1]
    if y_1 > y_lim[1]:
        y_1 = y_lim[1]
        x_1 = (y_lim[1] - point[1]) / arrow_k + point[0]
    elif y_1 < y_lim[0]:
        y_1 = y_lim[0]
        x_1 = (y_lim[0] - point[1]) / arrow_k + point[0]
    else:
        x_1 = x_lim[1]
    
    head_size = (x_lim[1] - x_lim[0])/30
    
    if mag_vector[0] > 0:
        axes.arrow(x_0, y_0, x_1 - x_0, y_1 - y_0, head_width=head_size, head_length=head_size, **arrow_kwargs)
    else:
        axes.arrow(x_1, y_1, x_0 - x_1, y_0 - y_1, head_width=head_size, head_length=head_size, **arrow_kwargs)



def plot_vdf_2d(axes: plt.Axes, pdf: u.Quantity|np.ndarray, v_grid: u.Quantity, compress_v_unit: int =2,
                core_marker: bool|dict =False, imf_vector: u.Quantity|np.ndarray|None =None, v_unit: np.ndarray|None =None,
                clip_lower_percentage: float =0, color_norm: Normalize|LogNorm|None =None, color_levels: List[float]|None =None, cax: plt.Axes|None =None,
                pcolormesh_kwargs: dict|None =None, contour_kwargs: bool|dict =True, arrow_kwargs: dict|None =None) -> QuadMesh:
    
    '''
    Plot the 2D VDF.
    
    :param axes: The axis to plot.
    :param pdf: The 3D grid of VDF in shape (grid_size, grid_size, grid_size).
    :param v_grid: The 3D grid of velocity in shape (3, grid_size).
    :param compress_v_unit: The unit of velocity that will not be plotted. Default is 2, which means the z-axis.
    :param core_marker: Whether to plot the core marker. Could be a dictionary of the marker properties. Default is False.
    :param imf_vector: Whether to plot the IMF vector. Default is None.
    :param v_units: The base vectors of the new coordinate system. This must be provided if `imf_vector` is not None. Default is None.
    :param clip_lower_percentage: The percentage of the lower limit of the color normalization. Should be between 0 and 1. Default is 0.
    :param color_norm: The color normalization. Default is None, which means the LogNorm will be used.
    :param color_levels: The color levels. Default is None, which means the levels will be calculated automatically.
    :param cax: The color bar axis. Default is None.
    :param pcolormesh_kwargs: The keyword arguments for the pcolormesh function. Default is None, which means {'cmap': 'jet', 'shading': 'auto'}.
    :param contour_kwargs: The keyword arguments for the contour function. Default is True, which means {'colors':'k'}.
    :param arrow_kwargs: The keyword arguments for the arrow function. Default is None, which means {'shape': 'full', 'fc': 'k', 'ec': 'k', 'lw': 1}.
    
    :return quadmesh: The quadmesh of the plot.
    '''
    
    if compress_v_unit not in [0, 1, 2]:
        raise ValueError("compress_v_unit must be 0, 1, or 2.")
    
    if isinstance(pdf, np.ndarray):
        pdf = pdf * u.dimensionless_unscaled
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {'cmap': 'jet', 'shading': 'auto'}
    if contour_kwargs is True:
        contour_kwargs = {'colors':'k'}
    if arrow_kwargs is None:
        arrow_kwargs = {'shape': 'full', 'fc': 'k', 'ec': 'k', 'lw': 1}
    
    v_grid_index = [i for i in range(len(pdf.shape))]
    v_grid_index.remove(compress_v_unit)
    
    v_grid = v_grid.to(u.km/u.s).to_value()
    pdf_2d = pdf.sum(axis=compress_v_unit).to_value()
    
    if color_norm is None:
        low_level = np.log10(np.min(pdf_2d[pdf_2d > 0])) + (np.log10(np.max(pdf_2d)) - np.log10(np.min(pdf_2d[pdf_2d > 0]))) * clip_lower_percentage
        high_level = np.log10(np.max(pdf_2d))
        color_norm = LogNorm(vmin=10**low_level, vmax=10**high_level)
    if color_levels is None and isinstance(color_norm, LogNorm):
        low_level = np.log10(color_norm.vmin)
        high_level = np.log10(color_norm.vmax)
        color_levels = np.logspace(int(low_level), int(high_level), (int(high_level) + 1 - int(low_level)))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        quadmesh = axes.pcolormesh(v_grid[v_grid_index[0]], v_grid[v_grid_index[1]], pdf_2d.T, norm=color_norm, **pcolormesh_kwargs)
        if color_levels is not None and contour_kwargs is not None:
            contour_kwargs['levels'] = color_levels
            axes.contour(v_grid[v_grid_index[0]], v_grid[v_grid_index[1]], pdf_2d.T, norm=color_norm, **contour_kwargs)
            
    core_pos = vdf_core(pdf[np.newaxis, :], v_grid[np.newaxis, :])[0]
    if core_marker is not False:
        index = [i for i in range(3) if i != compress_v_unit]
        if core_marker is True:
            axes.plot(core_pos[index[0]], core_pos[index[1]], 'kx', markersize=10)
        elif isinstance(core_marker, dict):
            axes.plot(core_pos[index[0]], core_pos[index[1]], **core_marker)
            
    if imf_vector is not None and v_unit is not None:
        _plot_arrow(axes, imf_vector.to_value() if isinstance(imf_vector, u.Quantity) else imf_vector, core_pos, v_unit, **arrow_kwargs)
        
    axes.grid('both', alpha=0.5)
    
    if cax is not None:
        if color_levels is None:
            plt.colorbar(quadmesh, cax=cax)
        else:
            plt.colorbar(quadmesh, cax=cax, ticks=color_levels)
        
    return quadmesh



def plot_vdf_1d_t(axes: plt.Axes, time: np.ndarray|List[datetime], pdf_t: u.Quantity|np.ndarray, v_grid_t: u.Quantity, target_v_unit: int =0, 
                  clip_lower_percentage: float =0, color_norm: Normalize|LogNorm|None =None, core_line: bool|dict =False, cax: plt.Axes|None =None, 
                  pcolormesh_kwargs: dict|None =None) -> QuadMesh:
    
    '''
    Plot 1D VDF time series.
    
    :param axes: The axis to plot.
    :param time: The time array.
    :param pdf_t: The 3D grid of VDF in shape (time, grid_size, grid_size, grid_size).
    :param v_grid_t: The 3D grid of velocity in shape (time, 3, grid_size).
    :param target_v_unit: The unit of velocity that will be plotted. Default is 0, which means the x-axis.
    :param clip_lower_percentage: The percentage of the lower limit of the color normalization. Should be between 0 and 1. Default is 0.
    :param color_norm: The color normalization. Default is None, which means the LogNorm will be used.
    :param core_line: Whether to plot the core line. Could be a dictionary of the line properties. Default is False.
    :param cax: The color bar axis. Default is None.
    :param pcolormesh_kwargs: The keyword arguments for the pcolormesh function. Default is None, which means {'cmap': 'jet', 'shading': 'auto'}.
    
    :return quadmesh: The quadmesh of the plot.
    '''
    
    if isinstance(pdf_t, np.ndarray):
        pdf_t = pdf_t * u.dimensionless_unscaled
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {'cmap': 'jet', 'shading': 'auto'}
        
    v_grid_index = [i for i in range(len(pdf_t.shape) - 1)]
    v_grid_index.remove(target_v_unit)
    
    v_grid_t = v_grid_t.to(u.km/u.s).to_value()
    pdf_1d_t = np.sum(pdf_t, axis=(v_grid_index[0] + 1, v_grid_index[1] + 1)).to_value()
    pdf_1d_t_max_index = np.argmax(pdf_1d_t, axis=1)
    
    if color_norm is None:
        low_level = np.log10(np.min(pdf_1d_t[pdf_1d_t > 0])) + (np.log10(np.max(pdf_1d_t)) - np.log10(np.min(pdf_1d_t[pdf_1d_t > 0]))) * clip_lower_percentage
        high_level = np.log10(np.max(pdf_1d_t))
        color_levels = np.logspace(int(low_level), int(high_level), (int(high_level) + 1 - int(low_level)))
        color_norm = LogNorm(vmin=10**low_level, vmax=10**high_level)
    else:
        color_levels = None
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # for i in range(pdf_1d_t.shape[0]):
        #     quadmesh = axes.pcolormesh([time[i], time[i] + (time[i+1] - time[i])/2 if i < pdf_1d_t.shape[0] - 1 else time[i] + (time[i] - time[i-1])/2], v_grid_t[i, target_v_unit, :], pdf_1d_t[[i, i], :].T, norm=color_norm, **pcolormesh_kwargs)
        quadmesh = plot_mesh1d_ts(axes, time, v_grid_t[:, target_v_unit, :], pdf_1d_t, norm=color_norm, **pcolormesh_kwargs)

        if core_line is True:
            axes.plot(time, np.array([v_grid_t[i, target_v_unit, pdf_1d_t_max_index[i]] for i in range(len(time))]).reshape(-1), 'w-', linewidth=2)
            axes.plot(time, np.array([v_grid_t[i, target_v_unit, pdf_1d_t_max_index[i]] for i in range(len(time))]).reshape(-1), 'k-', linewidth=1)
        elif isinstance(core_line, dict):
            axes.plot(time, np.array([v_grid_t[i, target_v_unit, pdf_1d_t_max_index[i]] for i in range(len(time))]).reshape(-1), **core_line)
        
    axes.grid('both', alpha=0.5)
    
    if cax is not None:
        if color_levels is None:
            plt.colorbar(quadmesh, cax=cax)
        else:
            plt.colorbar(quadmesh, cax=cax, ticks=color_levels)
    
    return quadmesh