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

from . import config
from .utils import check_parameters, _determine_processes


@check_parameters
def vdf_sph_to_cart(azimuth: u.Quantity,
             elevation: u.Quantity,
             energy: u.Quantity,
             vdf: u.Quantity,
             v_unit_new: np.ndarray|None=None,
             ) -> Tuple[u.Quantity, u.Quantity]:
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
    if v_unit_new is not None and v_unit_new.shape != (vdf.shape[0], 3, 3):
        raise ValueError("v_unit_new must have shape (time, 3, 3).")
    if not azimuth.unit.is_equivalent(u.deg) or not elevation.unit.is_equivalent(u.deg) or not energy.unit.is_equivalent(u.J):
        raise ValueError("Azimuth, elevation, and energy must have units equivalent to deg, deg, and J.")
    if not vdf.unit.is_equivalent(u.s**3 / u.m**6):
        warnings.warn("VDF does not have units equivalent to s^3/m^6. Is this phase space density instead of counts?", UserWarning)
    
    energy = energy.si
    
    if v_unit_new is None:
        v_unit_new = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3), (vdf.shape[0], 1, 1))
    
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


@check_parameters
def vdf_griddata(vdf_vec: u.Quantity, vdf_value: u.Quantity, grid_size: int, device: str ='cpu') -> Tuple[u.Quantity, u.Quantity]:
    
    '''
    This function interpolates the 3D scatters of VDF to a 3D grid.
    
    :param vdf_vec: The 3D scatters of VDF in shape (azimuth*elevation*energy, 3).
    :param vdf_value: The values of the 3D scatters of VDF in shape (azimuth*elevation*energy).
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
        v_grid.append(np.linspace(np.min(valid_vdf_vec[:, i]), np.max(valid_vdf_vec[:, i]), grid_size))
        
    v_gridmesh = np.meshgrid(*v_grid)
    v_gridmesh_points = np.hstack((v_gridmesh[0].reshape(-1, 1), v_gridmesh[1].reshape(-1, 1), v_gridmesh[2].reshape(-1, 1)))
    
    for i in range(v_gridmesh_points.shape[-1]):
        v_gridmesh_points = v_gridmesh_points[v_gridmesh_points[:, v_gridmesh_points.shape[1] - i - 1].argsort(kind='mergesort'), :]
    
    if device == 'cpu':
        pdf = sinterp.griddata(vdf_vec.to_value(), vdf_value.to_value(), v_gridmesh_points.to_value(), method='linear', fill_value=0) * valid_vdf_value.unit
        pdf_3d = pdf.reshape(grid_size, grid_size, grid_size)
        
    else:
        raise ValueError('Currently only support cpu.')
    
    return pdf_3d, np.concatenate([v_axis[:, np.newaxis] for v_axis in v_grid], axis=1).T


def _unpack_params(vdf_input):
    return vdf_griddata(vdf_vec=vdf_input[0], vdf_value=vdf_input[1], grid_size=vdf_input[2])

@check_parameters
def vdf_griddata_t(vdf_vec_t: u.Quantity, vdf_value_t: u.Quantity, grid_size: int, device: str ='cpu', multiprocess: int|float =1) -> Tuple[u.Quantity, u.Quantity]:
    
    '''
    This function packs the `vdf_griddata` function to calculate the 3D grid of VDFs.
    
    :param vdf_vec_t: The 3D scatters of VDF in the new coordinate system in shape (time, azimuth*elevation*energy, 3).
    :param vdf_value_t: The values of the 3D scatters of VDF in shape (time, azimuth*elevation*energy).
    :param grid_size: The size of the grid.
    :param device: The device to use. Default is 'cpu'.
    :param multiprocess: Whether to use multiprocess. Default is False.
    
    :return pdf_t: The 3D grid of VDF in shape (time, grid_size, grid_size, grid_size).
    :return v_grid_t: The 3D grid of velocity in shape (time, 3, grid_size).
    '''
    
    if not vdf_vec_t.unit.is_equivalent(u.m/u.s):
        raise ValueError("VDF vectors must have units equivalent to m/s.")
    if vdf_vec_t.shape[0] != vdf_value_t.shape[0] or vdf_vec_t.shape[1] != vdf_value_t.shape[1] or vdf_vec_t.shape[2] != 3 or vdf_value_t.ndim != 2 or vdf_vec_t.ndim != 3:
        raise ValueError("The shapes of VDF vectors must be (time, azimuth*elevation*energy, 3) and VDF values must be (time, azimuth*elevation*energy).")
    if device not in ['cpu', 'gpu']:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")

    pdf_t = np.zeros((vdf_value_t.shape[0], grid_size, grid_size, grid_size)) * vdf_value_t.unit
    v_grid_t = np.zeros((vdf_value_t.shape[0], 3, grid_size)) * vdf_vec_t.unit
    
    vdf_input_t = [(vdf_vec_t[i, :, :], vdf_value_t[i, :], grid_size) for i in range(vdf_value_t.shape[0])]
    
    if multiprocess != 1:
        with Pool(processes=_determine_processes(multiprocess)) as pool:
            result = list(tqdm(pool.imap(_unpack_params, vdf_input_t), total=len(vdf_input_t)))
        for i in range(vdf_value_t.shape[0]):
            pdf_t[i], v_grid_t[i] = result[i]
    else:
        for i in tqdm(range(vdf_value_t.shape[0])):
            pdf_t[i], v_grid_t[i] = vdf_griddata(vdf_vec_t[i, :, :], vdf_value_t[i, :], grid_size, device=device)
        
    return pdf_t, v_grid_t


@check_parameters
def vdf_core(pdf_t: u.Quantity, v_grid_t: u.Quantity) -> u.Quantity:
        
    v_t_max = np.zeros((v_grid_t.shape[0], 3)) * v_grid_t.unit
        
    for (i, n) in enumerate(range(3)):
        v_grid_index = [j for j in range(len(pdf_t.shape) - 1)]
        v_grid_index.remove(n)
        pdf_1d_t = np.sum(pdf_t, axis=(v_grid_index[0] + 1, v_grid_index[1] + 1))
        pdf_1d_t_max_index = np.argmax(pdf_1d_t, axis=1)
        v_t_max[:, i] = np.array([v_grid_t[j, n, pdf_1d_t_max_index[j]] for j in range(v_grid_t.shape[0])]).reshape(-1) * v_grid_t.unit
        
    return v_t_max

@check_parameters
def plot_vdf_2d(axes: plt.Axes, pdf: u.Quantity, v_grid: u.Quantity, compress_v_unit=2, color_norm: Normalize|LogNorm|None =None, color_levels: List[float]|None =None, cax: plt.Axes|None =None) -> QuadMesh:
    
    '''
    Plot the 2D VDF.
    
    :param axes: The axis to plot.
    :param pdf: The 3D grid of VDF in shape (grid_size, grid_size, grid_size).
    :param v_grid: The 3D grid of velocity in shape (3, grid_size).
    :param target_v_unit: The unit of velocity that will not be plotted. Default is 2, which means the z-axis.
    :param color_norm: The color normalization. Default is None, which means the LogNorm will be used.
    :param color_levels: The color levels. Default is None, which means the levels will be calculated automatically.
    
    :return: The figure and axis.
    '''
    
    
    v_grid_index = [i for i in range(len(pdf.shape))]
    v_grid_index.remove(compress_v_unit)
    
    v_grid = v_grid.to(u.km/u.s).to_value()
    pdf_2D = pdf.sum(axis=compress_v_unit).to_value()
    
    if color_norm is None:
        low_level = np.log10(np.min(pdf_2D[pdf_2D > 0])) + (np.log10(np.max(pdf_2D)) - np.log10(np.min(pdf_2D[pdf_2D > 0])))/2
        high_level = np.log10(np.max(pdf_2D))
        color_norm = LogNorm(vmin=10**low_level, vmax=10**high_level)
    if color_levels is None and isinstance(color_norm, LogNorm):
        color_levels = np.logspace(int(low_level), int(high_level), (int(high_level) + 1 - int(low_level)))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        quadmesh = axes.pcolormesh(v_grid[v_grid_index[0]], v_grid[v_grid_index[1]], pdf_2D.T, cmap='jet', norm=color_norm, shading='auto')
        if color_levels is not None:
            axes.contour(v_grid[v_grid_index[0]], v_grid[v_grid_index[1]], pdf_2D.T, colors='k', norm=color_norm, levels=color_levels)
        
    axes.grid('both', alpha=0.5)
    
    if cax is not None:
        if color_levels is None:
            plt.colorbar(quadmesh, cax=cax)
        else:
            plt.colorbar(quadmesh, cax=cax, ticks=color_levels)
        
    return quadmesh


def plot_vdf_1d_t(axes: plt.Axes, time: np.ndarray|List[datetime], pdf_t: u.Quantity, v_grid_t: u.Quantity, target_v_unit: int =0, color_norm: Normalize|LogNorm|None =None, core_line: str|None =None, cax: plt.Axes|None =None) -> QuadMesh:
    
    '''
    Plot 1D VDF time series.
    
    :param axes: The axis to plot.
    :param time: The time array.
    :param pdf_t: The 3D grid of VDF in shape (time, grid_size, grid_size, grid_size).
    :param v_grid_t: The 3D grid of velocity in shape (time, 3, grid_size).
    :param target_v_unit: The unit of velocity that will be plotted. Default is 0, which means the x-axis.
    :param color_norm: The color normalization. Default is None, which means the LogNorm will be used.
    :param core_line: The line style of the core line. Default is None.
    :param cax: The color bar axis. Default is None.
    
    :return: The figure and axis.
    '''
        
    v_grid_index = [i for i in range(len(pdf_t.shape) - 1)]
    v_grid_index.remove(target_v_unit)
    
    v_grid_t = v_grid_t.to(u.km/u.s).to_value()
    if not pdf_1D_t.unit.is_dimensionless():
        pdf_1D_t = np.mean(pdf_t, axis=(v_grid_index[0] + 1, v_grid_index[1] + 1)).to_value()
    else:
        pdf_1D_t = np.sum(pdf_t, axis=(v_grid_index[0] + 1, v_grid_index[1] + 1)).to_value()
    pdf_1D_t_max_index = np.argmax(pdf_1D_t, axis=1)
    
    if color_norm is None:
        low_level = np.log10(np.min(pdf_1D_t[pdf_1D_t > 0])) + (np.log10(np.max(pdf_1D_t)) - np.log10(np.min(pdf_1D_t[pdf_1D_t > 0])))/2
        high_level = np.log10(np.max(pdf_1D_t))
        color_levels = np.logspace(int(low_level), int(high_level), (int(high_level) + 1 - int(low_level)))
        color_norm = LogNorm(vmin=10**low_level, vmax=10**high_level)
    else:
        color_levels = None
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(pdf_1D_t.shape[0]):
            if i == pdf_1D_t.shape[0] - 1:
                d_date = (time[i] - time[i-1])/4
            elif i == 0:
                d_date = (time[i+1] - time[i])/4
            else:
                d_date = (time[i+1] - time[i-1])/8
            quadmesh = axes.pcolormesh([time[i] - d_date, time[i] + d_date], v_grid_t[i, target_v_unit, :], pdf_1D_t[[i, i], :].T, cmap='jet', norm=color_norm, shading='auto')

        if core_line is not None:
            axes.plot(time, np.array([v_grid_t[i, target_v_unit, pdf_1D_t_max_index[i]] for i in range(len(time))]).reshape(-1), core_line, linewidth=1)
        
    axes.grid('both', alpha=0.5)
    
    if cax is not None:
        if color_levels is None:
            plt.colorbar(quadmesh, cax=cax)
        else:
            plt.colorbar(quadmesh, cax=cax, ticks=color_levels)
    
    return quadmesh