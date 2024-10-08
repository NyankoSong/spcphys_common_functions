import numpy as np
from scipy.constants import physical_constants

from . import config
from .utils import check_parameters


@check_parameters
def Sph2Cart(azimuth: np.ndarray, 
             elevation: np.ndarray, 
             energy: np.ndarray, 
             vdf: np.ndarray, 
             v_unit_new: np.ndarray|None=None
             ) -> np.ndarray:
    '''
    This function calculates the 3D scatters of VDF in the new coordinate system. (Only tested for SolO data)
    
    :param azimuth: Azimuth angles in degrees.
    :param elevation: Elevation angles in degrees.
    :param energy: Energy in eV.
    :param vdf: VDF data in shape (time, azimuth, elevation, energy).
    :param v_unit_new: Base vectors of the new coordinate system in shape (time, 3, 3), where the last dimension is [e_ix, e_iy, e_iz]. Default is None, which means the original base vectors [[1,0,0],[0,1,0],[0,0,1]] is used.
    
    :return: Array of 3D scatters of VDF in shape (time, azimuth*elevation*energy, 4), where the last dimension is [v_x, v_y, v_z, f(v)].
    '''
    
    if config.ENABLE_VALUE_CHECKING:
        
        if azimuth.ndim != 1 or elevation.ndim != 1 or energy.ndim != 1:
            raise ValueError("Azimuth, elevation, and energy must be 1-dimensional arrays.")
        if vdf.ndim != 4 or vdf.shape[1] != len(azimuth) or vdf.shape[2] != len(elevation) or vdf.shape[3] != len(energy):
            raise ValueError("VDF must be a 4-dimensional array with shape (time, azimuth, elevation, energy).")
        if v_unit_new is not None and v_unit_new.shape != (vdf.shape[0], 3, 3):
            raise ValueError("v_unit_new must have shape (time, 3, 3).")
    
    if v_unit_new is None:
        v_unit_new = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3), (vdf.shape[0], 1, 1))
    
    vdf_array = np.zeros((vdf.shape[0], len(azimuth)*len(elevation)*len(energy), 4))
    v = np.sqrt(2*energy / physical_constants['proton mass'][0])
    
    for i in range(vdf.shape[0]):
        vdf_tmp = vdf[i, :, :, :]
        
        v_unit_tmp = np.array([np.tile(np.cos(np.deg2rad(azimuth)).reshape(-1, 1), (1, len(elevation))) * np.tile(np.cos(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1)), 
                               np.tile(np.sin(np.deg2rad(azimuth)).reshape(-1, 1), (1, len(elevation))) * np.tile(np.cos(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1)),
                               np.tile(np.sin(np.deg2rad(elevation)).reshape(1, -1), (len(azimuth), 1))])
        v_unit_tmp = np.transpose(v_unit_tmp, (1, 2, 0))
        
        v_tmp = np.zeros((len(azimuth)*len(elevation)*len(energy), v_unit_tmp.shape[-1]))
        for j in range(v_unit_tmp.shape[-1]): 
            v_tmp[:, j] = (np.tile(v.reshape(1, 1, -1), (len(azimuth), len(elevation), 1)) * np.tile((np.dot(v_unit_tmp, v_unit_new[i, j, :])/np.linalg.norm(v_unit_new[i, j, :])).reshape(len(azimuth), len(elevation), -1), (1, 1, len(energy)))).reshape(-1)

        vdf_tmp_array = np.hstack((v_tmp, vdf_tmp.reshape(-1, 1)))
            
        vdf_tmp_array[vdf_tmp_array[:, -1] < 0, -1] = 0
        
        vdf_array[i, :, :] = vdf_tmp_array
    
    return vdf_array