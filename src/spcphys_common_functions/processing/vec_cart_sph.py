from typing import Tuple
from astropy import units as u
import numpy as np

from ..utils.utils import check_parameters


@check_parameters
def vec_cart_to_sph(v: u.Quantity|np.ndarray, r: u.Quantity|np.ndarray, z: u.Quantity|np.ndarray|None =None) ->Tuple[u.Quantity|np.ndarray]:
    
    """
    Convert a vector from Cartesian coordinates to spherical coordinates.

    :param v: The vector to be converted. Shape should be (N, 3), where N is the number of vectors.
    :param r: The radial component of the vector. Shape should be (N, 3) or (3,).
    :param z: The z-component of the vector. Shape should be (N, 3) or (3,). If None, the function will only compute the magnitude and angle between v and r. Default is None.
    
    :return: If z is None, returns the magnitude and angle between v and r. Otherwise, returns the magnitude, azimuth, and elevation.
    """
        
    if len(r.shape) == 1 or r.shape[0] == 1:
        r = np.tile(r, (v.shape[0], 1))
    r = r / np.tile(np.linalg.norm(r, axis=1), (3, 1)).T
    
    v_mag = np.linalg.norm(v, axis=1)
    
    if z is None:
        theta = np.rad2deg(np.arccos(np.einsum('ij,ij->i', v, r) / v_mag))
            
        return v_mag, theta
    
    else:
        if len(z.shape) == 1 or z.shape[0] == 1:
            z = np.tile(z, (v.shape[0], 1))
        z = z / np.tile(np.linalg.norm(z, axis=1), (3, 1)).T
        
        y = np.cross(z, r)
        
        v_r, v_y, v_z = np.einsum('ij,ij->i', v, r), np.einsum('ij,ij->i', v, y), np.einsum('ij,ij->i', v, z)
        
        azimuth = np.rad2deg(np.arctan2(v_y, v_r))
        elevation = np.rad2deg(np.arcsin(v_z / v_mag))
            
        return v_mag, azimuth, elevation
    
    
@check_parameters
def vec_sph_to_cart(v_mag: u.Quantity|np.ndarray, azimuth: u.Quantity, elevation: u.Quantity|None = None) -> Tuple[u.Quantity|np.ndarray]:
    """
    Convert a vector from spherical coordinates to Cartesian coordinates.

    :param v_mag: The magnitude of the vector. Shape should be (N,).
    :param azimuth: The azimuth angle of the vector in degrees. Shape should be (N,).
    :param elevation: The elevation angle of the vector in degrees. Shape should be (N,). If None, the function will assume the vector lies in the xy-plane. Default is None.
    
    :return: If elevation is None, returns the x and y components of the vector. Otherwise, returns the x, y, and z components.
    """
    
    if not (azimuth.unit.is_equivalent(u.deg) or azimuth.unit.is_equivalent(u.rad)):
        raise ValueError("azimuth should be in degrees or radians.")
    if elevation is not None and not (elevation.unit.is_equivalent(u.deg) or elevation.unit.is_equivalent(u.rad)):
        raise ValueError("elevation should be in degrees or radians.")
    
    if azimuth.unit.is_equivalent(u.deg):
        azimuth = np.deg2rad(azimuth)
    
    if elevation is None:
        x = v_mag * np.cos(azimuth)
        y = v_mag * np.sin(azimuth)
        
        return x, y
    
    else:
        elevation = np.deg2rad(elevation)
        
        x = v_mag * np.cos(elevation) * np.cos(azimuth)
        y = v_mag * np.cos(elevation) * np.sin(azimuth)
        z = v_mag * np.sin(elevation)
        
        return x, y, z