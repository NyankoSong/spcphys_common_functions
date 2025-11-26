from typing import Tuple
import warnings
from astropy import units as u
import numpy as np



def vec_cart_to_sph(v: u.Quantity|np.ndarray, r: u.Quantity|np.ndarray, z: u.Quantity|np.ndarray|None =None) ->Tuple[u.Quantity|np.ndarray]:
    """Convert a vector from Cartesian coordinates to spherical coordinates.

    :param v: The vector to be converted. Shape should be (N, 3), where N is the number of vectors
    :type v: astropy.units.Quantity or numpy.ndarray
    :param r: The radial component of the vector. Shape should be (N, 3) or (3,)
    :type r: astropy.units.Quantity or numpy.ndarray
    :param z: The z-component of the vector. Shape should be (N, 3) or (3,), defaults to None
    :type z: astropy.units.Quantity or numpy.ndarray or None, optional
    :return: If z is None, returns the magnitude and angle between v and r. Otherwise, returns the magnitude, azimuth, and elevation
    :rtype: Tuple[astropy.units.Quantity or numpy.ndarray]
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
    
    

def vec_sph_to_cart(v_mag: u.Quantity|np.ndarray, azimuth: u.Quantity, elevation: u.Quantity|None = None) -> Tuple[u.Quantity|np.ndarray]:
    """Convert a vector from spherical coordinates to Cartesian coordinates.

    :param v_mag: The magnitude of the vector. Shape should be (N,)
    :type v_mag: astropy.units.Quantity or numpy.ndarray
    :param azimuth: The azimuth angle of the vector in degrees. Shape should be (N,)
    :type azimuth: astropy.units.Quantity
    :param elevation: The elevation angle of the vector in degrees. Shape should be (N,), defaults to None
    :type elevation: astropy.units.Quantity or None, optional
    :return: If elevation is None, returns the x and y components of the vector. Otherwise, returns the x, y, and z components
    :rtype: Tuple[astropy.units.Quantity or numpy.ndarray]
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
    
    
def quat_rot_vec(q: u.Quantity|np.ndarray, v: u.Quantity|np.ndarray, w_loc: int =0, q_normalization: bool =False) -> u.Quantity|np.ndarray:
    """Rotate vectors using a quaternion.

    :param q: The quaternion representing the rotation. Shape should be (N, 4), where N is the number of quaternions
    :type q: astropy.units.Quantity or numpy.ndarray
    :param v: The vectors to be rotated. Shape should be (N, 3)
    :type v: astropy.units.Quantity or numpy.ndarray
    :param w_loc: The index of the scalar component in the quaternion, defaults to 0
    :type w_loc: int, optional
    
    :return: The rotated vectors. Shape will be (N, 3)
    :rtype: astropy.units.Quantity or numpy.ndarray
    """
    
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError("q should be (N, 4), v should be (N, 3)")
    if q.shape[0] == 1 and v.shape[0] > 1: 
        warnings.warn("Broadcasting quaternion to match number of vectors.", UserWarning)
        q = np.repeat(q, v.shape[0], axis=0)
    if v.shape[0] == 1 and q.shape[0] > 1 or v.shape[0] != q.shape[0]: 
        # warnings.warn("Broadcasting vector to match number of quaternions.", UserWarning)
        # v = np.repeat(v, q.shape[0], axis=0)
        
        # This case may not appear often in practice, so we raise an error instead of warning and broadcasting.
        raise ValueError("Number of vectors must match number of quaternions.")
    if w_loc not in [0, 3, -1]:
        raise ValueError("w_loc must be either 0 or 3 (or -1).")
    
    if q_normalization:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    else:
        q_norms = np.linalg.norm(q, axis=1)
        if not np.allclose(q_norms, 1.0, 1e-3):
            warnings.warn("Input quaternions are not normalized and may produce incorrect results.", UserWarning)
    
    if w_loc == 0:
        q_w, q_x, q_y, q_z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif w_loc == 3 or w_loc == -1:
        q_w, q_x, q_y, q_z = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    
    v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]
    t2, t3, t4 = q_w*q_x, q_w*q_y, q_w*q_z
    t5, t6, t7 = -q_x*q_x, q_x*q_y, q_x*q_z
    t8, t9, t10 = -q_y*q_y, q_y*q_z, -q_z*q_z

    v_rot_x = 2 * ((t8 + t10) * v_x + (t6 - t4) * v_y + (t3 + t7) * v_z) + v_x
    v_rot_y = 2 * ((t4 + t6) * v_x + (t5 + t10) * v_y + (t9 - t2) * v_z) + v_y
    v_rot_z = 2 * ((t7 - t3) * v_x + (t2 + t9) * v_y + (t5 + t8) * v_z) + v_z
    
    v_rot = np.column_stack((v_rot_x, v_rot_y, v_rot_z))
    
    return v_rot