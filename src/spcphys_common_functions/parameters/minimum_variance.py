"""
Minimum Variance Analysis (MVA) for space physics data.

Modified from https://github.com/spedas/pyspedas/blob/master/pyspedas/cotrans_tools/minvar.py
"""

import numpy as np


def min_var(data: np.ndarray, verbose: bool = False):
    """Compute the principal variance directions and variances of a vector quantity.
    
    This function performs Minimum Variance Analysis (MVA) to find the principal
    axes of variance for a set of vector measurements.

    :param data: Input data array with shape (npoints, 3), where npoints is the
                 number of measurements and 3 represents the vector components (x, y, z)
    :type data: numpy.ndarray
    :param verbose: If True, print diagnostic information including average direction,
                    angle to minimum variance direction, and eigenvalue/eigenvector info,
                    defaults to False
    :type verbose: bool, optional
    :return: Tuple containing:
             - vrot: Rotated data in the new coordinate system with shape (npoints, 3).
               vrot[:, 0] is the maximum variance direction,
               vrot[:, 1] is the intermediate variance direction,
               vrot[:, 2] is the minimum variance direction.
             - v: Eigenvector matrix with shape (3, 3) containing the principal axes.
               v[:, 0] is the maximum variance direction eigenvector,
               v[:, 1] is the intermediate variance direction eigenvector,
               v[:, 2] is the minimum variance direction eigenvector.
             - w: Eigenvalues array with shape (3,) in descending order.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
            
    #  Min var starts here
    # data must be Nx3
    vecavg = np.nanmean(np.nan_to_num(data, nan=0.0), axis=0)

    # Vectorized computation of covariance matrix
    data_clean = np.nan_to_num(data, nan=0.0)
    mvamat = (data_clean.T @ data_clean) / data_clean.shape[0] - np.outer(vecavg, vecavg)

    # Calculate eigenvalues and eigenvectors
    w, v = np.linalg.eigh(mvamat, UPLO='U')

    # Sorting to ensure descending order
    w = np.abs(w)
    idx = np.flip(np.argsort(w))

    # IDL compatability
    if True:
        if np.sum(w) == 0.0:
            idx = [0, 2, 1]

    w = w[idx]
    v = v[:, idx]

    # Rotate intermediate var direction if system is not Right Handed
    YcrossZdotX = v[0, 0] * (v[1, 1] * v[2, 2] - v[2, 1] * v[1, 2])
    if YcrossZdotX < 0:
        v[:, 1] = -v[:, 1]
        # v[:, 2] = -v[:, 2] # Should not it is being flipped at Z-axis?

    # 以下内容疑似与所谓的FAC系统有关，暂时不考虑
    # # Ensure minvar direction is along +Z (for FAC system)
    # if v[2, 2] < 0:
    #     v[:, 2] = -v[:, 2]
    #     v[:, 1] = -v[:, 1]

    # Ensure minvar-Z and intvar-Z are both positive, to ensure matching results between IDL and Python
    if v[2, 1] < 0:
        v[:, 1] = -v[:, 1]
        v[:, 0] = -v[:, 0]

    vrot = data @ v

    if verbose:
        data_ave = np.mean(data, axis=0)
        print('avedir= \t', data_ave/np.linalg.norm(data_ave))
        print('theta_kB= \t', np.arccos(np.dot(data_ave, v[:, 2])/(np.linalg.norm(data_ave) * np.linalg.norm(v[:, 2])))*180/np.pi)
        print('mindir= %10.2f'%w[2], np.round(v[:, 2], decimals=4))
        print('intdir= %10.2f'%w[1], np.round(v[:, 1], decimals=4))
        print('maxdir= %10.2f'%w[0], np.round(v[:, 0], decimals=4))

    return vrot, v, w