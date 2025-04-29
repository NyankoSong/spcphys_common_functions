'''Modified from https://github.com/spedas/pyspedas/blob/master/pyspedas/cotrans_tools/minvar.py'''

import numpy as np





def min_var(data: np.ndarray, verbose: bool=False):
    
    """
    This program computes the principal variance directions and variances of a
    vector quantity as well as the associated eigenvalues.

    Parameters
    -----------
    data:
        Vxyz, an (npoints, ndim) array of data(ie Nx3)

    Returns
    -------
    vrot:
        an array of (npoints, ndim) containing the rotated data in the new coordinate system, ijk.
        Vi(maximum direction)=vrot[0,:]
        Vj(intermediate direction)=vrot[1,:]
        Vk(minimum variance direction)=Vrot[2,:]
    v:
        an (ndim,ndim) array containing the principal axes vectors
        Maximum variance direction eigenvector, Vi=v[*,0]
        Intermediate variance direction, Vj=v[*,1] (descending order)
    w:
        the eigenvalues of the computation
    """
            
    #  Min var starts here
    # data must be Nx3
    vecavg = np.nanmean(np.nan_to_num(data, nan=0.0), axis=0)

    mvamat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mvamat[i, j] = np.nanmean(np.nan_to_num(data[:, i] * data[:, j], nan=0.0)) - vecavg[i] * vecavg[j]

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

    vrot = np.array([np.dot(row, v) for row in data])

    if verbose:
        data_ave = np.mean(data, axis=0)
        print('avedir= \t', data_ave/np.linalg.norm(data_ave))
        print('theta_kB= \t', np.arccos(np.dot(data_ave, v[:, 2])/(np.linalg.norm(data_ave) * np.linalg.norm(v[:, 2])))*180/np.pi)
        print('mindir= %10.2f'%w[2], np.round(v[:, 2], decimals=4))
        print('intdir= %10.2f'%w[1], np.round(v[:, 1], decimals=4))
        print('maxdir= %10.2f'%w[0], np.round(v[:, 0], decimals=4))

    return vrot, v, w