'''Module for plotting tools.'''

from typing import List, Iterable
from datetime import datetime, timedelta
import warnings
import numpy as np
from astropy import units as u
from astropy import stats as astats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import QuadMesh
from scipy.signal import convolve2d
from scipy import stats

from ..utils.utils import check_parameters


def _determine_bins(x, bins, scale):
    x = x[~np.isnan(x)]
    if scale == 'linear':
        _, bins = astats.histogram(x, bins=bins)
    elif scale == 'log':
        _, bins = astats.histogram(np.log10(x), bins=bins)
        bins = 10**bins
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    return bins


def _log_or_linear_plot(scales: List[str], ax: plt.Axes =None):
    
    if scales[0] == 'log' and scales[1] == 'log':
        return plt.loglog if ax is None else ax.loglog
    elif scales[0] == 'log' and scales[1] == 'linear':
        return plt.semilogx if ax is None else ax.semilogx
    elif scales[0] == 'linear' and scales[1] == 'log':
        return plt.semilogy if ax is None else ax.semilogy
    else:
        return plt.plot if ax is None else ax.plot


def _logarithmic_error(x):
    log_x = np.log10(x)
    log_x_mean = np.nanmean(log_x, axis=0)
    unlog_x_mean = 10**log_x_mean
    log_x_std = np.nanstd(log_x, axis=0)
    # This method is not used in this way, but is used for observational data without raw data,
    # such as when only the mean and standard deviation are provided
    x_cap = [np.abs(unlog_x_mean - 10**(log_x_mean - log_x_std)),  np.abs(unlog_x_mean - 10**(log_x_mean + log_x_std))]
    return unlog_x_mean, log_x_std, x_cap


def _mean_std_params(x, scale):
    if scale == 'log':
        unlog_x_mean, log_x_std, x_cap = _logarithmic_error(x)
        x_label = r'$x=10\^($' + f'{np.log10(unlog_x_mean):.2f}' + r'$\pm$' + f'{log_x_std:.2f}' + r'$)$'
    else:
        x_mean, x_err = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        x_cap = [x_err, x_err]
        x_label = r'$x=$' + f'{x_mean:.2f}' + r'$\pm$' + f'{x_err:.2f}'
    return unlog_x_mean, x_cap, x_label


def _mean_std_line_params(x, y, x_edges, scale):
    unlog_y_means = []
    y_caps = []
    if scale == 'log':
        for i in range(len(x_edges)-1):
            unlog_y_mean, _, y_cap = _logarithmic_error(y[(x > x_edges[i]) & (x < x_edges[i+1])])
            unlog_y_means.append(unlog_y_mean)
            y_caps.append(y_cap)
    else:
        for i in range(len(x_edges)-1):
            y_window = y[(x > x_edges[i]) & (x < x_edges[i+1])]
            y_mean, y_err = np.nanmean(y_window), np.std(y_window)
            unlog_y_means.append(y_mean)
            y_caps.append([y_err, y_err])

    return np.array(unlog_y_means), np.array(y_caps).T


@check_parameters
def plot_hist2d(axes: plt.Axes, x: np.ndarray|u.Quantity, y: np.ndarray|u.Quantity, z: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1, scales: list|tuple|str='linear', norm_type: str|None=None,
                color_norm_type: str='linear', color_norm_range: list|tuple|None=None, bins: int|np.ndarray|list|str='freedman', hist_pcolormesh_kwargs: dict|None=None,
                contour_levels: list|np.ndarray|None=None, contour_smooth: int|float|None=None, contour_kwargs: dict|None=None,
                fit_line: bool=False, fit_line_plot_kwargs: dict|None=None,
                mean_std: bool=False, mean_std_errorbar_kwargs: dict|None=None,
                separate: str|None=None,
                mean_std_line: bool=False, mean_std_line_kwargs: dict|None=None,
                ) -> QuadMesh:
    
    '''
    Plot a 2D histogram. If z is provided, overplot z as color.

    :param axes: The matplotlib axes to plot on.
    :param x: The x-axis data, can be a numpy array or an astropy Quantity.
    :param y: The y-axis data, can be a numpy array or an astropy Quantity.
    :param z: The z-axis data, can be a numpy array or an astropy Quantity. Default is None.
    :param least_samples_per_cell: The least number of samples per cell to plot, valid only when z is not None. Default is 0.
    :param scales: The scales for the axes, can be 'linear' or 'log'. Default is 'linear'.
    :param norm_type: The normalization type, can be 'max', 'sum', or None. Default is None.
    :param color_norm_type: The color normalization type, can be 'linear' or 'log'. Default is 'linear'.
    :param color_norm_range: The range for color normalization. Default is None.
    :param bins: The bins for the histogram, can be an integer, numpy array, list, or a string ('freedman', 'scott', 'knuth', 'blocks'). Default is 'freedman'.
    :param hist_pcolormesh_kwargs: Additional arguments for pcolormesh. Default is {'cmap':'jet'}.
    :param contour_levels: The levels for contour plotting. Default is None.
    :param contour_smooth: The smoothing parameter for contours. Default is None.
    :param contour_kwargs: Additional arguments for contour plotting. Default is {}.
    :param fit_line: Whether to fit a line to the data. Default is False.
    :param fit_line_plot_kwargs: Arguments for plotting the fit line. Default is {'color':'k', 'linestyle':'--', 'linewidth':1}.
    :param mean_std: Whether to plot mean and standard deviation. Default is False.
    :param mean_std_errorbar_kwargs: Arguments for plotting error bars of mean and standard deviation. Default is {'fmt':'none', 'color':'k', 'capsize':2}.
    :param separate: Whether to normalize separately along 'x' or 'y' axis. Default is None.
    :param mean_std_line: Whether to plot a line for mean and standard deviation. Default is False.
    :param mean_std_line_kwargs: Arguments for plotting the mean and standard deviation line. Default is {'fmt':'o', 'ms':2, 'color':'k', 'capsize':2, 'linewidth':1, 'linestyle':'-'}.
    
    :return quadmesh: The QuadMesh object of the 2D histogram.
    '''
    
    if least_samples_per_cell < 1:
        raise ValueError("least_samples_per_cell must be a positive integer.")
    if norm_type and norm_type not in ['max', 'sum']:
        raise ValueError("norm_type must be 'max', 'sum' or None.")
    if color_norm_type not in ['log', 'linear']:
        raise ValueError("color_norm_type must be 'log' or 'linear'.")
    if isinstance(bins, str) and bins not in ['freedman', 'scott', 'knuth', 'blocks']:
        raise ValueError("bins must be 'freedman', 'scott', 'knuth', 'blocks', an integer, a numpy array, or a list.")
    if separate and separate not in ['x', 'y']:
        raise ValueError("separate must be 'x', 'y' or None.")
        
    if hist_pcolormesh_kwargs is None:
        hist_pcolormesh_kwargs = {'cmap':'jet'}
    if contour_kwargs is None:
        contour_kwargs = {}
    if fit_line_plot_kwargs is None:
        fit_line_plot_kwargs = {'color':'k', 'linestyle':'--', 'linewidth':1}
    if mean_std_errorbar_kwargs is None:
        mean_std_errorbar_kwargs = {'fmt':'none', 'color':'k', 'capsize':2}
    if mean_std_line_kwargs is None:
        mean_std_line_kwargs = {'fmt':'o', 'ms':2, 'color':'k', 'capsize':2, 'linewidth':1, 'linestyle':'-'}
        

    if isinstance(x, u.Quantity):
        x = x.value
    if isinstance(y, u.Quantity):
        y = y.value
    
    
    bins = [bins, bins] if isinstance(bins, str|int) else bins
    scales = [scales, scales] if isinstance(scales, str) and z is None else [scales, scales, scales] if isinstance(scales, str) else scales
    
    bins_x = _determine_bins(x, bins[0], scales[0])
    bins_y = _determine_bins(y, bins[1], scales[1])
    
    z_hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y])
    if len(z_hist.shape) < 2:
        raise ValueError("No enough data to plot 2D histogram.")
        
    x_mid = (x_edges[1:] + x_edges[:-1]) / 2
    y_mid = (y_edges[1:] + y_edges[:-1]) / 2
    
    if z is not None:
        z_scaled = np.log10(z) if scales[2] == 'log' else z
        z_mat = np.full(z_hist.shape, np.nan)
        for j in range(len(x_edges) - 1):
            for k in range(len(y_edges) - 1):
                z_jk = z_scaled[(x > x_edges[j]) & (x <= x_edges[j + 1]) & (y > y_edges[k]) & (y <= y_edges[k + 1])]
                z_mat[j, k] = np.nanmean(z_jk) if len(z_jk) >= least_samples_per_cell else np.nan
        z_mat = 10**z_mat if scales[2] == 'log' else z_mat
        
        if scales[2] == 'log':
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(z_mat), vmax=np.nanmax(z_mat))
        elif scales[2] == 'linear':
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(z_mat), vmax=np.nanmax(z_mat))
            
        quadmesh = axes.pcolormesh(x_edges, y_edges, z_mat.T, norm=color_norm, **hist_pcolormesh_kwargs)
        
    else:
        if norm_type == 'max':
            if separate == 'x':
                z_hist /= np.nanmax(z_hist, axis=1, keepdims=True)
            elif separate == 'y':
                z_hist /= np.nanmax(z_hist, axis=0, keepdims=True)
            else:
                z_hist /= np.nanmax(z_hist)
        elif norm_type == 'sum':
            if separate == 'x':
                z_hist /= np.nansum(z_hist, axis=1, keepdims=True)
            elif separate == 'y':
                z_hist /= np.nansum(z_hist, axis=0, keepdims=True)
            else:
                z_hist /= np.nansum(z_hist)
        
        if color_norm_type == 'log':
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(z_hist[z_hist > 0]), vmax=np.nanmax(z_hist))
        elif color_norm_type == 'linear':
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(z_hist), vmax=np.nanmax(z_hist))
        
        z_masked = np.ma.masked_where(z_hist == 0, z_hist)
        quadmesh = axes.pcolormesh(x_edges, y_edges, z_masked.T, norm=color_norm, **hist_pcolormesh_kwargs)
    
        if not separate:
            if contour_levels:
                if contour_smooth:
                    if isinstance(contour_smooth, int) or int(contour_smooth) == contour_smooth:
                        z_hist = convolve2d(z_hist, np.ones((contour_smooth, contour_smooth)), mode='same')
                    elif isinstance(contour_smooth, float):
                        z_hist = convolve2d(z_hist, np.ones((int(contour_smooth*z_hist.shape[0]), int(contour_smooth*z_hist.shape[1]))), mode='same')
                axes.contour(x_mid, y_mid, z_hist.T, levels=contour_levels, **contour_kwargs)
            
            if fit_line:
                if scales[0] == 'log':
                    x_fit = np.log10(x)
                else:
                    x_fit = x
                if scales[1] == 'log':
                    y_fit = np.log10(y)
                else:
                    y_fit = y
                    
                slope, intercept, r_value, p_value, _ = stats.linregress(x_fit, y_fit)
                x_fitted = np.array([np.nanmin(x_fit), np.nanmax(x_fit)])
                y_fitted = slope * x_fitted + intercept
                
                if scales[0] == 'log':
                    x_fitted = 10**x_fitted
                    right_label = f'{slope:.2f}' + r'$\log_{10}x$'
                else:
                    right_label = f'{slope:.2f}' + r'$x$'
                if scales[1] == 'log':
                    y_fitted = 10**y_fitted
                    left_label = r'$\log_{10}y=$'
                else:
                    left_label = r'$y=$'
                    
                end_label =  (r'$+$' if intercept > 0 else '') + f'{intercept:.2f}\n' + r'$r\approx$' + f'{r_value:.2f}' + r'$\ \ p\geqslant$' + f'{p_value:.2f}'
                
                _log_or_linear_plot(scales, axes)(x_fitted, y_fitted, label=left_label+right_label+end_label, **fit_line_plot_kwargs)
            
            if mean_std:
                unlog_x_mean, x_cap, x_label = _mean_std_params(x, scales[0])
                unlog_y_mean, y_cap, y_label = _mean_std_params(y, scales[1])
                    
                axes.errorbar(unlog_x_mean, unlog_y_mean, xerr=x_cap, yerr=y_cap, label=x_label+'\n'+y_label, **mean_std_errorbar_kwargs)
            
        else:
            if mean_std_line:
                if separate == 'x':
                    unlog_y_means, y_caps = _mean_std_line_params(x, y, x_edges, scales[1])
                    axes.errorbar(x_mid, unlog_y_means, yerr=y_caps, **mean_std_line_kwargs)
                elif separate == 'y':
                    unlog_x_means, x_caps = _mean_std_line_params(y, x, y_edges, scales[0])
                    axes.errorbar(unlog_x_means, y_mid, xerr=x_caps, **mean_std_line_kwargs)
                    
    
        
    if scales[0] == 'log':
        axes.set_xscale('log')
    if scales[1] == 'log':
        axes.set_yscale('log')
        
    return quadmesh


def _auto_downsample(axes: plt.Axes, x: np.ndarray|list|u.Quantity, y: np.ndarray|u.Quantity, max_dens: int = 1000):
    """
    Downsamples the data points to ensure the density of points on the plot does not exceed a specified maximum density.

    :param axes: The matplotlib axes object where the data will be plotted.
    :param x: The x-coordinates of the data points. Can be a list of datetime objects.
    :param y: The y-coordinates of the data points.
    :param max_dens: The maximum allowed number of points per inch on the plot.
    
    :return x: The downsampled x (preserves original datetime type if applicable).
    :return y: The downsampled y.
    """
    examine_size = 10  # axes_width / examine_size

    # Convert x to numpy array and handle datetime types
    x_original = np.asarray(x) if isinstance(x, list) else x.copy()
    x_num = x_original  # Initialize numerical x for histogram

    # Handle datetime conversion to numerical values
    if isinstance(x_original, np.ndarray) and np.issubdtype(x_original.dtype, np.datetime64):
        x_num = x_original.astype('datetime64[ns]').astype('float64')
    elif isinstance(x_original, np.ndarray) and x_original.dtype == object and len(x_original) > 0:
        if isinstance(x_original.flat[0], datetime):
            x_num = np.array([dt.timestamp() for dt in x_original.ravel()]).reshape(x_original.shape)

    # Handle Quantity types
    if isinstance(x_original, u.Quantity):
        x_num = x_original.value

    # Mask NaN values based on y
    nan_mask = np.isnan(y) if len(y.shape) == 1 else np.isnan(y).all(axis=1)
    y_nan = y[nan_mask]
    y_non_nan = y[~nan_mask]
    x_non_nan_original = x_original[~nan_mask]
    x_non_nan_num = x_num[~nan_mask]

    # Calculate histogram bins based on numerical x values
    axes_width = axes.get_position().width * axes.get_figure().get_size_inches()[0]
    hists, bin_edges = np.histogram(x_non_nan_num, bins=int(axes_width)*examine_size)
    high_density_bins = np.where(hists > max_dens/examine_size)[0]

    if len(high_density_bins) == 0:
        return x_original, y

    low_density_mask = np.ones(len(x_non_nan_num), dtype=bool)
    downsampled_indices = []

    # Downsample high-density bins
    for bin_index in high_density_bins:
        bin_mask = (x_non_nan_num >= bin_edges[bin_index]) & (x_non_nan_num < bin_edges[bin_index + 1])
        indices = np.flatnonzero(bin_mask)
        if len(indices) == 0:
            continue
        low_density_mask[indices] = False
        num_samples = int(max_dens/examine_size)
        if num_samples <= 0:
            num_samples = 1
        downsampled_indices.extend(np.linspace(indices[0], indices[-1], num_samples, dtype=int))

    # Combine indices and sort
    non_nan_indices = np.concatenate([np.flatnonzero(low_density_mask), downsampled_indices]).astype(int)
    non_nan_indices = np.sort(non_nan_indices)

    # Reconstruct final arrays preserving original types
    final_x_unsorted = np.concatenate([x_non_nan_original[non_nan_indices], x_original[nan_mask]])
    final_indices = np.argsort(final_x_unsorted)
    final_x = final_x_unsorted[final_indices]
    final_y = np.concatenate([y_non_nan[non_nan_indices], y_nan])[final_indices]

    return final_x, final_y


def plot(x: np.ndarray|list|u.Quantity, y: np.ndarray|u.Quantity, axes: plt.Axes|None =None, scale: str='linear', auto_downsample: bool =True, max_dens: int =1000, **plot_kwargs):
    '''
    Plot x and y data.
    
    :param x: The x-axis data.
    :param y: The y-axis data.
    :param axes: The matplotlib axes to plot on. Default is None.
    :param scale: The scale for the plot, can be 'linear' or 'log'. Default is 'linear'.
    :param auto_downsample: Whether to automatically downsample the data. Default is True.
    :param max_dens: The maximum number of data points per inch. Default is 1000.
    :param plot_kwargs: Additional arguments for the plot function.
    '''
    
    if axes is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        ax = axes
    
    if scale == 'log':
        plot_func = ax.semilogy
    elif scale == 'linear':
        plot_func = ax.plot
    
    if auto_downsample:
        x, y = _auto_downsample(ax, x, y, max_dens)
        
    plot_func(x, y, **plot_kwargs)



def plot_mesh1d_ts(axes: plt.Axes, t: np.ndarray|List[datetime], y: np.ndarray|u.Quantity|List[np.ndarray|u.Quantity], z: np.ndarray|u.Quantity|List[np.ndarray|u.Quantity],
                   width: List[timedelta]|np.ndarray|timedelta|float|None=None, norm: str|LogNorm|Normalize ='linear', **pcolormesh_kwargs) -> QuadMesh:
    
    '''
    Plot a 1D mesh plot with time as the x-axis.
    
    :param axes: The matplotlib axes to plot on.
    :param t: The time data.
    :param y: The y-axis data in shape (t, y) or (y,). Can be a list if length of y varies with time.
    :param z: The z-axis data in shape (t, y). Can be a list if length of y varies with time.
    '''

    INVALID_DIFF_PERCENT = 0.1
    
    t = np.asarray(t) if isinstance(t, list) else t
    
    if width is None:
        t_diff = np.diff(t)
        t_diff_mean = np.mean(t_diff)
        if not all(t_diff < t_diff_mean * (1 + INVALID_DIFF_PERCENT)) or not all(t_diff > t_diff_mean * (1 - INVALID_DIFF_PERCENT)):
            warnings.warn("The time intervals are not equal. Using the difference of every two points as width.")
        width = np.concatenate((t_diff, [t_diff[0]]))
            
    elif isinstance(width, Iterable):
        if len(width) != len(t):
            raise ValueError("The width must have the same length as t - 1.")
        width = np.asarray(width)
        
    else:
        width = np.full(len(t), width)
            
    if norm == 'linear':
        norm = Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    elif norm == 'log':
        norm = LogNorm(vmin=np.nanmin(z[z > 0]), vmax=np.nanmax(z))
            
    if 'norm' not in pcolormesh_kwargs:
        pcolormesh_kwargs['norm'] = norm
    
    if isinstance(y[0], Iterable):
        for i in range(t.shape[0]):
            yi_valid_indices = np.isfinite(y[i])
            if yi_valid_indices.sum() < 2:
                continue
            quadmesh = axes.pcolormesh([t[i] + width[i]/4, t[i] + width[i]/4*3], y[i, yi_valid_indices], np.vstack(([z[i, yi_valid_indices], z[i, yi_valid_indices]])).T, **pcolormesh_kwargs)
    else:
        quadmesh = axes.pcolormesh(t + width/2, y, z.T, **pcolormesh_kwargs)
            
    return quadmesh