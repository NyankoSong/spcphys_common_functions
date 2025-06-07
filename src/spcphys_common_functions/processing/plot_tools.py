'''Module for plotting tools.'''

from typing import List, Iterable, Literal
from datetime import datetime, timedelta
import warnings
import numpy as np
from astropy import units as u
from astropy import stats as astats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from scipy import stats




def _determine_bins(x, bins, scale):
    x = x[~np.isnan(x)]
    if scale == 'linear':
        _, bins = astats.histogram(x, bins=bins)
    elif scale == 'log':
        if any(x <= 0):
            raise ValueError("Logarithmic scale requires all x values to be positive.")
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


def _logarithmic_error(x, mean_func):
    log_x = np.log10(x)
    log_x_mean = mean_func(log_x, axis=0)
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


def _mean_std_line_params(x, y, x_edges, scale, method):
    unlog_y_means = []
    y_caps = []
    if method == 'mean':
        mean_func = np.nanmean
    elif method == 'max':
        mean_func = np.nanmax
    elif method == 'median':
        mean_func = np.nanmedian
    if scale == 'log':
        for i in range(len(x_edges)-1):
            unlog_y_mean, _, y_cap = _logarithmic_error(y[(x > x_edges[i]) & (x < x_edges[i+1])], mean_func)
            unlog_y_means.append(unlog_y_mean)
            y_caps.append(y_cap)
    else:
        for i in range(len(x_edges)-1):
            y_window = y[(x > x_edges[i]) & (x < x_edges[i+1])]
            y_mean, y_err = mean_func(y_window), np.std(y_window)
            unlog_y_means.append(y_mean)
            y_caps.append([y_err, y_err])

    return np.array(unlog_y_means), np.array(y_caps).T


# def plot_hist2d(
#     axes: plt.Axes, x: np.ndarray|u.Quantity, y: np.ndarray|u.Quantity, z: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1, 
#     scales: list|tuple|str='linear', norm_type: Literal['sum', 'max', None]=None,
#     color_norm_type: str='linear', color_norm_range: list|tuple|None=None, bins: int|np.ndarray|list|str='freedman', hist_pcolormesh_kwargs: dict|None=None,
#     contour_only: bool=False, contour_smooth: int|float|None=None, contour_kwargs: dict|None=None,
#     fit_line: bool=False, fit_line_kwargs: dict|None=None,
#     mean_std: bool=False, mean_std_errorbar_kwargs: dict|None=None,
#     separate: str|None=None,
#     mean_std_line: Literal['x', 'y', None] =None, mean_std_line_method: Literal['mean', 'max', 'median'] ='mean', mean_std_line_kwargs: dict|None=None,
# ) -> QuadMesh | QuadContourSet:
    
#     '''
#     Plot a 2D histogram. If z is provided, overplot z as color.

#     :param axes: The matplotlib axes to plot on.
#     :param x: The x-axis data, can be a numpy array or an astropy Quantity.
#     :param y: The y-axis data, can be a numpy array or an astropy Quantity.
#     :param z: The z-axis data, can be a numpy array or an astropy Quantity. Default is None.
#     :param least_samples_per_cell: The least number of samples per cell to plot, valid only when z is not None. Default is 0.
#     :param scales: The scales for the axes, can be 'linear' or 'log'. Default is 'linear'.
#     :param norm_type: The normalization type, can be 'max', 'sum', or None. Default is None.
#     :param color_norm_type: The color normalization type, can be 'linear' or 'log'. Default is 'linear'.
#     :param color_norm_range: The range for color normalization. Default is None.
#     :param bins: The bins for the histogram, can be an integer, numpy array, list, or a string ('freedman', 'scott', 'knuth', 'blocks'). Default is 'freedman'.
#     :param hist_pcolormesh_kwargs: Additional arguments for pcolormesh. Default is {'cmap':'jet'}.
#     :param contour_levels: The levels for contour plotting. Default is None.
#     :param contour_smooth: The smoothing parameter for contours. Default is None.
#     :param contour_kwargs: Additional arguments for contour plotting. Default is {}.
#     :param fit_line: Whether to fit a line to the data. Default is False.
#     :param fit_line_kwargs: Arguments for plotting the fit line. Default is {'c':'k', 'ls':'--', 'lw':1}.
#     :param mean_std: Whether to plot mean and standard deviation. Default is False.
#     :param mean_std_errorbar_kwargs: Arguments for plotting error bars of mean and standard deviation. Default is {'fmt':'none', 'c':'k', 'capsize':2}.
#     :param separate: Whether to normalize separately along 'x' or 'y' axis. Default is None.
#     :param mean_std_line: Whether to plot mean and standard deviation line. Can be 'x', 'y', or False. Default is False.
#     :param mean_std_line_method: The method for calculating the mean and standard deviation line, can be 'mean', 'max', or 'median'. Default is 'mean'.
#     :param mean_std_line_kwargs: Arguments for plotting the mean and standard deviation line. Default is {'fmt':'o', 'ms':2, 'c':'k', 'capsize':2, 'lw':1, 'ls':'-'}.
    
#     :return quadmesh: The QuadMesh object of the 2D histogram.
#     '''
    
#     if least_samples_per_cell < 1:
#         raise ValueError("least_samples_per_cell must be a positive integer.")
#     if norm_type and norm_type not in ['max', 'sum']:
#         raise ValueError("norm_type must be 'max', 'sum' or None.")
#     if color_norm_type not in ['log', 'linear']:
#         raise ValueError("color_norm_type must be 'log' or 'linear'.")
#     if isinstance(bins, str) and bins not in ['freedman', 'scott', 'knuth', 'blocks']:
#         raise ValueError("bins must be 'freedman', 'scott', 'knuth', 'blocks', an integer, a numpy array, or a list.")
#     if separate and separate not in ['x', 'y']:
#         raise ValueError("separate must be 'x', 'y' or None.")
#     if separate and z is not None:
#         raise ValueError("separate cannot be used when z is provided.")
#     if contour_only and z is not None:
#         raise ValueError("contour_only cannot be True when z is provided.")
#     if norm_type is not None and z is not None:
#         raise ValueError("norm_type cannot be used when z is provided.")
        
#     if hist_pcolormesh_kwargs is None:
#         hist_pcolormesh_kwargs = {}
#     hist_pcolormesh_kwargs.setdefault('cmap', 'jet')
    
#     if contour_kwargs is None:
#         contour_kwargs = {}
        
#     if fit_line_kwargs is None:
#         fit_line_kwargs = {}
#     fit_line_kwargs.setdefault('c', 'k')
#     fit_line_kwargs.setdefault('ls', '--')
#     fit_line_kwargs.setdefault('lw', 1)
        
#     if mean_std_errorbar_kwargs is None:
#         mean_std_errorbar_kwargs = {}
#     mean_std_errorbar_kwargs.setdefault('fmt', 'none')
#     mean_std_errorbar_kwargs.setdefault('c', 'k')
#     mean_std_errorbar_kwargs.setdefault('capsize', 2)
        
#     if mean_std_line_kwargs is None:
#         mean_std_line_kwargs = {}
#     mean_std_line_kwargs.setdefault('fmt', 'o')
#     mean_std_line_kwargs.setdefault('ms', 2)
#     mean_std_line_kwargs.setdefault('c', 'k')
#     mean_std_line_kwargs.setdefault('capsize', 2)
#     mean_std_line_kwargs.setdefault('lw', 1)
#     mean_std_line_kwargs.setdefault('ls', '-')
        

#     if isinstance(x, u.Quantity):
#         x = x.value
#     if isinstance(y, u.Quantity):
#         y = y.value
    
    
#     bins = [bins, bins] if isinstance(bins, str|int) else bins
#     scales = [scales, scales] if isinstance(scales, str) and z is None else [scales, scales, scales] if isinstance(scales, str) else scales
    
#     bins_x = _determine_bins(x, bins[0], scales[0])
#     bins_y = _determine_bins(y, bins[1], scales[1])
    
#     z_hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y])
#     if len(z_hist.shape) < 2:
#         raise ValueError("No enough data to plot 2D histogram.")
        
#     x_mid = (x_edges[1:] + x_edges[:-1]) / 2
#     y_mid = (y_edges[1:] + y_edges[:-1]) / 2
    
#     if z is not None:
#         z_scaled = np.log10(z) if scales[2] == 'log' else z
#         z_mat = np.full(z_hist.shape, np.nan)
#         for j in range(len(x_edges) - 1):
#             for k in range(len(y_edges) - 1):
#                 z_jk = z_scaled[(x > x_edges[j]) & (x <= x_edges[j + 1]) & (y > y_edges[k]) & (y <= y_edges[k + 1])]
#                 z_mat[j, k] = np.nanmean(z_jk) if len(z_jk) >= least_samples_per_cell else np.nan
#         z_mat = 10**z_mat if scales[2] == 'log' else z_mat
        
#         if scales[2] == 'log':
#             color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(z_mat), vmax=np.nanmax(z_mat))
#         elif scales[2] == 'linear':
#             color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(z_mat), vmax=np.nanmax(z_mat))
        
#         if contour_only:
#             raise ValueError("contour_only cannot be True when z is provided.")
#         else:
#             quadmesh = axes.pcolormesh(x_edges, y_edges, z_mat.T, norm=color_norm, **hist_pcolormesh_kwargs)
        
#     else:
#         if norm_type == 'max':
#             if separate == 'x':
#                 z_hist /= np.nanmax(z_hist, axis=1, keepdims=True)
#             elif separate == 'y':
#                 z_hist /= np.nanmax(z_hist, axis=0, keepdims=True)
#             else:
#                 z_hist /= np.nanmax(z_hist)
#         elif norm_type == 'sum':
#             if separate == 'x':
#                 z_hist /= np.nansum(z_hist, axis=1, keepdims=True)
#             elif separate == 'y':
#                 z_hist /= np.nansum(z_hist, axis=0, keepdims=True)
#             else:
#                 z_hist /= np.nansum(z_hist)
        
#         if color_norm_type == 'log':
#             color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(z_hist[z_hist > 0]), vmax=np.nanmax(z_hist))
#             lower_level, upper_level = np.ceil(np.log10(np.nanmin(z_hist[z_hist > 0]))), np.floor(np.log10(np.nanmax(z_hist)))
#             contour_levels_gen = 10**np.arange(lower_level, upper_level + 1)
#         elif color_norm_type == 'linear':
#             color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(z_hist), vmax=np.nanmax(z_hist))
#             lower_level, upper_level = np.nanmin(z_hist), np.nanmax(z_hist)
#             contour_levels_gen = np.linspace(lower_level, upper_level, 5)
        
#         z_masked = np.ma.masked_where(z_hist == 0, z_hist)
#         if not contour_only:
#             quadmesh = axes.pcolormesh(x_edges, y_edges, z_masked.T, norm=color_norm, **hist_pcolormesh_kwargs)
    
#         if not separate:
#             if len(contour_kwargs) > 0 or contour_only:
#                 if 'levels' not in contour_kwargs:
#                     contour_kwargs['levels'] = contour_levels_gen
#                 if contour_smooth:
#                     if isinstance(contour_smooth, int) or int(contour_smooth) == contour_smooth:
#                         z_hist = convolve2d(z_hist, np.ones((contour_smooth, contour_smooth)), mode='same')
#                     elif isinstance(contour_smooth, float):
#                         z_hist = convolve2d(z_hist, np.ones((int(contour_smooth*z_hist.shape[0]), int(contour_smooth*z_hist.shape[1]))), mode='same')
#                 quadcontourset = axes.contour(x_mid, y_mid, z_hist.T, norm=color_norm, **contour_kwargs)
            
#         if fit_line:
#             if scales[0] == 'log':
#                 x_fit = np.log10(x)
#             else:
#                 x_fit = x
#             if scales[1] == 'log':
#                 y_fit = np.log10(y)
#             else:
#                 y_fit = y
                
#             slope, intercept, r_value, p_value, _ = stats.linregress(x_fit, y_fit)
#             x_fitted = np.array([np.nanmin(x_fit), np.nanmax(x_fit)])
#             y_fitted = slope * x_fitted + intercept
            
#             if scales[0] == 'log':
#                 x_fitted = 10**x_fitted
#                 right_label = f'{slope:.2f}' + r'$\log_{10}x$'
#             else:
#                 right_label = f'{slope:.2f}' + r'$x$'
#             if scales[1] == 'log':
#                 y_fitted = 10**y_fitted
#                 left_label = r'$\log_{10}y=$'
#             else:
#                 left_label = r'$y=$'
                
#             # end_label =  (r'$+$' if intercept > 0 else '') + f'{intercept:.2f}\n' + r'$r\approx$' + f'{r_value:.2f}' + r'$\ \ p\geqslant$' + f'{p_value:.2f}'
#             end_label =  (r'$+$' if intercept > 0 else '') + f'{intercept:.2f}\n' + r'$r\approx$' + f'{r_value:.2f}' + r'$\ \ p\leqslant$' + f'{p_value + 0.01 if p_value >= 0.01 else 0.01:.2f}'
#             fit_line_kwargs.setdefault('label', left_label+right_label+end_label)
            
#             _log_or_linear_plot(scales, axes)(x_fitted, y_fitted, **fit_line_kwargs)
            
#         if mean_std:
#             unlog_x_mean, x_cap, x_label = _mean_std_params(x, scales[0])
#             unlog_y_mean, y_cap, y_label = _mean_std_params(y, scales[1])
                
#             axes.errorbar(unlog_x_mean, unlog_y_mean, xerr=x_cap, yerr=y_cap, label=x_label+'\n'+y_label, **mean_std_errorbar_kwargs)
            
#         if mean_std_line:
#             if separate == 'x' or mean_std_line == 'x':
#                 unlog_y_means, y_caps = _mean_std_line_params(x, y, x_edges, scales[1], method=mean_std_line_method)
#                 axes.errorbar(x_mid, unlog_y_means, yerr=y_caps, **mean_std_line_kwargs)
#             elif separate == 'y' or mean_std_line == 'y':
#                 unlog_x_means, x_caps = _mean_std_line_params(y, x, y_edges, scales[0], method=mean_std_line_method)
#                 axes.errorbar(unlog_x_means, y_mid, xerr=x_caps, **mean_std_line_kwargs)
                    
    
        
#     if scales[0] == 'log':
#         axes.set_xscale('log')
#     if scales[1] == 'log':
#         axes.set_yscale('log')
        
#     if contour_only:
#         return quadcontourset
#     else:
#         return quadmesh
    


def plot_hist2d(
    axes: plt.Axes, x: np.ndarray|u.Quantity, y: np.ndarray|u.Quantity, z: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1, 
    scales: list|tuple|str='linear', norm_type: Literal['sum', 'max', None]=None,
    color_norm_type: str='linear', color_norm_range: list|tuple|None=None, bins: int|np.ndarray|list|str='freedman', 
    plot_method: Literal['pcolormesh', 'contour', 'contourf']='pcolormesh',
    plot_kwargs: dict|None=None,
    fit_line: bool=False, fit_line_kwargs: dict|None=None,
    mean_std: bool=False, mean_std_errorbar_kwargs: dict|None=None,
    separate: str|None=None,
    mean_std_line: Literal['x', 'y', None] =None, mean_std_line_method: Literal['mean', 'max', 'median'] ='mean', mean_std_line_kwargs: dict|None=None,
) -> QuadMesh | QuadContourSet:
    
    '''
    Plot a 2D histogram. If z is provided, overplot z as color.

    :param axes: The matplotlib axes to plot on.
    :param x: The x-axis data, can be a numpy array or an astropy Quantity.
    :param y: The y-axis data, can be a numpy array or an astropy Quantity.
    :param z: The z-axis data, can be a numpy array or an astropy Quantity. Default is None.
    :param least_samples_per_cell: The least number of samples per cell to plot, valid only when z is not None. Default is 1.
    :param scales: The scales for the axes, can be 'linear' or 'log'. Default is 'linear'.
    :param norm_type: The normalization type, can be 'max', 'sum', or None. Default is None.
    :param color_norm_type: The color normalization type, can be 'linear' or 'log'. Default is 'linear'.
    :param color_norm_range: The range for color normalization. Default is None.
    :param bins: The bins for the histogram, can be an integer, numpy array, list, or a string ('freedman', 'scott', 'knuth', 'blocks'). Default is 'freedman'.
    :param plot_method: The plotting method ('pcolormesh', 'contour', or 'contourf'). Default is 'pcolormesh'.
    :param plot_kwargs: Additional arguments for the plotting method. Default varies by method.
    :param fit_line: Whether to fit a line to the data. Default is False.
    :param fit_line_kwargs: Arguments for plotting the fit line. Default is {'c':'k', 'ls':'--', 'lw':1}.
    :param mean_std: Whether to plot mean and standard deviation. Default is False.
    :param mean_std_errorbar_kwargs: Arguments for plotting error bars of mean and standard deviation. Default is {'fmt':'none', 'c':'k', 'capsize':2}.
    :param separate: Whether to normalize separately along 'x' or 'y' axis. Default is None.
    :param mean_std_line: Whether to plot mean and standard deviation line. Can be 'x', 'y', or None. Default is None.
    :param mean_std_line_method: The method for calculating the mean and standard deviation line, can be 'mean', 'max', or 'median'. Default is 'mean'.
    :param mean_std_line_kwargs: Arguments for plotting the mean and standard deviation line. Default is {'fmt':'o', 'ms':2, 'c':'k', 'capsize':2, 'lw':1, 'ls':'-'}.
    
    :return: The plot object (QuadMesh or QuadContourSet).
    '''
    
    # Validation
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
    if separate and z is not None:
        raise ValueError("separate cannot be used when z is provided.")
    if norm_type is not None and z is not None:
        raise ValueError("norm_type cannot be used when z is provided.")
    if plot_method not in ['pcolormesh', 'contour', 'contourf']:
        raise ValueError("plot_method must be 'pcolormesh', 'contour', or 'contourf'.")
        
    # Set default kwargs based on plot method
    if plot_kwargs is None:
        plot_kwargs = {}
    
    if plot_method in ['pcolormesh', 'contourf']:
        plot_kwargs.setdefault('cmap', 'jet')
    elif plot_method == 'contour':
        plot_kwargs.setdefault('colors', 'k')  # contour默认使用黑色线条
        # 如果用户指定了cmap，则移除colors
        if 'cmap' in plot_kwargs:
            plot_kwargs.pop('colors', None)
        
    if fit_line_kwargs is None:
        fit_line_kwargs = {}
    fit_line_kwargs.setdefault('c', 'k')
    fit_line_kwargs.setdefault('ls', '--')
    fit_line_kwargs.setdefault('lw', 1)

    if mean_std_errorbar_kwargs is None:
        mean_std_errorbar_kwargs = {}
    mean_std_errorbar_kwargs.setdefault('fmt', 'none')
    mean_std_errorbar_kwargs.setdefault('c', 'k')
    mean_std_errorbar_kwargs.setdefault('capsize', 2)
        
    if mean_std_line_kwargs is None:
        mean_std_line_kwargs = {}
    mean_std_line_kwargs.setdefault('fmt', 'o')
    mean_std_line_kwargs.setdefault('ms', 2)
    mean_std_line_kwargs.setdefault('c', 'k')
    mean_std_line_kwargs.setdefault('capsize', 2)
    mean_std_line_kwargs.setdefault('lw', 1)
    mean_std_line_kwargs.setdefault('ls', '-')

    # Convert quantities to values
    if isinstance(x, u.Quantity):
        x = x.value
    if isinstance(y, u.Quantity):
        y = y.value
    if isinstance(z, u.Quantity):
        z = z.value
    
    # Handle bins and scales
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
        
        data_to_plot = z_mat
        
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
            valid_data = z_hist[z_hist > 0]
            if len(valid_data) == 0:
                raise ValueError("No valid data for log normalization.")
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
            lower_level, upper_level = np.ceil(np.log10(np.nanmin(valid_data))), np.floor(np.log10(np.nanmax(valid_data)))
            contour_levels_gen = 10**np.arange(lower_level, upper_level + 1)
        elif color_norm_type == 'linear':
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(z_hist), vmax=np.nanmax(z_hist))
            lower_level, upper_level = np.nanmin(z_hist), np.nanmax(z_hist)
            contour_levels_gen = np.linspace(lower_level, upper_level, 10)
        
        data_to_plot = z_hist
    
    # Set default contour levels for contour methods
    if plot_method in ['contour', 'contourf'] and 'levels' not in plot_kwargs and z is None:
        plot_kwargs['levels'] = contour_levels_gen
    
    # Create the plot based on method
    if plot_method == 'pcolormesh':
        if z is None:
            # Mask zero values for histogram data
            data_masked = np.ma.masked_where(data_to_plot == 0, data_to_plot)
            plot_obj = axes.pcolormesh(x_edges, y_edges, data_masked.T, norm=color_norm, **plot_kwargs)
        else:
            plot_obj = axes.pcolormesh(x_edges, y_edges, data_to_plot.T, norm=color_norm, **plot_kwargs)
    elif plot_method == 'contourf':
        plot_obj = axes.contourf(x_mid, y_mid, data_to_plot.T, norm=color_norm, **plot_kwargs)
    elif plot_method == 'contour':
        plot_obj = axes.contour(x_mid, y_mid, data_to_plot.T, norm=color_norm, **plot_kwargs)
            
    # Additional features (only when not using separate normalization for contour plots)
    if not separate or plot_method == 'pcolormesh':
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
                
            end_label =  (r'$+$' if intercept > 0 else '') + f'{intercept:.2f}\n' + r'$r\approx$' + f'{r_value:.2f}' + r'$\ \ p\leqslant$' + f'{p_value + 0.01 if p_value >= 0.01 else 0.01:.2f}'
            fit_line_kwargs.setdefault('label', left_label+right_label+end_label)
            
            _log_or_linear_plot(scales, axes)(x_fitted, y_fitted, **fit_line_kwargs)
            
        if mean_std:
            unlog_x_mean, x_cap, x_label = _mean_std_params(x, scales[0])
            unlog_y_mean, y_cap, y_label = _mean_std_params(y, scales[1])
                
            axes.errorbar(unlog_x_mean, unlog_y_mean, xerr=x_cap, yerr=y_cap, label=x_label+'\n'+y_label, **mean_std_errorbar_kwargs)
            
        if mean_std_line:
            if separate == 'x' or mean_std_line == 'x':
                unlog_y_means, y_caps = _mean_std_line_params(x, y, x_edges, scales[1], method=mean_std_line_method)
                axes.errorbar(x_mid, unlog_y_means, yerr=y_caps, **mean_std_line_kwargs)
            elif separate == 'y' or mean_std_line == 'y':
                unlog_x_means, x_caps = _mean_std_line_params(y, x, y_edges, scales[0], method=mean_std_line_method)
                axes.errorbar(unlog_x_means, y_mid, xerr=x_caps, **mean_std_line_kwargs)
        
    # Set axis scales
    if scales[0] == 'log':
        axes.set_xscale('log')
    if scales[1] == 'log':
        axes.set_yscale('log')
        
    return plot_obj

    
def plot_hist3d_slices(
    axes: plt.Axes, x: np.ndarray|u.Quantity, y: np.ndarray|u.Quantity, z: np.ndarray|u.Quantity, 
    w: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1,
    scales: list|tuple|str='linear', norm_type: Literal['sum', 'max', None]=None,
    color_norm_type: str='linear', color_norm_range: list|tuple|None=None, 
    bins: int|np.ndarray|list|str='freedman', 
    slice_direction: Literal['x', 'y', 'z']='z', slice_positions: list|np.ndarray|None=None, slice_count: int=5,
    plot_method: Literal['contour', 'contourf']='contourf',
    slice_kwargs: dict|None=None, alpha: float=0.7,
) -> list:
    
    '''
    Plot 3D histogram as slices in 3D space. If w is provided, overplot w as color.

    :param axes: The matplotlib 3D axes to plot on (must be created with projection='3d').
    :param x: The x-axis data, can be a numpy array or an astropy Quantity.
    :param y: The y-axis data, can be a numpy array or an astropy Quantity.
    :param z: The z-axis data, can be a numpy array or an astropy Quantity.
    :param w: The w-axis data for coloring, can be a numpy array or an astropy Quantity. Default is None.
    :param least_samples_per_cell: The least number of samples per cell to plot, valid only when w is not None. Default is 1.
    :param scales: The scales for the axes, can be 'linear' or 'log'. Default is 'linear'.
    :param norm_type: The normalization type, can be 'max', 'sum', or None. Default is None.
    :param color_norm_type: The color normalization type, can be 'linear' or 'log'. Default is 'linear'.
    :param color_norm_range: The range for color normalization. Default is None.
    :param bins: The bins for the histogram, can be an integer, numpy array, list, or a string ('freedman', 'scott', 'knuth', 'blocks'). Default is 'freedman'.
    :param slice_direction: The direction to slice ('x', 'y', or 'z'). Default is 'z'.
    :param slice_positions: Specific positions to slice at in original data coordinates. If None, will use slice_count to generate evenly spaced slices. Default is None.
    :param slice_count: Number of slices to generate if slice_positions is None. Default is 5.
    :param plot_method: The plotting method ('contour' or 'contourf'). Default is 'contourf'.
    :param slice_kwargs: Additional arguments for the plotting method. Default varies by method.
    :param alpha: Transparency of the slices. Default is 0.7.
    
    :return slice_objects: List of plot objects for each slice.
    '''
    
    # Validation
    if least_samples_per_cell < 1:
        raise ValueError("least_samples_per_cell must be a positive integer.")
    if norm_type and norm_type not in ['max', 'sum']:
        raise ValueError("norm_type must be 'max', 'sum' or None.")
    if color_norm_type not in ['log', 'linear']:
        raise ValueError("color_norm_type must be 'log' or 'linear'.")
    if isinstance(bins, str) and bins not in ['freedman', 'scott', 'knuth', 'blocks']:
        raise ValueError("bins must be 'freedman', 'scott', 'knuth', 'blocks', an integer, a numpy array, or a list.")
    if slice_direction not in ['x', 'y', 'z']:
        raise ValueError("slice_direction must be 'x', 'y', or 'z'.")
    if plot_method not in ['contour', 'contourf']:
        raise ValueError("plot_method must be 'contour' or 'contourf'.")
    if norm_type is not None and w is not None:
        raise ValueError("norm_type cannot be used when w is provided.")
    
    # Check if axes is 3D
    if not hasattr(axes, 'zaxis'):
        raise ValueError("axes must be a 3D axes (created with projection='3d').")
        
    # Set default kwargs based on plot method
    if slice_kwargs is None:
        slice_kwargs = {}
    
    if plot_method == 'contourf':
        slice_kwargs.setdefault('cmap', 'jet')
        slice_kwargs.setdefault('alpha', alpha)
    elif plot_method == 'contour':
        slice_kwargs.setdefault('colors', 'k')  # contour默认使用黑色线条
        slice_kwargs.setdefault('alpha', alpha)
        # 如果用户指定了cmap，则移除colors
        if 'cmap' in slice_kwargs:
            slice_kwargs.pop('colors', None)

    # Convert quantities to values
    if isinstance(x, u.Quantity):
        x = x.value
    if isinstance(y, u.Quantity):
        y = y.value
    if isinstance(z, u.Quantity):
        z = z.value
    if isinstance(w, u.Quantity):
        w = w.value
    
    # Handle bins and scales
    bins = [bins, bins, bins] if isinstance(bins, (str, int)) else bins
    scales = [scales, scales, scales] if isinstance(scales, str) and w is None else [scales, scales, scales, scales] if isinstance(scales, str) else scales
    
    # Apply log transformation to data if needed (replace log axes with log data)
    x_plot = np.log10(x) if scales[0] == 'log' else x
    y_plot = np.log10(y) if scales[1] == 'log' else y
    z_plot = np.log10(z) if scales[2] == 'log' else z
    
    # Check for invalid log values
    if scales[0] == 'log' and np.any(x <= 0):
        raise ValueError("Logarithmic scale for x requires all x values to be positive.")
    if scales[1] == 'log' and np.any(y <= 0):
        raise ValueError("Logarithmic scale for y requires all y values to be positive.")
    if scales[2] == 'log' and np.any(z <= 0):
        raise ValueError("Logarithmic scale for z requires all z values to be positive.")
    
    # Determine bins for each axis (use transformed data for log scales)
    bins_x = _determine_bins(x_plot, bins[0], 'linear')  # Always use linear for transformed data
    bins_y = _determine_bins(y_plot, bins[1], 'linear')
    bins_z = _determine_bins(z_plot, bins[2], 'linear')
    
    # Create 3D histogram using transformed data
    hist_3d, edges = np.histogramdd(np.column_stack([x_plot, y_plot, z_plot]), bins=[bins_x, bins_y, bins_z])
    x_edges, y_edges, z_edges = edges
    
    if hist_3d.size == 0:
        raise ValueError("No enough data to plot 3D histogram.")
        
    # Calculate midpoints
    x_mid = (x_edges[1:] + x_edges[:-1]) / 2
    y_mid = (y_edges[1:] + y_edges[:-1]) / 2
    z_mid = (z_edges[1:] + z_edges[:-1]) / 2
    
    # Handle w data if provided
    if w is not None:
        w_scaled = np.log10(w) if scales[3] == 'log' else w
        w_mat = np.full(hist_3d.shape, np.nan)
        
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                for k in range(len(z_edges) - 1):
                    mask = ((x_plot > x_edges[i]) & (x_plot <= x_edges[i + 1]) & 
                           (y_plot > y_edges[j]) & (y_plot <= y_edges[j + 1]) & 
                           (z_plot > z_edges[k]) & (z_plot <= z_edges[k + 1]))
                    w_ijk = w_scaled[mask]
                    if len(w_ijk) >= least_samples_per_cell:
                        w_mat[i, j, k] = np.nanmean(w_ijk)
                    else:
                        w_mat[i, j, k] = np.nan
        
        w_mat = 10**w_mat if scales[3] == 'log' else w_mat
        data_to_plot = w_mat
        
        # Set color normalization for w data
        if scales[3] == 'log':
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(w_mat), vmax=np.nanmax(w_mat))
        else:
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(w_mat), vmax=np.nanmax(w_mat))
    else:
        data_to_plot = hist_3d.copy()
        
        # Apply normalization
        if norm_type == 'max':
            data_to_plot = data_to_plot / np.nanmax(data_to_plot)
        elif norm_type == 'sum':
            data_to_plot = data_to_plot / np.nansum(data_to_plot)
        
        # Set color normalization
        if color_norm_type == 'log':
            valid_data = data_to_plot[data_to_plot > 0]
            if len(valid_data) == 0:
                raise ValueError("No valid data for log normalization.")
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
            lower_level, upper_level = np.ceil(np.log10(np.nanmin(valid_data))), np.floor(np.log10(np.nanmax(valid_data)))
            contour_levels_gen = 10**np.arange(lower_level, upper_level + 1)
        else:
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(data_to_plot), vmax=np.nanmax(data_to_plot))
            lower_level, upper_level = np.nanmin(data_to_plot), np.nanmax(data_to_plot)
            contour_levels_gen = np.linspace(lower_level, upper_level, 10)
    
    # Set default contour levels
    if 'levels' not in slice_kwargs and w is None:
        slice_kwargs['levels'] = contour_levels_gen
    
    # Determine slice positions
    if slice_positions is not None:
        # User specified positions - transform them to histogram coordinates if needed
        slice_positions = np.array(slice_positions)
        if slice_direction == 'x' and scales[0] == 'log':
            slice_positions_plot = np.log10(slice_positions)
        elif slice_direction == 'y' and scales[1] == 'log':
            slice_positions_plot = np.log10(slice_positions)
        elif slice_direction == 'z' and scales[2] == 'log':
            slice_positions_plot = np.log10(slice_positions)
        else:
            slice_positions_plot = slice_positions
    else:
        # Auto-generate slice positions
        if slice_direction == 'x':
            slice_positions_plot = np.linspace(x_edges[1], x_edges[-2], slice_count)
        elif slice_direction == 'y':
            slice_positions_plot = np.linspace(y_edges[1], y_edges[-2], slice_count)
        else:  # slice_direction == 'z'
            slice_positions_plot = np.linspace(z_edges[1], z_edges[-2], slice_count)
    
    # Get current view to determine depth sorting for drawing order
    elev = axes.elev
    azim = axes.azim
    
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # 计算完整的3D观察方向向量 (单位向量)
    view_dir = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),  # X分量
        np.cos(elev_rad) * np.sin(azim_rad),  # Y分量
        np.sin(elev_rad)                     # Z分量
    ])

    if slice_direction == 'x':
        # X切片：使用完整的X观察分量，按投影深度降序排序
        depth_proj = slice_positions_plot * view_dir[0]
        sorted_indices = np.argsort(-depth_proj)  # 负号实现降序
        
    elif slice_direction == 'y':
        # Y切片：使用完整的Y观察分量
        depth_proj = slice_positions_plot * view_dir[1]
        sorted_indices = np.argsort(-depth_proj)  # 降序
        
    else:  # slice_direction == 'z'
        # Z切片：使用完整的Z观察分量
        depth_proj = slice_positions_plot * view_dir[2]
        sorted_indices = np.argsort(-depth_proj)  # 降序
    
    # Reorder slice positions for proper depth sorting (draw far to near)
    slice_positions_plot = slice_positions_plot[sorted_indices]
    
    slice_objects = []
    
    # Create slices in depth-sorted order (far to near)
    for pos in slice_positions_plot:
        if slice_direction == 'x':
            # Find closest x index
            idx = np.argmin(np.abs(x_mid - pos))
            slice_data = data_to_plot[idx, :, :]  # Shape: (n_y, n_z)
            
            Y_mid_mesh, Z_mid_mesh = np.meshgrid(y_mid, z_mid)
            
            if plot_method == 'contourf':
                obj = axes.contourf(Y_mid_mesh, Z_mid_mesh, slice_data.T, zdir='x', offset=pos, norm=color_norm, **slice_kwargs)
            else:  # contour
                obj = axes.contour(Y_mid_mesh, Z_mid_mesh, slice_data.T, zdir='x', offset=pos, norm=color_norm, **slice_kwargs)
            
        elif slice_direction == 'y':
            # Find closest y index
            idx = np.argmin(np.abs(y_mid - pos))
            slice_data = data_to_plot[:, idx, :]  # Shape: (n_x, n_z)
            
            X_mid_mesh, Z_mid_mesh = np.meshgrid(x_mid, z_mid)
            
            if plot_method == 'contourf':
                obj = axes.contourf(X_mid_mesh, Z_mid_mesh, slice_data.T, zdir='y', offset=pos, norm=color_norm, **slice_kwargs)
            else:  # contour
                obj = axes.contour(X_mid_mesh, Z_mid_mesh, slice_data.T, zdir='y', offset=pos, norm=color_norm, **slice_kwargs)
            
        else:  # slice_direction == 'z'
            # Find closest z index
            idx = np.argmin(np.abs(z_mid - pos))
            slice_data = data_to_plot[:, :, idx]  # Shape: (n_x, n_y)
            
            X_mid_mesh, Y_mid_mesh = np.meshgrid(x_mid, y_mid)
            
            if plot_method == 'contourf':
                obj = axes.contourf(X_mid_mesh, Y_mid_mesh, slice_data.T, zdir='z', offset=pos, norm=color_norm, **slice_kwargs)
            else:  # contour
                obj = axes.contour(X_mid_mesh, Y_mid_mesh, slice_data.T, zdir='z', offset=pos, norm=color_norm, **slice_kwargs)
        
        slice_objects.append(obj)
    
    return slice_objects


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