'''Module for plotting tools.'''

from typing import List
import numpy as np
from astropy import units as u
from astropy import stats as astats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import convolve2d
import scipy.stats as stats

from . import config
from .utils import check_parameters


def _determine_bins(x, bins, scale):
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
    log_x_mean = np.mean(log_x, axis=0)
    unlog_x_mean = 10**log_x_mean
    log_x_std = np.std(log_x, axis=0)
    # log_x_err = np.abs(log_x_std/log_x_mean/2.303) # 废弃，此方法并不是这么用的，而是用于没有原始数据的观测数据，例如只提供了均值和标准差
    # x_cap = [np.abs(x_mid - 10**(log_x_mean - log_x_err)),  np.abs(x_mid - 10**(log_x_mean + log_x_err))]
    x_cap = [np.abs(unlog_x_mean - 10**(log_x_mean - log_x_std)),  np.abs(unlog_x_mean - 10**(log_x_mean + log_x_std))]
    return unlog_x_mean, log_x_std, x_cap


def _mean_std_params(x, scale):
    if scale == 'log':
        unlog_x_mean, log_x_std, x_cap = _logarithmic_error(x)
        x_label = r'$x=10\^($' + f'{np.log10(unlog_x_mean):.2f}' + r'$\pm$' + f'{log_x_std:.2f}' + r'$)$'
    else:
        x_mean, x_err = np.mean(x, axis=0), np.std(x, axis=0)
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
            y_mean, y_err = np.mean(y_window), np.std(y_window)
            unlog_y_means.append(y_mean)
            y_caps.append([y_err, y_err])

    return np.array(unlog_y_means), np.array(y_caps).T


@check_parameters
def plot_hist2d(axes: plt.Axes, x: np.ndarray|u.Quantity, y: np.ndarray|u.Quantity, scales: list|tuple|str='linear', norm_type: str|None=None,
                color_norm_type: str='linear', color_norm_range: list|tuple|None=None, bins: int|np.ndarray|list|str='freedman', hist_pcolormesh_args: dict|None=None,
                contour_levels: list|np.ndarray|None=None, contour_smooth: int|float|None=None, contour_args: dict|None=None,
                fit_line: bool=False, fit_line_plot_args: dict|None=None,
                mean_std: bool=False, mean_std_errorbar_args: dict|None=None,
                separate: str|None=None,
                mean_std_line: bool=False, mean_std_line_args: dict|None=None,
                ) -> plt.Axes:
    
    '''
    Plot a 2D histogram.

    :param axes: The matplotlib axes to plot on.
    :param x: The x-axis data, can be a numpy array or an astropy Quantity.
    :param y: The y-axis data, can be a numpy array or an astropy Quantity.
    :param scales: The scales for the axes, can be 'linear' or 'log'. Default is 'linear'.
    :param norm_type: The normalization type, can be 'max', 'sum', or None. Default is None.
    :param color_norm_type: The color normalization type, can be 'linear' or 'log'. Default is 'linear'.
    :param color_norm_range: The range for color normalization. Default is None.
    :param bins: The bins for the histogram, can be an integer, numpy array, list, or a string ('freedman', 'scott', 'knuth', 'blocks'). Default is 'freedman'.
    :param hist_pcolormesh_args: Additional arguments for pcolormesh. Default is {'cmap':'jet'}.
    :param contour_levels: The levels for contour plotting. Default is None.
    :param contour_smooth: The smoothing parameter for contours. Default is None.
    :param contour_args: Additional arguments for contour plotting. Default is {}.
    :param fit_line: Whether to fit a line to the data. Default is False.
    :param fit_line_plot_args: Arguments for plotting the fit line. Default is {'color':'k', 'linestyle':'--', 'linewidth':1}.
    :param mean_std: Whether to plot mean and standard deviation. Default is False.
    :param mean_std_errorbar_args: Arguments for plotting error bars of mean and standard deviation. Default is {'fmt':'none', 'color':'k', 'capsize':2}.
    :param separate: Whether to normalize separately along 'x' or 'y' axis. Default is None.
    :param mean_std_line: Whether to plot a line for mean and standard deviation. Default is False.
    :param mean_std_line_args: Arguments for plotting the mean and standard deviation line. Default is {'fmt':'o', 'ms':2, 'color':'k', 'capsize':2, 'linewidth':1, 'linestyle':'-'}.
    
    :return: The matplotlib axes with the plot.
    '''
    
    if config.ENABLE_VALUE_CHECKING:
        if norm_type and norm_type not in ['max', 'sum']:
            raise ValueError("norm_type must be 'max', 'sum' or None.")
        if color_norm_type not in ['log', 'linear']:
            raise ValueError("color_norm_type must be 'log' or 'linear'.")
        if isinstance(bins, str) and bins not in ['freedman', 'scott', 'knuth', 'blocks']:
            raise ValueError("bins must be 'freedman', 'scott', 'knuth', 'blocks', an integer, a numpy array, or a list.")
        if separate and separate not in ['x', 'y']:
            raise ValueError("separate must be 'x', 'y' or None.")
        
    if hist_pcolormesh_args is None:
        hist_pcolormesh_args = {'cmap':'jet'}
    if contour_args is None:
        contour_args = {}
    if fit_line_plot_args is None:
        fit_line_plot_args = {'color':'k', 'linestyle':'--', 'linewidth':1}
    if mean_std_errorbar_args is None:
        mean_std_errorbar_args = {'fmt':'none', 'color':'k', 'capsize':2}
    if mean_std_line_args is None:
        mean_std_line_args = {'fmt':'o', 'ms':2, 'color':'k', 'capsize':2, 'linewidth':1, 'linestyle':'-'}
        

    if isinstance(x, u.Quantity):
        x = x.value
    if isinstance(y, u.Quantity):
        y = y.value
    
    
    bins = [bins, bins] if len(bins) == 1 else bins
    scales = [scales, scales] if len(scales) == 1 else scales
    
    bins_x = _determine_bins(x, bins, scales[0])
    bins_y = _determine_bins(y, bins, scales[1])
    
    z_hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y])
    if len(z_hist.shape) < 2:
        raise ValueError("No enough data to plot 2D histogram.")
    
    if norm_type == 'max':
        if separate == 'x':
            z_hist /= z_hist.max(axis=1)
        elif separate == 'y':
            z_hist /= z_hist.max(axis=0)
        else:
            z_hist /= z_hist.max()
    elif norm_type == 'sum':
        if separate == 'x':
            z_hist /= z_hist.sum(axis=1)
        elif separate == 'y':
            z_hist /= z_hist.sum(axis=0)
        else:
            z_hist /= z_hist.sum()
        
    if color_norm_type == 'log':
        color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=z_hist.min(), vmax=z_hist.max())
    elif color_norm_type == 'linear':
        color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=z_hist.min(), vmax=z_hist.max())
        
    x_mid = (x_edges[1:] + x_edges[:-1]) / 2
    y_mid = (y_edges[1:] + y_edges[:-1]) / 2
    
    z_masked = np.ma.masked_where(z_hist == 0, z_hist)
    axes.pcolormesh(x_edges, y_edges, z_masked.T, norm=color_norm, **hist_pcolormesh_args)
    
    if scales[0] == 'log':
        axes.set_xscale('log')
    if scales[1] == 'log':
        axes.set_yscale('log')
        
    if not separate:
        if contour_levels:
            if contour_smooth:
                if isinstance(contour_smooth, int) or int(contour_smooth) == contour_smooth:
                    z_hist = convolve2d(z_hist, np.ones((contour_smooth, contour_smooth)), mode='same')
                elif isinstance(contour_smooth, float):
                    z_hist = convolve2d(z_hist, np.ones((int(contour_smooth*z_hist.shape[0]), int(contour_smooth*z_hist.shape[1]))), mode='same')
            axes.contour(x_mid, y_mid, z_hist.T, levels=contour_levels, **contour_args)
        
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
            x_fitted = np.array([x_fit.min(), x_fit.max()])
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
            
            _log_or_linear_plot(scales, axes)(x_fitted, y_fitted, label=left_label+right_label+end_label, **fit_line_plot_args)
        
        if mean_std:
            unlog_x_mean, x_cap, x_label = _mean_std_params(x, scales[0])
            unlog_y_mean, y_cap, y_label = _mean_std_params(y, scales[1])
                
            axes.errorbar(unlog_x_mean, unlog_y_mean, xerr=x_cap, yerr=y_cap, label=x_label+'\n'+y_label, **mean_std_errorbar_args)
        
    else:
        if mean_std_line:
            if separate == 'x':
                unlog_y_means, y_caps = _mean_std_line_params(x, y, x_edges, scales[1])
                axes.errorbar(x_mid, unlog_y_means, yerr=y_caps, **mean_std_line_args)
            elif separate == 'y':
                unlog_x_means, x_caps = _mean_std_line_params(y, x, y_edges, scales[0])
                axes.errorbar(unlog_x_means, y_mid, xerr=x_caps, **mean_std_line_args)
        
    return axes

