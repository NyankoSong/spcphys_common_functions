"""Module for plotting tools used in space physics data visualization."""

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


def box_stats(data, scale: Literal['linear', 'log'] = 'linear', label=None):
    """Calculate box plot statistics for matplotlib's bxp function.
    
    :param data: Input data array
    :type data: array-like
    :param scale: Scale for data processing, either 'linear' or 'log', defaults to 'linear'
    :type scale: Literal['linear', 'log'], optional
    :param label: Label for the box plot, defaults to None
    :type label: any, optional
    :return: Dictionary containing box plot statistics (whislo, q1, med, q3, whishi, fliers, mean, cilo, cihi)
    :rtype: dict
    """
    if scale == 'log':
        log_nr2 = np.log10(data)
        q1 = np.nanpercentile(log_nr2, 25)
        q3 = np.nanpercentile(log_nr2, 75)
        iqr = q3 - q1
        med_log = np.nanmedian(log_nr2)
        stats = {
            'label': label,
            'whislo': 10**(q1 - 1.5*iqr),
            'q1': 10**q1,
            'med': 10**med_log,
            'q3': 10**q3,
            'whishi': 10**(q3 + 1.5*iqr),
            'fliers': (10**log_nr2[log_nr2 < q1 - 1.5*iqr]).tolist() + (10**log_nr2[log_nr2 > q3 + 1.5*iqr]).tolist(),
            'mean': 10**np.nanmean(log_nr2),
            'cilo': 10**(med_log - 1.57 * iqr / np.sqrt(len(log_nr2))),
            'cihi': 10**(med_log + 1.57 * iqr / np.sqrt(len(log_nr2))),
        }
    else:
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        med = np.nanmedian(data)
        stats = {
            'label': label,
            'whislo': q1 - 1.5*iqr,
            'q1': q1,
            'med': med,
            'q3': q3,
            'whishi': q3 + 1.5*iqr,
            'fliers': data[data < q1 - 1.5*iqr].tolist() + data[data > q3 + 1.5*iqr].tolist(),
            'mean': np.nanmean(data),
            'cilo': med - 1.57 * iqr / np.sqrt(len(data)),
            'cihi': med + 1.57 * iqr / np.sqrt(len(data)),
        }
    
    return stats


def violin_stats(data, scale: Literal['linear', 'log'] = 'linear', n_kde_points: int =1000):
    """Prepare statistics for matplotlib.axes.Axes.violin's vpstats parameter.
    
    :param data: Input data array
    :type data: array-like
    :param scale: Scale for data processing, either 'linear' or 'log'.
                  If 'log', data is log10 transformed (positive, finite values only).
                  Defaults to 'linear'.
    :type scale: Literal['linear', 'log'], optional
    :param n_kde_points: Number of points to evaluate the KDE, defaults to 1000
    :type n_kde_points: int, optional
    :return: Dictionary containing violin plot statistics:
             - 'coords': List of scalars where KDE was evaluated
             - 'vals': List of KDE values at 'coords' (normalized density)
             - 'mean': Mean of (processed) data
             - 'median': Median of (processed) data
             - 'min': Min of (processed) data
             - 'max': Max of (processed) data
             Returns empty 'coords'/'vals' and NaN stats if data is unsuitable.
    :rtype: dict
    """
    data_arr = np.asarray(data)

    if scale == 'log':
        processed_data = data_arr[np.isfinite(data_arr) & (data_arr > 0)]
        if processed_data.size > 0:
            processed_data = np.log10(processed_data)
        else:
            processed_data = np.array([])
    else: # linear scale
        processed_data = data_arr[np.isfinite(data_arr)]

    # Initialize stats
    mean_val, median_val, min_val, max_val = np.nan, np.nan, np.nan, np.nan
    kde_eval_points_list = []
    density_values_list = []

    if processed_data.size > 0:
        mean_val = np.mean(processed_data)
        median_val = np.median(processed_data)
        min_val = np.min(processed_data)
        max_val = np.max(processed_data)

        unique_pts = np.unique(processed_data)
        if len(unique_pts) >= 2:
            try:
                kde = stats.gaussian_kde(processed_data)
                
                kde_eval_min = unique_pts.min()
                kde_eval_max = unique_pts.max()
                
                if np.isclose(kde_eval_min, kde_eval_max) and n_kde_points > 1:
                    # All unique points were essentially identical.
                    # KDE is not meaningful for a shape, treat as single point.
                    # Or, let plt.violin handle this case if it can.
                    # For vpstats, we might provide a single point.
                    # However, plt.violin might expect multiple points for coords/vals.
                    # Let's default to empty if we can't make a sensible KDE shape.
                    pass # kde_eval_points_list and density_values_list remain empty
                else:
                    num_eval_points = max(2, n_kde_points) if not np.isclose(kde_eval_min, kde_eval_max) else 1
                    kde_eval_at = np.linspace(kde_eval_min, kde_eval_max, num_eval_points)
                    
                    if len(kde_eval_at) == 1: # Single unique point after processing
                        # Represent as a very narrow spike if forced, or let it be empty.
                        # For vpstats, providing a single coord/val might be tricky for plotting.
                        # Matplotlib's internal violin creation might handle single-value datasets
                        # by not drawing a body or drawing a line.
                        # Let's provide the single point and a nominal density.
                        # density_at_points = np.array([1.0]) # Arbitrary single point density
                        # For now, let's ensure we have at least two points for linspace if min != max
                        # If min == max, num_eval_points will be 1.
                        # If kde_eval_at has only one point, KDE might not be very informative.
                        # Let's stick to the logic: if num_eval_points is 1, kde_eval_at has 1 point.
                        density_at_points = kde.evaluate(kde_eval_at)

                    else:
                        density_at_points = kde.evaluate(kde_eval_at)

                    max_density = density_at_points.max()
                    if max_density > 0:
                        density_normalized = density_at_points / max_density
                    else:
                        density_normalized = np.zeros_like(density_at_points)
                    
                    kde_eval_points_list = kde_eval_at.tolist()
                    density_values_list = density_normalized.tolist()

            except Exception: # Catch all KDE errors
                # kde_eval_points_list and density_values_list remain empty
                pass
            
    if scale == 'log':
        # Convert stats back to original scale
        if mean_val is not None:
            mean_val = 10**mean_val
        if median_val is not None:
            median_val = 10**median_val
        if min_val is not None:
            min_val = 10**min_val
        if max_val is not None:
            max_val = 10**max_val
        kde_eval_points_list = [10**pt for pt in kde_eval_points_list]
    
    stats_dict = {
        'coords': kde_eval_points_list,
        'vals': density_values_list,
        'mean': mean_val,
        'median': median_val,
        'min': min_val,
        'max': max_val,
    }
    return stats_dict


def _determine_bins(x, bins, scale):
    
    if isinstance(bins, np.ndarray):
        return bins
    
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


def _mid_err_params(x, scale, mid_method, err_method):
    """
    Calculate central value and error bars for data.
    
    :param x: Input data array
    :param scale: 'linear' or 'log' scale
    :param mid_method: 'mean', 'max', or 'median' for central value calculation
    :param err_method: 'std', 'iqr', or 'whisiqr' for error calculation method
        - 'std': standard deviation
        - 'iqr': uses Q1 and Q3 (interquartile range)
        - 'whisiqr': uses Q1-1.5*IQR and Q3+1.5*IQR (whiskers)
    :return: (x_mid, x_cap, x_label) where x_cap is [lower_err, upper_err]
    """
    
    if mid_method == 'mean':
        mid_func = np.nanmean
    elif mid_method == 'max':
        mid_func = np.nanmax
    elif mid_method == 'median':
        mid_func = np.nanmedian
    else:
        raise ValueError("mid_method must be 'mean', 'max', or 'median'")
        
    if scale == 'log':
        # 对数尺度：先取对数，然后计算
        log_x = np.log10(x)
        x_mid_log = mid_func(log_x, axis=0)
        
        if err_method == 'std':
            x_err_log = np.nanstd(log_x, axis=0)
            # 转换回原尺度（非对称）
            x_mid = 10**x_mid_log
            x_cap = np.array([10**x_mid_log - 10**(x_mid_log - x_err_log), 
                             10**(x_mid_log + x_err_log) - 10**x_mid_log])
            
            # 生成标签
            if np.isclose(x_cap[0], x_cap[1], rtol=0.01):
                # 近似对称的情况
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'\pm' + f'{x_err_log:.2f}' + r'}$'
            else:
                # 非对称的情况
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'^{+' + f'{x_err_log:.2f}' + r'}_{-' + f'{x_err_log:.2f}' + r'}}$'
                
        elif err_method == 'iqr':
            # 使用box_stats计算IQR，使用四分位点
            box_dict = box_stats(x, scale='log')
            x_err_lower = x_mid_log - np.log10(box_dict['q1'])  # x_mid - q1
            x_err_upper = np.log10(box_dict['q3']) - x_mid_log  # q3 - x_mid
            
            # 转换回原尺度
            x_mid = 10**x_mid_log
            x_cap = np.array([10**x_mid_log - 10**(x_mid_log - x_err_lower), 
                             10**(x_mid_log + x_err_upper) - 10**x_mid_log])
            
            # 生成标签
            if np.isclose(x_err_lower, x_err_upper, rtol=0.01):
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'\pm' + f'{x_err_lower:.2f}' + r'}$'
            else:
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'^{+' + f'{x_err_upper:.2f}' + r'}_{-' + f'{x_err_lower:.2f}' + r'}}$'
                
        elif err_method == 'whisiqr':
            # 使用box_stats计算IQR，使用whiskers (Q1-1.5*IQR 和 Q3+1.5*IQR)
            box_dict = box_stats(x, scale='log')
            x_err_lower = x_mid_log - np.log10(box_dict['whislo'])  # x_mid - (q1 - 1.5*iqr)
            x_err_upper = np.log10(box_dict['whishi']) - x_mid_log  # (q3 + 1.5*iqr) - x_mid
            
            # 转换回原尺度
            x_mid = 10**x_mid_log
            x_cap = np.array([10**x_mid_log - 10**(x_mid_log - x_err_lower), 
                             10**(x_mid_log + x_err_upper) - 10**x_mid_log])
            
            # 生成标签
            if np.isclose(x_err_lower, x_err_upper, rtol=0.01):
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'\pm' + f'{x_err_lower:.2f}' + r'}$'
            else:
                x_label = r'$x=10^{' + f'{x_mid_log:.2f}' + r'^{+' + f'{x_err_upper:.2f}' + r'}_{-' + f'{x_err_lower:.2f}' + r'}}$'
        else:
            raise ValueError("err_method must be 'std', 'iqr', or 'whisiqr'")
            
    else:  # linear scale
        x_mid = mid_func(x, axis=0)
        
        if err_method == 'std':
            x_err = np.nanstd(x, axis=0)
            # 线性尺度下的对称误差
            x_cap = np.array([x_err, x_err])
            
            # 生成标签
            x_label = r'$x=' + f'{x_mid:.2f}' + r'\pm' + f'{x_err:.2f}' + r'$'
            
        elif err_method == 'iqr':
            # 使用box_stats计算IQR，使用四分位点
            box_dict = box_stats(x, scale='linear')
            x_err_lower = x_mid - box_dict['q1']  # x_mid - q1
            x_err_upper = box_dict['q3'] - x_mid  # q3 - x_mid
            x_cap = np.array([x_err_lower, x_err_upper])
            
            # 生成标签
            if np.isclose(x_err_lower, x_err_upper, rtol=0.01):
                x_label = r'$x=' + f'{x_mid:.2f}' + r'\pm' + f'{x_err_lower:.2f}' + r'$'
            else:
                x_label = r'$x=' + f'{x_mid:.2f}' + r'^{+' + f'{x_err_upper:.2f}' + r'}_{-' + f'{x_err_lower:.2f}' + r'}$'
                
        elif err_method == 'whisiqr':
            # 使用box_stats计算IQR，使用whiskers (Q1-1.5*IQR 和 Q3+1.5*IQR)
            box_dict = box_stats(x, scale='linear')
            x_err_lower = x_mid - box_dict['whislo']  # x_mid - (q1 - 1.5*iqr)
            x_err_upper = box_dict['whishi'] - x_mid  # (q3 + 1.5*iqr) - x_mid
            x_cap = np.array([x_err_lower, x_err_upper])
            
            # 生成标签
            if np.isclose(x_err_lower, x_err_upper, rtol=0.01):
                x_label = r'$x=' + f'{x_mid:.2f}' + r'\pm' + f'{x_err_lower:.2f}' + r'$'
            else:
                x_label = r'$x=' + f'{x_mid:.2f}' + r'^{+' + f'{x_err_upper:.2f}' + r'}_{-' + f'{x_err_lower:.2f}' + r'}$'
        else:
            raise ValueError("err_method must be 'std', 'iqr', or 'whisiqr'")
            
    return x_mid, x_cap, x_label


def _mid_err_line_params(x, y, x_edges, scale, mid_method, err_method):
    y_mids = []
    y_caps = []
    
    for i in range(len(x_edges)-1):
        y_window = y[(x > x_edges[i]) & (x < x_edges[i+1])]
        if len(y_window) > 0:
            y_mid, y_cap, _ = _mid_err_params(y_window, scale, mid_method, err_method)
            y_mids.append(y_mid)
            y_caps.append(y_cap)
        else:
            y_mids.append(np.nan)
            y_caps.append([np.nan, np.nan])

    return np.array(y_mids), np.array(y_caps).T


def histogram(
    data: List[np.ndarray|u.Quantity], 
    bins: int|np.ndarray|list|str ='freedman',
    scales: list|str ='linear',
    color_data: np.ndarray|u.Quantity|None =None,
    color_type: Literal['mean', 'median'] ='mean',
    color_scale: str ='linear',
    least_samples_per_cell: int =1,
):
    
    if isinstance(bins, str) and bins not in ['freedman', 'scott', 'knuth', 'blocks']:
        raise ValueError("bins must be 'freedman', 'scott', 'knuth', 'blocks', an integer, a numpy array, or a list.")
    elif isinstance(bins, list):
        for b in bins:
            if not isinstance(b, (str, int, np.ndarray)):
                raise ValueError("Each bin in the list must be 'freedman', 'scott', 'knuth', 'blocks', an integer, or a numpy array.")
            elif isinstance(b, str) and b not in ['freedman', 'scott', 'knuth', 'blocks']:
                raise ValueError("Each bin in the list must be 'freedman', 'scott', 'knuth', 'blocks', an integer, or a numpy array.")
    if least_samples_per_cell < 1:
        raise ValueError("least_samples_per_cell must be a positive integer.")
    if color_scale not in ['log', 'linear']:
        raise ValueError("color_scale must be 'log' or 'linear'.")
    if least_samples_per_cell < 1:
        raise ValueError("least_samples_per_cell must be a positive integer.")

    for i, x in enumerate(data):
        if isinstance(x, u.Quantity):
            data[i] = x.value
    
    if color_data is not None and isinstance(color_data, u.Quantity):
        color_data = color_data.value
    
    if color_type == 'mean':
        color_func = np.nanmean
    elif color_type == 'median':
        color_func = np.nanmedian

    bins = [bins] * len(data) if isinstance(bins, (str, int, np.ndarray)) else bins
    scales = [scales] * len(data) if isinstance(scales, str) else scales
    
    data_bins = [_determine_bins(x, bins[i], scales[i]) for i, x in enumerate(data)]
    hist, edges = np.histogramdd(np.column_stack(data), bins=data_bins)
    
    if hist.size == 0:
        raise ValueError("No enough data to create histogram.")

    mids = [(edges[i][1:] + edges[i][:-1]) / 2 if scales[i] == 'linear' else 10**((np.log10(edges[i][1:]) + np.log10(edges[i][:-1])) / 2) for i in range(len(edges))]
    
    if color_data is not None:
        color_data_scaled = np.log10(color_data) if color_scale == 'log' else color_data
        color_data_mat = np.full(hist.shape, np.nan)
        it = np.ndindex(hist.shape)
        for idx in it:
            mask = np.ones(len(color_data_scaled), dtype=bool)
            for dim, edge in enumerate(edges):
                if idx[dim] == 0:
                    mask &= (data[dim] > edge[0]) & (data[dim] <= edge[1])
                else:
                    mask &= (data[dim] > edge[idx[dim]]) & (data[dim] <= edge[idx[dim] + 1])
            cell_values = color_data_scaled[mask]
            if len(cell_values) >= least_samples_per_cell:
                color_data_mat[idx] = color_func(cell_values)
            else:
                color_data_mat[idx] = np.nan
        color_data_mat = 10**color_data_mat if color_scale == 'log' else color_data_mat
    else:
        color_data_mat = None
        
    return {
        'hist': hist,
        'edges': edges,
        'mids': mids,
        'bins': bins,
        'scales': scales,
        'color_data_mat': color_data_mat,
        'color_scale': color_scale,
        'least_samples_per_cell': least_samples_per_cell,
    }


def reproject_histogram(
    hist_dict: dict,
    dims: int|List[int],
    method: Literal['sum', 'slice'] ='sum',
    slice_indices: int|List[int]|None =None,
) -> dict:
    
    if hist_dict['color_data_mat'] is not None:
        raise ValueError("Cannot reproject histogram with color data.")
    
    keep_dims = [dims] if isinstance(dims, int) else dims
    new_hist = {}
    if method == 'slice':
        all_axes = list(range(hist_dict['hist'].ndim))
        drop_axes = [ax for ax in all_axes if ax not in keep_dims]
        if slice_indices is None:
            raise ValueError("slice_indices must be provided for method='slice'.")
        slice_list = [slice_indices] if isinstance(slice_indices, int) else slice_indices
        if len(slice_list) != len(drop_axes):
            raise ValueError(f"Need {len(drop_axes)} slice_indices, got {len(slice_list)}.")
        index = []
        for ax in all_axes:
            if ax in keep_dims:
                index.append(slice(None))
            else:
                idx = drop_axes.index(ax)
                index.append(slice_list[idx])
        new_hist['hist'] = hist_dict['hist'][tuple(index)]
    else:
        new_hist['hist'] = np.nansum(hist_dict['hist'], axis=tuple(i for i in range(len(hist_dict['hist'].shape)) if i not in keep_dims))
        
    new_hist['edges'] = [hist_dict['edges'][i] for i in keep_dims]
    new_hist['mids'] = [hist_dict['mids'][i] for i in keep_dims]
    new_hist['bins'] = [hist_dict['bins'][i] for i in keep_dims]
    new_hist['scales'] = [hist_dict['scales'][i] for i in keep_dims]
    new_hist['color_data_mat'] = hist_dict['color_data_mat']
    new_hist['color_scale'] = hist_dict['color_scale']
    new_hist['least_samples_per_cell'] = hist_dict['least_samples_per_cell']
    
    return new_hist


def plot_hist2d(
    axes: plt.Axes, 
    hist_dict: dict|None=None,
    x: np.ndarray|u.Quantity|None=None, y: np.ndarray|u.Quantity|None=None, z: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1, 
    scales: list|tuple|str='linear', norm_type: Literal['sum', 'max', None]=None, color_type: Literal['mean', 'median']='mean',
    color_norm_type: Literal['log', 'linear']='linear', color_norm_range: list|tuple|None=None, bins: int|np.ndarray|list|str='freedman', 
    plot_method: Literal['pcolormesh', 'contour', 'contourf']='pcolormesh',
    plot_kwargs: dict|None=None,
    fit_line: bool=False, fit_line_kwargs: dict|None=None,
    mid_err: bool=False, mid_err_kwargs: dict|None=None,
    separate: str|None=None,
    mid_err_line: Literal['x', 'y', None] =None, mid_method: Literal['mean', 'max', 'median'] ='mean', err_method: Literal['std', 'iqr', 'whisiqr'] ='std', mid_err_line_kwargs: dict|None=None,
    return_histogram: bool=False,
) -> QuadMesh | QuadContourSet | tuple | dict:
    
    '''
    Plot a 2D histogram. If z is provided, overplot z as color.

    :param axes: The matplotlib axes to plot on.
    :param hist_dict: The histogram dictionary containing precomputed histogram data. Default is None.
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
    :param mid_err: Whether to plot mean and standard deviation. Default is False.
    :param mid_err_kwargs: Arguments for plotting error bars of mean and standard deviation. Default is {'fmt':'none', 'c':'k', 'capsize':2}.
    :param separate: Whether to normalize separately along 'x' or 'y' axis. Default is None.
    :param mid_err_line: Whether to plot mean and standard deviation line. Can be 'x', 'y', or None. Default is None.
    :param mid_err_line_kwargs: Arguments for plotting the mean and standard deviation line. Default is {'fmt':'o', 'ms':2, 'c':'k', 'capsize':2, 'lw':1, 'ls':'-'}.
    
    :return: The plot object (QuadMesh or QuadContourSet).
    '''
    
    # Validation
    if norm_type and norm_type not in ['max', 'sum']:
        raise ValueError("norm_type must be 'max', 'sum' or None.")
    if separate:
        if separate not in ['x', 'y']:
            raise ValueError("separate must be 'x', 'y' or None.")
        if z is not None:
            raise ValueError("separate cannot be used when z is provided.")
        if fit_line or mid_err or mid_err_line:
            if x is None or y is None:
                raise ValueError("x and y must be provided when using fit_line, mid_err, or mid_err_line with separate normalization.")
    if norm_type is not None and z is not None:
        raise ValueError("norm_type cannot be used when z is provided.")
    if plot_method not in ['pcolormesh', 'contour', 'contourf']:
        raise ValueError("plot_method must be 'pcolormesh', 'contour', or 'contourf'.")
        
    # Set default kwargs based on plot method
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # if plot_method in ['pcolormesh', 'contourf']:
    #     plot_kwargs.setdefault('cmap', 'jet')
    # elif plot_method == 'contour':
    #     plot_kwargs.setdefault('colors', 'k')  # contour默认使用黑色线条
    #     # 如果用户指定了cmap，则移除colors
    #     if 'cmap' in plot_kwargs:
    #         plot_kwargs.pop('colors', None)
        
    if fit_line_kwargs is None:
        fit_line_kwargs = {}
    fit_line_kwargs.setdefault('c', 'k')
    fit_line_kwargs.setdefault('ls', '--')
    fit_line_kwargs.setdefault('lw', 1)

    if mid_err_kwargs is None:
        mid_err_kwargs = {}
    mid_err_kwargs.setdefault('fmt', 'none')
    mid_err_kwargs.setdefault('c', 'k')
    mid_err_kwargs.setdefault('capsize', 2)
        
    if mid_err_line_kwargs is None:
        mid_err_line_kwargs = {}
    mid_err_line_kwargs.setdefault('fmt', 'o')
    mid_err_line_kwargs.setdefault('ms', 2)
    mid_err_line_kwargs.setdefault('c', 'k')
    mid_err_line_kwargs.setdefault('capsize', 2)
    mid_err_line_kwargs.setdefault('lw', 1)
    mid_err_line_kwargs.setdefault('ls', '-')

    if hist_dict is None:
        # Convert quantities to values
        if isinstance(x, u.Quantity):
            x = x.value
        if isinstance(y, u.Quantity):
            y = y.value
        if isinstance(z, u.Quantity):
            z = z.value
        
        hist_dict = histogram(
            data=[x, y],
            bins=bins,
            scales=scales,
            color_data=z,
            color_type=color_type,
            color_scale=color_norm_type,
            least_samples_per_cell=least_samples_per_cell,
        )
        
    scales = hist_dict['scales']
    color_norm_type = hist_dict['color_scale']
    least_samples_per_cell = hist_dict['least_samples_per_cell']
    
    z_hist = hist_dict['hist']
    x_edges, y_edges = hist_dict['edges']
    x_mid, y_mid = hist_dict['mids']
    z_mat = hist_dict['color_data_mat']
    
    if z_mat is not None:
        if color_norm_type == 'log':
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(z_mat[z_mat > 0]), vmax=np.nanmax(z_mat))
        elif color_norm_type == 'linear':
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
    if not separate:
        if fit_line:
            if scales[0] == 'log':
                x_fit = np.log10(x)
            else:
                x_fit = x
            if scales[1] == 'log':
                y_fit = np.log10(y)
            else:
                y_fit = y
            
            valid_fit_indices = ~np.isnan(x_fit) & ~np.isnan(y_fit)
            x_fit, y_fit = x_fit[valid_fit_indices], y_fit[valid_fit_indices]
                
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
            
        if mid_err:
            unlog_x_mean, x_cap, x_label = _mid_err_params(x, scales[0])
            unlog_y_mean, y_cap, y_label = _mid_err_params(y, scales[1])
                
            axes.errorbar(unlog_x_mean, unlog_y_mean, xerr=x_cap, yerr=y_cap, label=x_label+'\n'+y_label, **mid_err_kwargs)
            
        if mid_err_line:
            if separate == 'x' or mid_err_line == 'x':
                unlog_y_means, y_caps = _mid_err_line_params(x, y, x_edges, scales[1], mid_method=mid_method, err_method=err_method)
                axes.errorbar(x_mid, unlog_y_means, yerr=y_caps, **mid_err_line_kwargs)
            elif separate == 'y' or mid_err_line == 'y':
                unlog_x_means, x_caps = _mid_err_line_params(y, x, y_edges, scales[0], mid_method=mid_method, err_method=err_method)
                axes.errorbar(unlog_x_means, y_mid, xerr=x_caps, **mid_err_line_kwargs)
        
    # Set axis scales
    if scales[0] == 'log':
        axes.set_xscale('log')
    if scales[1] == 'log':
        axes.set_yscale('log')
    
    if return_histogram:
        return plot_obj, hist_dict
    else:
        return plot_obj

    
def plot_hist3d_slices(
    axes: plt.Axes, 
    hist_dict: dict|None =None,
    x: np.ndarray|u.Quantity|None =None, y: np.ndarray|u.Quantity|None =None, z: np.ndarray|u.Quantity|None =None, 
    w: np.ndarray|u.Quantity|None=None, least_samples_per_cell: int=1,
    scales: list|tuple|str='linear', norm_type: Literal['sum', 'max', None]=None, color_type: Literal['mean', 'median']='mean',
    color_norm_type: str='linear', color_norm_range: list|tuple|None=None, 
    bins: int|np.ndarray|list|str='freedman', 
    slice_direction: Literal['x', 'y', 'z']='z', slice_positions: list|np.ndarray|None=None, slice_count: int=5,
    plot_method: Literal['contour', 'contourf']='contourf',
    slice_kwargs: dict|None=None, 
    return_histogram: bool=False,
) -> tuple | list | dict:
    
    '''
    Plot 3D histogram as slices in 3D space. If w is provided, overplot w as color.

    :param axes: The matplotlib 3D axes to plot on (must be created with projection='3d').
    :param hist_dict: The histogram dictionary containing precomputed histogram data. Default is None.
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
    
    # if plot_method == 'contourf':
    #     slice_kwargs.setdefault('cmap', 'jet')
    # elif plot_method == 'contour':
    #     slice_kwargs.setdefault('cmap', 'jet')

    if hist_dict is None:
        # Convert quantities to values
        if isinstance(x, u.Quantity):
            x = x.value
        if isinstance(y, u.Quantity):
            y = y.value
        if isinstance(z, u.Quantity):
            z = z.value
        if isinstance(w, u.Quantity):
            w = w.value
        
        hist_dict = histogram(
            data=[x, y, z],
            bins=bins,
            scales=scales,
            color_data=w,
            color_type=color_type,
            color_scale=color_norm_type,
            least_samples_per_cell=least_samples_per_cell,
        )
    
    scales = hist_dict['scales']
    color_norm_type = hist_dict['color_scale']
    least_samples_per_cell = hist_dict['least_samples_per_cell']
    
    hist_3d = hist_dict['hist']
    x_edges, y_edges, z_edges = hist_dict['edges']
    x_mid, y_mid, z_mid = hist_dict['mids']
    w_mat = hist_dict['color_data_mat']
    
    if scales[0] == 'log':
        x_edges = np.log10(x_edges)
        x_mid = np.log10(x_mid)

    if scales[1] == 'log':
        y_edges = np.log10(y_edges)
        y_mid = np.log10(y_mid)

    if scales[2] == 'log':
        z_edges = np.log10(z_edges)
        z_mid = np.log10(z_mid)

    # Handle w data if provided
    if w_mat is not None:
        if color_norm_type == 'log':
            color_norm = LogNorm(*color_norm_range) if color_norm_range else LogNorm(vmin=np.nanmin(w_mat[w_mat > 0]), vmax=np.nanmax(w_mat))
        else:
            color_norm = Normalize(*color_norm_range) if color_norm_range else Normalize(vmin=np.nanmin(w_mat), vmax=np.nanmax(w_mat))
        
        data_to_plot = w_mat
    else:
        data_to_plot = hist_3d
        
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
    
    ##################################################################################
    # THESE LOGICS ARE NOT WORKING                                                   #
    ##################################################################################
    # Get current view to determine depth sorting for drawing order
    # elev = axes.elev
    # azim = axes.azim
    
    # elev_rad = np.radians(elev)
    # azim_rad = np.radians(azim)

    # # 计算完整的3D观察方向向量 (单位向量)
    # view_dir = np.array([
    #     np.cos(elev_rad) * np.cos(azim_rad),  # X分量
    #     np.cos(elev_rad) * np.sin(azim_rad),  # Y分量
    #     np.sin(elev_rad)                     # Z分量
    # ])

    # if slice_direction == 'x':
    #     # X切片：使用完整的X观察分量，按投影深度降序排序
    #     depth_proj = slice_positions_plot * view_dir[0]
    #     sorted_indices = np.argsort(-depth_proj)  # 负号实现降序
        
    # elif slice_direction == 'y':
    #     # Y切片：使用完整的Y观察分量
    #     depth_proj = slice_positions_plot * view_dir[1]
    #     sorted_indices = np.argsort(-depth_proj)  # 降序
        
    # else:  # slice_direction == 'z'
    #     # Z切片：使用完整的Z观察分量
    #     depth_proj = slice_positions_plot * view_dir[2]
    #     sorted_indices = np.argsort(-depth_proj)  # 降序
    
    # # Reorder slice positions for proper depth sorting (draw far to near)
    # slice_positions_plot = slice_positions_plot[sorted_indices]
    ##################################################################################
    # THESE LOGICS ARE NOT WORKING                                                   #
    ##################################################################################
    
    slice_objects = []
    
    # Create slices in depth-sorted order (far to near)
    for pos in slice_positions_plot:
        if slice_direction == 'x':
            # Find closest x index
            idx = np.argmin(np.abs(x_mid - pos))
            slice_data = data_to_plot[idx, :, :]  # Shape: (n_y, n_z)
            
            Y_mid_mesh, Z_mid_mesh = np.meshgrid(y_mid, z_mid)
            
            if plot_method == 'contourf':
                obj = axes.contourf(slice_data.T, Y_mid_mesh, Z_mid_mesh, zdir='x', offset=pos, norm=color_norm, **slice_kwargs)
            else:  # contour
                obj = axes.contour(slice_data.T, Y_mid_mesh, Z_mid_mesh, zdir='x', offset=pos, norm=color_norm, **slice_kwargs)
            
        elif slice_direction == 'y':
            # Find closest y index
            idx = np.argmin(np.abs(y_mid - pos))
            slice_data = data_to_plot[:, idx, :]  # Shape: (n_x, n_z)
            
            X_mid_mesh, Z_mid_mesh = np.meshgrid(x_mid, z_mid)
            
            if plot_method == 'contourf':
                obj = axes.contourf(X_mid_mesh, slice_data.T, Z_mid_mesh, zdir='y', offset=pos, norm=color_norm, **slice_kwargs)
            else:  # contour
                obj = axes.contour(X_mid_mesh, slice_data.T, Z_mid_mesh, zdir='y', offset=pos, norm=color_norm, **slice_kwargs)
            
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
    
    if return_histogram:
        return slice_objects, hist_dict
    else:
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