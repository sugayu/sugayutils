''' miscellaneous
'''

import numpy as np
from typing import Any


##
__all__ = ['scale', 'get_nearest', 'get_argnearest', 'stat']


def scale(ndarray, scale):
    '''Scaling array values.
    Input:
    ndarray -- target array.
    scale -- list [min, max], including minimum and maximum of scaling.
    '''
    vmin = np.nanmin(ndarray)
    vmax = np.nanmax(ndarray)
    return (ndarray - vmin) / (vmax - vmin) * (scale[1] - scale[0]) + scale[0]


def get_nearest(value, value_list, num=1):
    '''Get values near the input value from value_list.
    Input:
    num -- number of outputs
    '''
    diff = np.array(value_list) - value
    idx = np.argsort(np.abs(diff))
    return (diff[idx])[:num] + value


def get_argnearest(value, value_list, num=1):
    '''Get index of a value near the input value from value_list.
    Input:
    num -- number of outputs
    '''
    diff = np.array(value_list) - value
    idx = np.argsort(np.abs(diff))
    return idx[:num]


def stat(array: np.ndarray) -> dict:
    '''Compute and Print statistical values of array.'''
    dict_output = {
        'min': np.nanmin(array),
        'max': np.nanmax(array),
        'mean': np.nanmean(array),
        'med': np.nanmedian(array),
        'std': np.nanstd(array),
        'sum': np.nansum(array),
        'Npix': array.size,
        'N_nan_or_inf': np.count_nonzero(np.logical_not(np.isfinite(array))),
    }
    return dict_output


def listup_instancevar(instance: Any):
    '''List up instance variables in classes.'''
    return [
        attr
        for attr in dir(instance)
        if not callable(getattr(instance, attr)) and not attr.startswith("__")
    ]


def to_logerr(error: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Make error in logarithm.'''
    lerr = np.log10(value) - np.log10(value - error)
    herr = np.log10(value + error) - np.log10(value)
    return lerr, herr


def upsampling1d(grid: np.ndarray, rate_upsampling: int) -> np.ndarray:
    '''Make grids up-sampling.

    Args:
        grid (np.ndarray):
        rate_upsampling (int):
    '''
    if rate_upsampling == 1:
        return grid
    nbins_to = len(grid) * rate_upsampling
    normgrid = (grid - grid.min()) / (grid.max() - grid.min()) * len(grid)
    res = np.linspace(normgrid[0] - 0.5, normgrid[-1] + 0.5, (nbins_to) * 2 + 1)[1:-1:2]
    return res * (grid.max() - grid.min()) / len(grid) + grid.min()


def upsampling1d_irregular(grid: np.ndarray, rate_upsampling: int) -> np.ndarray:
    '''Make irregular grids up-sampring.'''
    mid_array = (grid[1:] + grid[:-1]) / 2.0
    edge0 = grid[0] - (mid_array[0] - grid[0])
    edge1 = grid[-1] + (grid[-1] - mid_array[-1])
    edges = np.concatenate(((edge0,), mid_array, (edge1,)))

    wbins = (edges[1:] - edges[:-1]) / rate_upsampling
    margins_left = wbins / 2.0
    newgrid = np.array(
        [edges[:-1] + margins_left + wbins * i for i in range(rate_upsampling)]
    ).ravel()
    newgrid = np.sort(newgrid)
    return newgrid


def finitediff(array: np.ndarray) -> np.ndarray:
    '''Get separations of data points.

    The edges of the data array is treated in the same way as the data point one step inside.
    '''
    mid_array = (array[1:] + array[:-1]) / 2.0
    d_array = mid_array[1:] - mid_array[:-1]
    edge0 = (mid_array[0] - array[0]) * 2
    edge1 = (array[-1] - mid_array[-1]) * 2
    return np.concatenate(((edge0,), d_array, (edge1,)))
