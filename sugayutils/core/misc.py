''' miscellaneous
'''
import numpy as np


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
    '''Compute and Print statistical values of array.
    '''
    dict_output = {
        'min': np.nanmin(array),
        'max': np.nanmax(array),
        'std': np.nanstd(array),
        'sum': np.nansum(array),
        'Npix': array.size,
        'N_nan_or_inf': np.count_nonzero(np.logical_not(np.isfinite(array))),
    }
    return dict_output
