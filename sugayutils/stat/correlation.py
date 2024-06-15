'''Correlation functions
'''

import numpy as np
from logging import getLogger

logger = getLogger(__name__)

__all__ = ['autocorrelation']


##
def autocorrelation(data: np.ndarray) -> np.ndarray:
    '''Give an autocorrelation function of the input data.

    DEPRECATED:
    '''
    logger.error('This function is deprecated.')
    raise DeprecationWarning('This function is deprecated.')

    mean, var = np.nanmean(data), np.nanvar(data)
    ndata = data - mean
    triu = np.triu(np.tile(data, (len(data), 1)))
    triu[triu == 0.0] = np.nan
    tril = np.tril(np.tile(data, (len(data), 1)))
    tril[tril == 0.0] = np.nan

    meanu = np.nanmean(triu, axis=1).reshape(-1, 1)
    triu_mean = triu - meanu
    meanl = np.nanmean(tril, axis=1).reshape(-1, 1)
    tril_mean = (tril - meanl)[::-1]
    for i, a in enumerate(tril_mean):
        a[:] = np.roll(a, i)

    cov = np.nansum(triu_mean * tril_mean, axis=1) / (np.arange(len(data))[::-1] + 1)
    stdu = np.sqrt(np.nansum(triu_mean**2, axis=1) / (np.arange(len(data))[::-1] + 1))
    stdl = np.sqrt(np.nansum(tril_mean**2, axis=1) / (np.arange(len(data))[::-1] + 1))

    # var = np.nanvar(np.triu(np.tile(ndata, (len(ndata), 1))), axis=1)
    # stdu, stdl = np.nanstd(triu, axis=1), np.nanstd(tril, axis=1)[::-1]
    # acorr = np.correlate(ndata, ndata, 'full')[len(ndata) - 1 :]
    # (np.arange(len(ndata))[::-1] + 1)
    # return acorr / var / len(ndata)
    # return acorr / stdu / stdl / (np.arange(len(ndata))[::-1] + 1)
    return cov / stdu / stdl
