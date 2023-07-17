'''Wrapper of Kernel Distribution Estimates.
'''
from typing import Sequence
import numpy as np
from scipy.stats import gaussian_kde


##
class KDE(gaussian_kde):
    '''Wrapper of Scipy gaussian_kde.

    Provides:
    - reflected KDE.

    Reference:
    https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/core/plots/bounded_1d_kde.py
    '''

    def __init__(self, dataset, *args, **kwargs) -> None:
        _dataset = dataset
        if not np.isfinite(dataset).all():
            idx_infinite = ~np.isfinite(dataset)
            _dataset = dataset[~idx_infinite]
        if _dataset.size == 0:
            raise ValueError('No valid data included in input dataset.')
        super().__init__(_dataset, *args, **kwargs)

    def evaluate(self, points: np.ndarray, lim: Sequence = [None, None]) -> np.ndarray:
        '''Reflected KDE evaluation.'''
        if self.dataset.squeeze().ndim > 1:
            raise TypeError('Reflected KDE can be only used for 1d input.')
        normalization = np.ones_like(points, dtype=float)
        binsize = points[1] - points[0]
        delta = 1e-3 * binsize  # very small amount for calcuration

        pdf = super().evaluate(points)
        if lim[0] is not None:
            pdf += super().evaluate(2 * lim[0] - points - binsize)
            normalization[points < lim[0] - delta] = 0.0
        if lim[1] is not None:
            pdf += super().evaluate(2 * lim[1] - points + binsize)
            normalization[points > lim[1] + delta] = 0.0

        return pdf * normalization

    __call__ = evaluate
