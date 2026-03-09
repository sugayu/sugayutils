'''Noise simulations.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
from logging import getLogger
import numpy as np
from numpy.random import default_rng

__all__ = ['get_noisesource', 'corrmatrix', 'covmatrix']

logger = getLogger(__name__)


##
class NoiseSourceBase(ABC):
    '''Base class of noise sources.'''

    def __init__(self, seed: int | None = None) -> None:
        self.rng = default_rng(seed)

    def __call__(self, nspec: int = 1) -> np.ndarray:
        return self.generate(nspec)

    @abstractmethod
    def generate(self, nspec: int = 1) -> np.ndarray: ...


class IndependentNoiseSource(NoiseSourceBase):
    '''Give independent noises to data.'''

    def __init__(self, sigma: np.ndarray, *, seed: int | None = None) -> None:
        if sigma.ndim > 1:
            raise ValueError(f'ndim of var must be 1, but now {sigma.ndim}.')
        self.sigma = sigma
        self.size = len(sigma)
        super().__init__(seed)

    def generate(self, nspec: int = 1) -> np.ndarray:
        if nspec == 1:
            return self.rng.standard_normal(self.size) * self.sigma
        return self.rng.standard_normal((nspec, self.size)) * self.sigma


class CorrelatedNoiseSource(NoiseSourceBase):
    '''Give noises to data.'''

    def __init__(self, covar: np.ndarray, *, seed: int | None = None) -> None:
        if covar.ndim > 2:
            raise ValueError(
                'Currently, covar with ndim > 2 is not implemented:'
                f'dim == {covar.ndim}.'
            )
        if covar.shape[0] != covar.shape[1]:
            raise ValueError(f'Covar is not square: {covar.shape}.')

        self.covar = covar
        self.size = len(covar)
        super().__init__(seed)

    def generate(self, nspec: int = 1) -> np.ndarray:
        return self.rng.multivariate_normal(np.zeros(self.size), self.covar, size=nspec)


def get_noisesource(
    covar_or_sigma: np.ndarray | float, *, length: int = 1, seed: int | None = None
) -> NoiseSourceBase:
    '''Constructor of NoiseSources.'''
    if isinstance(covar_or_sigma, float):
        sigma = np.full(length, covar_or_sigma)
        return IndependentNoiseSource(sigma, seed=seed)

    elif covar_or_sigma.ndim == 1:
        sigma = covar_or_sigma
        return IndependentNoiseSource(sigma, seed=seed)

    elif covar_or_sigma.ndim == 2:
        covar = covar_or_sigma
        return CorrelatedNoiseSource(covar, seed=seed)

    raise ValueError(f'Currently, covar ndim=={covar.ndim} is not implemented.')


def corrmatrix(*args, size: int) -> np.ndarray:
    '''Retrun correlation matrix.

    The input arguments indicate correlation value of n-th off-diagonal elements.

    Note:
        This function is copied from roble. It might be better to be integrated
        in the future.
    '''
    corr = np.identity(size)
    for i, arg in enumerate(args, 1):
        r = np.full(size - i, arg)
        corr += np.diag(r, i) + np.diag(r, -i)
    return corr


def covmatrix(sigma: np.ndarray, *args) -> np.ndarray:
    '''Retrun covariance matrix.

    The input arguments indicate correlation value of n-th off-diagonal elements.
    '''
    if sigma.ndim != 1:
        raise ValueError(f'sigma.ndim must be 1, but now {sigma.ndim}')

    corr = corrmatrix(args, size=sigma.size)
    return sigma.reshape(-1, 1) * corr * sigma
