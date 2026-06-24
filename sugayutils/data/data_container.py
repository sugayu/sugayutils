'''Data Container.
'''

from __future__ import annotations
import itertools
from typing import NamedTuple
from logging import getLogger
import numpy as np
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SpectralCoord

__all__ = ['GridData']

logger = getLogger(__name__)


##
class GridData(CCDData):
    '''General data container.'''

    def __init__(self, *args, **kwd) -> None:
        super().__init__(*args, **kwd)
        self.sky = None
        self.spec = None

        self.pix = PixelCoordUtils(self.data.shape)
        if self.wcs.has_celestial:
            self.sky = SkyCoordUtils(self.data.shape, self.wcs)
        if self.wcs.has_spectral:
            self.spec = SpecCoordUtils(self.data.shape, self.wcs)


class SkyCoordUtils:
    '''Utilities returning sky coordinates.'''

    def __init__(self, shape: tuple[int, ...], wcs: WCS) -> None:
        assert wcs.naxis == len(shape)
        dim = wcs.naxis
        ira, idec = dim - wcs.wcs.lng - 1, dim - wcs.wcs.lat - 1
        self._shape = (shape[ira], shape[idec])
        self._pix = PixelCoordUtils(self._shape)
        self._wcs = wcs.celestial

    @property
    def center(self) -> SkyCoord:
        return self._wcs.pixel_to_world(*self._pix.center)

    @property
    def corners(self) -> SkyCoord:
        return self._wcs.pixel_to_world(
            *tuple(itertools.zip_longest(*self._pix.corners))
        )


class SpecCoordUtils:
    '''Utilities returning spec coordinates.'''

    def __init__(self, shape: tuple[int, ...], wcs: WCS) -> None:
        assert wcs.naxis == len(shape)
        dim = wcs.naxis
        ispec = dim - wcs.wcs.spec
        self._shape = (shape[ispec],)
        self._pix = PixelCoordUtils(self._shape)
        self._wcs = wcs.spectral

    @property
    def center(self) -> SpectralCoord:
        return self._wcs.pixel_to_world(*self._pix.center)

    @property
    def corners(self) -> SkyCoord:
        return self._wcs.pixel_to_world(
            *tuple(itertools.zip_longest(*self._pix.corners))
        )


class PixelCoordUtils:
    '''Utilities returning pixel coordinates.'''

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def center(self) -> tuple[float, ...]:
        return tuple([s / 2.0 for s in self._shape])

    @property
    def corners(self) -> tuple[tuple[int, ...], ...]:
        '''Return corners of the data.

        Returns:
            tuple[tuple[int$ ...], ...]: [description]

        Example:
            >>> assert pix._shape == (2,3)
            >>> pix.corners
            ((0, 0), (2, 0), (0, 3), (2, 3))
        '''
        seed = tuple(itertools.product((0,), self._shape))
        return tuple([tuple(reversed(i)) for i in itertools.product(*seed[::-1])])
