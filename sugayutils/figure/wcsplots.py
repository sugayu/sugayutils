'''Plot utilities using WCS.
'''

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import NDData, Cutout2D
from astropy.coordinates import SkyCoord
from reproject.mosaicking import find_optimal_celestial_wcs


__all__ = ['get_wcsproj_north_is_up']


##
def get_wcsproj_north_is_up(
    skyposition: SkyCoord,
    size: u.Quantity,
    image: NDData | Cutout2D | np.ndarray,
    wcs: WCS | None = None,
) -> WCS:
    '''Get WCS towards north.'''
    if isinstance(image, (NDData, Cutout2D)):
        data, wcs = image.data, image.wcs
    elif isinstance(image, np.ndarray):
        data = image
        if wcs is None:
            raise ValueError('When the input data is numpy.ndarray, wcs must be given.')
        wcs = wcs
    else:
        raise TypeError(f'Image type {type(image)} cannnot be recognized as input.')
    _wcs, _shape = find_optimal_celestial_wcs((data, wcs))
    return Cutout2D(np.empty(_shape), skyposition, size, wcs=_wcs).wcs
