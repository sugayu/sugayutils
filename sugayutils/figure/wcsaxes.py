'''WCS axes.
'''

from __future__ import annotations
from logging import getLogger
from astropy.wcs import WCS
from astropy.visualization import wcsaxes
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

__all__ = ['WCS', 'WCSAxes', 'WCSAxesProjection']

logger = getLogger(__name__)


##
class WCSAxes(wcsaxes.WCSAxes):
    '''Wrapper of astropy WCSAxes.'''

    def set_xywidths(self, center: SkyCoord, xw: u.Quantity, yw: u.Quantity) -> None:
        xlim, ylim = self.wcs.world_to_pixel(
            SkyCoord(
                [center.ra + xw / 2.0, center.ra - xw / 2.0],
                [center.dec - yw / 2.0, center.dec + yw / 2.0],
            )
        )
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_xwidth(self, center: SkyCoord, xw: u.Quantity) -> None:
        xlim = self.wcs.world_to_pixel_values(
            [center.ra - xw / 2.0, center.ra + xw / 2.0], [center.dec, center.dec]
        )
        self.set_xlim(xlim)

    def set_ywidth(self, center: SkyCoord, yw: u.Quantity) -> None:
        ylim = self.wcs.world_to_pixel_values(
            [center.ra, center.ra], [center.dec - yw / 2.0, center.ra + yw / 2.0]
        )
        self.set_ylim(ylim)


class WCSAxesProjection:

    def __init__(self, wcs) -> None:
        self.wcs = wcs

    def _as_mpl_axes(self):
        logger.info('in _as_mpl_axes')
        return (WCSAxes, {'wcs': self.wcs})
