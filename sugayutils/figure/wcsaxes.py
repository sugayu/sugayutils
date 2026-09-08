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

    def remove_xticklabel(self):
        '''Erase xlabel'''
        return self.coords[0].set_ticklabel_visible(False)

    def remove_yticklabel(self):
        '''Erase ylabel'''
        return self.coords[1].set_ticklabel_visible(False)

    def remove_xyticklabels(self):
        '''Erase both x and y labels'''
        self.remove_xticklabel()
        self.remove_yticklabel()

    def set_xylabels(self, xlabel, ylabel, **kwargs):
        '''Set xlabel and ylable at the same time.'''
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)


class WCSAxesProjection:

    def __init__(self, wcs) -> None:
        self.wcs = wcs

    def _as_mpl_axes(self):
        return (WCSAxes, {'wcs': self.wcs})
