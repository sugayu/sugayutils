'''Plot six images of stars and galaxies.
'''

from pathlib import Path
from typing import Callable
import numpy as np
import astropy.units as u
from astropy.nddata import NDDataArray, Cutout2D
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord
from sugayutils.figure.matplotlib import DS9LogNorm
from sugayutils.figure import makefig, Figure, get_wcsproj_north_is_up


__all__ = ['fig_image_six_stamps', 'DrawStamp']


##
def fig_image_six_stamps(
    data: list[NDDataArray],
    skyposition: SkyCoord,
    size: u.Quantity,
    fsave: str | Path | None = None,
) -> None:
    ''' '''
    fig = makefig(figsize=('large', 0.675))
    fig.subplots_adjust(left=0.15, wspace=0.0, hspace=0.0)

    draw_stamp = DrawStamp(fig, skyposition=skyposition, size=size, nyx=(2, 3))
    for i, d in enumerate(data):
        ax = draw_stamp(d)
        ax.coords[0].set_ticks(spacing=1.5 * u.arcsec)
        draw_stamp.set_axislabels_only_at_edge(ax)

    fig.save_or_plot(fsave)


class DrawStamp:
    '''Main class to draw the figure.'''

    def __init__(
        self,
        fig: Figure,
        skyposition: SkyCoord,
        size: u.Quantity,
        nyx: tuple[int, int] = (2, 3),
    ) -> None:
        self.fig = fig
        self.nyx = nyx
        self.skyposition = skyposition
        self.size = size
        self.counter = 0
        self.wcsproj: WCS
        self.norm: Callable

    def __call__(
        self,
        data: NDDataArray | np.ndarray | Cutout2D,
        wcs: WCS | None = None,
        cmap: str = 'turbo',
        **kwargs
    ) -> WCSAxes:
        self.counter += 1

        if isinstance(data, Cutout2D):
            data = NDDataArray(data=data.data, wcs=data.wcs)
        if isinstance(data, np.ndarray):
            if wcs is None:
                raise ValueError('If data is numpy.ndarray, wcs must be needed.')
            data = NDDataArray(data=data, wcs=wcs)

        image = Cutout2D(data, self.skyposition, self.size)
        if self.counter == 1:
            self._set_imageconfig(image)

        ax: WCSAxes
        ax = self.fig.add_subplot(
            *self.nyx, self.counter, projection=self.wcsproj, slices=('x', 'y')
        )
        ax.imshow(
            image.data,
            origin='lower',
            norm=self.norm(),
            cmap=cmap,
            transform=ax.get_transform(image.wcs),
            **kwargs,
        )
        return ax

    def _set_imageconfig(self, image: Cutout2D) -> None:
        self.wcsproj = get_wcsproj_north_is_up(self.skyposition, self.size, image=image)
        self.norm = DS9LogNorm(
            xmin=np.min(image.data) / 10.0, xmax=np.max(image.data) / 2.0
        )

    def set_axislabels_only_at_edge(self, ax: WCSAxes) -> None:
        '''Set axis labels if the current axis is at the edge of the figure.'''
        i = self.counter - 1
        ny, nx = self.nyx

        if i % nx == 0:
            ax.coords[1].set_axislabel('Decl. (ICRS)')
        else:
            ax.coords[1].set_ticklabel_visible(False)
        if i // nx == ny - 1:
            ax.coords[0].set_axislabel('R.A (ICRS)')
        else:
            ax.coords[0].set_ticklabel_visible(False)
