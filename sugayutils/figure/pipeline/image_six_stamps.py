'''Plot six images of stars and galaxies.
'''

from pathlib import Path
from typing import Callable
import numpy as np
import astropy.units as u
from astropy.nddata import CCDData, Cutout2D
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord
from sugayutils.figure.matplotlib import DS9LogNorm
from sugayutils.figure import makefig, Figure, get_wcsproj_north_is_up


__all__ = ['fig_image_six_stamps', 'DrawStamp']


##
def fig_image_six_stamps(
    fnames: list[Path],
    skyposition: SkyCoord,
    size: u.Quantity,
    fsave: str | Path | None = None,
) -> None:
    ''' '''
    fig = makefig(figsize=('large', 0.675))
    fig.subplots_adjust(left=0.15, wspace=0.0, hspace=0.0)

    data: CCDData
    draw_stamp = DrawStamp(fig, skyposition=skyposition, size=size, nyx=(2, 3))
    for i, f in enumerate(fnames):
        data = CCDData.read(f)
        ax = draw_stamp(data)

        print(type(ax))

        ax.coords[0].set_ticks(spacing=1.5 * u.arcsec)
        if i % 3 == 0:
            ax.coords[1].set_axislabel('Decl. (ICRS)')
        else:
            ax.coords[1].set_ticklabel_visible(False)
        if i // 3 == 1:
            ax.coords[0].set_axislabel('R.A (ICRS)')
        else:
            ax.coords[0].set_ticklabel_visible(False)

    fig.save_or_plot()


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

    def __call__(self, data: CCDData, **kwargs) -> WCSAxes:
        self.counter += 1
        image = Cutout2D(data, self.skyposition, self.size)
        if self.counter == 1:
            self.set_imageconfig(image)

        ax: WCSAxes
        ax = self.fig.add_subplot(
            *self.nyx, self.counter, projection=self.wcsproj, slices=('x', 'y')
        )
        ax.imshow(
            image.data,
            origin='lower',
            norm=self.norm(),
            cmap='turbo',
            transform=ax.get_transform(image.wcs),
            **kwargs,
        )
        return ax

    def set_imageconfig(self, image: Cutout2D) -> None:
        self.wcsproj = get_wcsproj_north_is_up(self.skyposition, self.size, image=image)
        self.norm = DS9LogNorm(
            xmin=np.min(image.data) / 10.0, xmax=np.max(image.data) / 5.0
        )
