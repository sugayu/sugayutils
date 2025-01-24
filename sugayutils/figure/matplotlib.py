'''Wrapper of Matplotlib.
'''

from __future__ import annotations
from typing import Iterable, Sequence, TypeVar
from pathlib import Path
from logging import getLogger
from fontTools.ttLib import TTCollection
import numpy as np
import numpy.typing as npt
import matplotlib.figure as mplfig
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patheffects as path_effects
from ..core.const import colors
from ..core.misc import listup_instancevar
from ..stat.kde import KDE
from . import mlmodern

logger = getLogger(__name__)


__all__ = ['makefig', 'Axes', 'Figure', 'DS9LogNorm']


##
class Axes(mplaxes.Axes):
    '''Wrapper of Axes'''

    colornames: tuple = tuple(listup_instancevar(colors))

    def plot(
        self,
        *args,
        c: str | None = None,
        mec: str | None = None,
        mfc: str | None = None,
        m: str | None = None,
        **kwargs,
    ):
        '''Wrapper of plot'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if mec is not None:
            _kwargs['mec'] = self.colorful(mec)
        if mfc is not None:
            _kwargs['mfc'] = self.colorful(mfc)
        if m is not None:
            _kwargs['marker'] = m
        return super().plot(*args, **_kwargs)

    def scatter(
        self,
        *args,
        c: str | Iterable | None = None,
        mec: str | Iterable | None = None,
        mew: float | Iterable | None = None,
        m: str | None = None,
        **kwargs,
    ):
        '''Wrapper of scatter'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['c'] = self.colorful(c) if isinstance(c, str) else c
        if mec is not None:
            _kwargs['edgecolors'] = self.colorful(mec) if isinstance(mec, str) else mec
        if mew is not None:
            _kwargs['linewidths'] = mew
        if m is not None:
            _kwargs['marker'] = m
        return super().scatter(*args, **_kwargs)

    def errorbar(
        self,
        *args,
        c: str | None = None,
        m: str | None = None,
        ec: str | None = None,
        mec: str | None = None,
        elw: float | None = None,
        **kwargs,
    ):
        '''Wrapper of scatter'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if m is not None:
            _kwargs['marker'] = m
        if ec is not None:
            _kwargs['ecolor'] = self.colorful(ec)
        if mec is not None:
            _kwargs['markeredgecolor'] = self.colorful(mec)
        if elw is not None:
            _kwargs['elinewidth'] = elw
        return super().errorbar(*args, **_kwargs)

    def hist(self, *args, c: str | None = None, ec: str | None = None, **kwargs):
        '''Wrapper of scatter'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if ec is not None:
            _kwargs['ecolor'] = self.colorful(ec)
        if 'rwidth' not in _kwargs:
            _kwargs['rwidth'] = 0.95
        return super().hist(*args, **_kwargs)

    def contour(
        self,
        *args,
        c: str | None = None,
        lw: str | None = None,
        ls: str | None = None,
        **kwargs,
    ):
        '''Wrapper of plot'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['colors'] = self.colorful(c)
        if lw is not None:
            _kwargs['linewidths'] = lw
        if ls is not None:
            _kwargs['linestyles'] = ls
        return super().contour(*args, **_kwargs)

    def text(
        self,
        *args,
        c: str | None = None,
        borders: tuple[str, float] | list[tuple[str, float]] | None = None,
        **kwargs,
    ):
        '''Wrapper of text'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        txt = super().text(*args, **_kwargs)

        if borders is not None:
            border = styling_border(borders)
            txt.set_path_effects(border)
        return txt

    def axhline(self, *args, c: str | None = None, **kwargs):
        '''Wrapper of axhline'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        return super().axhline(*args, **_kwargs)

    def axvline(self, *args, c: str | None = None, **kwargs):
        '''Wrapper of axvline'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        return super().axvline(*args, **_kwargs)

    def fill_between(
        self, *args, c: str | None = None, ec: str | None = None, **kwargs
    ):
        '''Wrapper of fill_between.'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if ec is not None:
            _kwargs['edgecolor'] = self.colorful(ec)
        return super().fill_between(*args, **_kwargs)

    def fill_betweenx(
        self, *args, c: str | None = None, ec: str | None = None, **kwargs
    ):
        '''Wrapper of fill_betweenx.'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if ec is not None:
            _kwargs['edgecolor'] = self.colorful(ec)
        return super().fill_betweenx(*args, **_kwargs)

    def kdehist(
        self,
        data: np.ndarray,
        x: np.ndarray | None = None,
        is_fill: bool = False,
        is_mark: bool = False,
        fill_kwargs: dict = {},
        mark_kwargs: dict = {},
        lim: Sequence = [None, None],
        baseline: float | np.ndarray = 0.0,
        orientation: str = 'horizontal',
        **kwargs,
    ):
        '''Histogram with KDE.

        Args:
            data (np.ndarray): 1d data
            x (np.ndarray | None, optional): x data points to plot kde.
                Defaults to None.
            is_fill (bool, optional): If True, kde is filled. Defaults to False.
            is_mark (bool, optional): If True, plot markers at percentiles.
                Defaults to False.
            fill_kwargs (dict, optional): Keywords for ax.fill_between().
                Defaults to {}.
            mark_kwargs (dict, optional): Keywords for ax.scatter().
                Defaults to {}.
            lim (Sequence, optional): Tuple or list defining minimum and
                maximum of kde.Defaults to [None, None].
            baseline (float | np.ndarray, optional): y-value of x-axis.
                This argument is usefull to plot multiple kdes in one plot.
                Defaults to 0.0.
            orientation (str, optional): 'vertical' ('v') or 'horizontal' ('h').
                Defaults to horizontal'.
        '''
        if data.squeeze().ndim > 1:
            raise TypeError('kdehist can be only used for 1d input.')

        kernel = KDE(data)
        _data = kernel.dataset.squeeze()
        if x is None:
            range_data = _data.max() - _data.min()
            xmin = _data.min() - range_data * 0.4
            xmax = _data.max() + range_data * 0.4
            x = np.linspace(xmin, xmax, 100)
        pdf = kernel(x, lim) + baseline

        if orientation == 'horizontal' or orientation == 'h':
            xx, yy = x, pdf
            fill = self.fill_between
        elif orientation == 'vertical' or orientation == 'v':
            xx, yy = pdf, x
            fill = self.fill_betweenx
        else:
            raise ValueError('The orientation is "vertical" or "horizontal".')

        p = self.plot(xx, yy, **kwargs)
        if is_fill:
            fill(x, pdf, baseline, **fill_kwargs)
        if is_mark:
            xmarks = np.nanpercentile(data, [16, 50, 84])
            ymarks = kernel(xmarks, lim) + baseline
            if orientation == 'vertical' or orientation == 'v':
                xmarks, ymarks = ymarks, xmarks
            self.scatter(xmarks, ymarks, **mark_kwargs)
        return p

    def remove_xticklabel(self):
        '''Erase xlabel'''
        return self.xaxis.set_ticklabels('')

    def remove_yticklabel(self):
        '''Erase ylabel'''
        return self.yaxis.set_ticklabels('')

    def remove_xyticklabels(self):
        '''Erase both x and y labels'''
        self.xaxis.set_ticklabels('')
        self.yaxis.set_ticklabels('')

    def set_xylims(self, xlim, ylim):
        '''Set xlim and ylim at the same time.'''
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_xylabels(self, xlabel, ylabel, **kwargs):
        '''Set xlabel and ylable at the same time.'''
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)

    def remove_frame(self):
        '''Remove all contents of the frame.'''
        self.spines['top'].set_color('none')
        self.spines['bottom'].set_color('none')
        self.spines['left'].set_color('none')
        self.spines['right'].set_color('none')
        self.tick_params(
            which='both',
            labelcolor='none',
            top=False,
            bottom=False,
            left=False,
            right=False,
        )

    C = TypeVar('C', str, list)

    def colorful(self, color_key: C) -> C:
        '''Get favorite colors.'''
        if isinstance(color_key, str):
            if color_key in self.colornames:
                return getattr(colors, color_key)
            return color_key

        res = []
        if isinstance(color_key, list):
            for c in color_key:
                if c in self.colornames:
                    res.append(getattr(colors, c))
                else:
                    res.append(color_key)
            return res

        return color_key

    def dummy(self):
        '''Create dummy object to arange the legend order.'''
        (dummy,) = self.plot(0, 0, ls='None', marker='', label='')
        return dummy


class Figure(mplfig.Figure):
    '''Wrapper of Figure'''

    def subplots(
        self, *args, subplot_kw: dict = {}, **kwargs
    ) -> npt.NDArray[np.object_]:
        if ('projection' not in subplot_kw) and ('projection' not in kwargs):
            subplot_kw.setdefault('axes_class', Axes)
        return super().subplots(subplot_kw=subplot_kw, *args, **kwargs)

    def add_axes(self, *args, **kwargs) -> Axes:
        if 'projection' not in kwargs:
            kwargs.setdefault('axes_class', Axes)
        return super().add_axes(*args, **kwargs)

    def add_subplot(self, *args, **kwargs) -> Axes:
        if 'projection' not in kwargs:
            kwargs.setdefault('axes_class', Axes)
        return super().add_subplot(*args, **kwargs)

    def colorbar(
        self, mapparable, cax: Axes | None = None, ax_for_autopos=None, **kwargs
    ):
        '''Wrapper of Color bar.
        ax_for_autopos: automatically set color bar position.
        '''
        cax_ = super().colorbar(mapparable, cax=cax, **kwargs)
        if ax_for_autopos is not None:
            cax_ = autolocate_cax(cax_, ax_for_autopos, kwargs.get('location', 'right'))
        return cax_

    def save_or_plot(self, fname: str | Path | None = None, **kwargs) -> None:
        '''Save or plot figure.

        Save figure with that file name. If fname=None, plt.plot figure.

        Args:
            fname (str | Path | None, optional): File name for save. Defaults to None.
        '''
        if fname is None:
            plt.show(**kwargs)
        elif fname:
            self.savefig(fname, **kwargs)
            logger.info(f'Save fig in {fname}')
        self.clear()
        plt.close(self)

    def add_colorbar(
        self,
        mapping,
        axs=None,
        barratio: float = 0.5,
        barspace: float | None = None,
        **kwargs,
    ) -> None:
        '''Add colorbars with wise mecanisms to locate a position.

        Args:
            mapping: Mapping.
            axs: List of Axes. Defaults to None.
            barratio: Ratio of colorbar width (or height) to the space of widths (heights)
                of subplots. Defaults to 0.5.

        Returns:
            None:

        Examples:
            >>> fig = makefig()
            >>> axs = fig.subplots(2, 3)
            >>> im = axs[0].imshow(np.arange(900).reshape(30, 30))
            >>> fig.add_colorbar(im, axs=axs)
        '''

        try:
            axs = axs.ravel()
            pos0 = axs[0].get_position()
            pos_all = axs[0].get_position()
            for ax in axs:
                _pos = ax.get_position()
                pos_all.x0 = min(pos_all.x0, _pos.x0)
                pos_all.y0 = min(pos_all.y0, _pos.y0)
                pos_all.x1 = max(pos_all.x1, _pos.x1)
                pos_all.y1 = max(pos_all.y1, _pos.y1)
        except AttributeError:
            pos0 = axs.get_position()
            pos_all = axs.get_position()
        if barspace is None:
            barspace = barratio

        subpars = self.subplotpars
        w = subpars.wspace * pos0.width * barratio
        h = subpars.hspace * pos0.height * barratio
        ws = subpars.wspace * pos0.width * barspace
        hs = subpars.hspace * pos0.height * barspace
        right = pos_all.x1
        left = pos_all.x0
        top = pos_all.y1
        bottom = pos_all.y0
        fullwidth = pos_all.width
        fullheight = pos_all.height

        loc = kwargs.get('location', 'right')
        if loc == 'right':
            cax = self.add_axes((right + ws, bottom, w, fullheight))
        if loc == 'top':
            cax = self.add_axes((left, top + hs, fullwidth, h))
        if loc == 'left':
            cax = self.add_axes((left - 2 * ws, bottom, w, fullheight))
        if loc == 'bottom':
            cax = self.add_axes((left, bottom - 2 * hs, fullwidth, h))
        self.colorbar(mapping, cax=cax, **kwargs)


def autolocate_cax(cax, ax, location='right'):
    '''Automatically set position of colorbar'''
    ax_pos = ax.get_position()
    cax_pos = cax.ax.get_position()
    if location in ['right', 'left']:
        cax.ax.set_position([cax_pos.x0, ax_pos.y0, cax_pos.width, ax_pos.height])
    elif location in ['top', 'bottom']:
        cax.ax.set_position([ax_pos.x0, cax_pos.y0, ax_pos.width, cax_pos.height])
    return cax


def styling_border(types):
    '''Add borders to text and other objects.

    type is (color, width).
    '''
    if isinstance(types, tuple):
        fg = Axes.colorful(Axes, types[0])
        border = [
            path_effects.Stroke(foreground=fg, linewidth=types[1]),
            path_effects.Normal(),
        ]
    elif isinstance(types, list):
        border = [
            path_effects.Stroke(foreground=Axes.colorful(t[0]), linewidth=t[1])
            for t in types
        ]
        border.append(path_effects.Normal())
    return border


def makefig(**kwargs) -> Figure:
    '''Wrapper of plt.figure().'''
    _kwargs = kwargs.copy()
    if ('figsize' in kwargs) and ('a4' in kwargs['figsize']):
        _kwargs['figsize'] = (8.27, 11.69)
    if ('figsize' in kwargs) and ('small' in kwargs['figsize']):
        _kwargs['figsize'] = (3.5, 3.5 * kwargs['figsize'][1])
    if ('figsize' in kwargs) and ('large' in kwargs['figsize']):
        _kwargs['figsize'] = (7.3, 7.3 * kwargs['figsize'][1])
    return plt.figure(FigureClass=Figure, **_kwargs)


class DS9LogNorm:
    '''Log scale used in DS9.
    http://ds9.si.edu/doc/ref/how.html
    '''

    def __init__(self, xmin: float, xmax: float, a: float = 1000.0) -> None:
        _min = xmin
        _max = xmax

        def log_scale(data):
            scale = (data - _min) / (_max - _min)
            scale[scale < 0] = 0
            scale[scale > 1] = 1
            scale_log = np.log10(a * scale + 1) / np.log10(a)
            return scale_log

        def log_scale_inverse(scale_log):
            scale = (10.0 ** (scale_log * np.log10(a)) - 1) / a
            data = scale * (_max - _min) + _min
            return data

        self.min = _min
        self.max = _max
        self.log_scale = log_scale
        self.log_scale_inverse = log_scale_inverse

    def __call__(
        self, vmin: None | float = None, vmax: None | float = None
    ) -> mplcolors.FuncNorm:
        _vmin = vmin if vmin is not None else self.min
        _vmax = vmax if vmin is not None else self.max
        return mplcolors.FuncNorm(
            (self.log_scale, self.log_scale_inverse), vmin=_vmin, vmax=_vmax
        )


def convert_ttc_to_ttf(font: str | Path) -> None:
    '''Convert font files from ttc to ttf.

    This function is convenent to use ttc fonts in matplotlib.
    https://butami-study.com/python/51/
    '''
    if isinstance(font, str):
        pfont = Path(f'/System/Library/Fonts/{font}.ttc')
    else:
        pfont = font
    if not pfont.exists():
        raise FileNotFoundError(f'The ttc file "{pfont}" does not exist.')
    dsave = Path.home() / 'lib/font/'

    ttc = TTCollection(pfont)
    for ttf in ttc:
        fname = ttf['name'].getBestFullName()
        fsave = dsave / ttf['name'].getBestFamilyName() / f'{fname}.ttf'
        ttf.save(fsave)
        logger.info(f'Saved: {fsave}')
