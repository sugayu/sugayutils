'''Wrapper of Matplotlib.
'''
from __future__ import annotations
from typing import Iterable
import numpy as np
import matplotlib.figure as mplfig
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patheffects as path_effects
from ..core.const import colors
from ..core.misc import listup_instancevar

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
        if not 'rwidth' in _kwargs:
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

    def set_xylabels(self, xlabel, ylabel):
        '''Set xlabel and ylable at the same time.'''
        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

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

    def colorful(self, color_key: str) -> str:
        '''Get favorite colors.'''
        if color_key in self.colornames:
            return getattr(colors, color_key)
        return color_key


class Figure(mplfig.Figure):
    '''Wrapper of Figure'''

    def subplots(self, *args, subplot_kw: dict = {}, **kwargs) -> list[Axes]:
        subplot_kw.setdefault('axes_class', Axes)
        return super().subplots(subplot_kw=subplot_kw, *args, **kwargs)

    def add_axes(self, *args, **kwargs) -> Axes:
        kwargs.setdefault('axes_class', Axes)
        return super().add_axes(*args, **kwargs)

    def add_subplot(self, *args, **kwargs) -> Axes:
        kwargs.setdefault('axes_class', Axes)
        return super().add_subplot(*args, **kwargs)

    def colorbar(self, *args, ax_for_autopos=None, **kwargs):
        '''Wrapper of Color bar.
        ax_for_autopos: automatically set color bar position.
        '''
        cax = super().colorbar(*args, **kwargs)
        if ax_for_autopos is not None:
            cax = autolocate_cax(cax, ax_for_autopos, kwargs.get('location', 'right'))
        return cax


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
    return plt.figure(FigureClass=Figure, **kwargs)


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
