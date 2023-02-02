'''Wrapper of Matplotlib.
'''
from __future__ import annotations
from typing import Iterable
import matplotlib.figure as mplfig
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt
from ..core.const import colors
from ..core.misc import listup_instancevar

__all__ = ['makefig']


##
class Axes(mplaxes.Axes):
    '''Wrapper of Axes'''

    colornames: tuple = tuple(listup_instancevar(colors))

    def plot(
        self,
        *args,
        c: None | str = None,
        mec: None | str = None,
        mfc: None | str = None,
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
        return super().plot(*args, **_kwargs)

    def scatter(
        self,
        *args,
        c: None | str | Iterable = None,
        mec: None | str | Iterable = None,
        **kwargs,
    ):
        '''Wrapper of scatter'''
        _kwargs = kwargs.copy()
        if c is not None:
            if isinstance(c, str):
                _kwargs['c'] = self.colorful(c)
            else:
                _kwargs['c'] = c
        if mec is not None:
            if isinstance(mec, str):
                _kwargs['edgecolors'] = self.colorful(mec)
            else:
                _kwargs['edgecolors'] = mec
        return super().scatter(*args, **_kwargs)

    def errorbar(self, *args, c: None | str = None, ec: None | str = None, **kwargs):
        '''Wrapper of scatter'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        if ec is not None:
            _kwargs['ecolor'] = self.colorful(ec)
        return super().scatter(*args, **_kwargs)

    def text(self, *args, c: None | str = None, **kwargs):
        '''Wrapper of text'''
        _kwargs = kwargs.copy()
        if c is not None:
            _kwargs['color'] = self.colorful(c)
        return super().scatter(*args, **_kwargs)

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
            labelcolor='none', top=False, bottom=False, left=False, right=False
        )

    def colorful(self, color_key: str) -> str:
        '''Get favorite colors.'''
        if color_key in self.colornames:
            return getattr(colors, color_key)
        return color_key


class Figure(mplfig.Figure):
    '''Wrapper of Figure'''

    def subplots(self, *args, subplot_kw=None, **kwargs) -> list[Axes]:
        if subplot_kw is None:
            subplot_kw = {'axes_class': Axes}
        else:
            subplot_kw['axes_class'] = Axes
        axes = super().subplots(subplot_kw=subplot_kw, *args, **kwargs)
        return axes

    def add_axes(self, *args, **kwargs) -> Axes:
        ax = super().add_axes(axes_class=Axes, *args, **kwargs)
        return ax

    def add_subplot(self, *args, **kwargs) -> Axes:
        ax = super().add_subplot(axes_class=Axes, *args, **kwargs)
        return ax


def makefig(**kwargs) -> Figure:
    '''Wrapper of plt.figure().'''
    return plt.figure(FigureClass=Figure, **kwargs)
