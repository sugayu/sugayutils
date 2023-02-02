'''Wrapper of Matplotlib.
'''
import matplotlib.figure as mplfig
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt


__all__ = ['makefig']


##
class Axes(mplaxes.Axes):
    '''Wrapper of Axes'''

    def hello(self):
        print('hello')


class Figure(mplfig.Figure):
    '''Wrapper of Figure'''

    def subplots(self, subplot_kw=None, *args, **kwargs):
        if subplot_kw is None:
            subplot_kw = {'axes_class': Axes}
        else:
            subplot_kw['axes_class'] = Axes
        axes = super().subplots(subplot_kw=subplot_kw, *args, **kwargs)
        return axes

    def add_axes(self, *args, **kwargs):
        ax = super().add_axes(axes_class=Axes, *args, **kwargs)
        return ax

    def add_subplot(self, *args, **kwargs):
        ax = super().add_subplot(axes_class=Axes, *args, **kwargs)
        return ax


def makefig(**kwargs):
    '''Wrapper of plt.figure().'''
    return plt.figure(FigureClass=Figure, **kwargs)
