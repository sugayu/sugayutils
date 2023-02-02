'''Wrapper of Matplotlib.
'''
import matplotlib.figure as mplfig
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt


__all__ = ['makefig']


##
class Axes(mplaxes.Axes):
    '''Wrapper of Axes'''

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
