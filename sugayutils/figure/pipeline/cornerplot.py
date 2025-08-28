'''Corner plot for MCMC.
'''

from __future__ import annotations
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde, norm
from .. import makefig, Figure, Axes
from ...stat.kde import KDE
from logging import getLogger

logger = getLogger(__name__)


___all__ = ['fig_cornerplot', 'Cornerplot']


##
def fig_cornerplot(data: np.ndarray, fsave: str | Path | None = None) -> None:
    '''Illustrate corner plot.'''

    fig = makefig(figsize=['large', 1.0])
    fig.subplots_adjust(0.1, 0.05, 0.9, 0.95, hspace=0.05, wspace=0.05)

    cornerplot = CornerPlotter(data)
    axs = cornerplot(fig)

    fig.save_or_plot(fsave)


class CornerPlotter:
    '''Corner plot class.'''

    def __init__(
        self,
        data: np.ndarray,
        colnames: list[str] | None = None,
        limits: list[tuple[float, float]] | None = None,
        *,
        f_margin: float = 10 / 100,
        nbins_hist: int = 20,
        nbins_scatter: int = 30,
        npix_kde: int = 500,
        sigmas_contour: list[float] = [0.2, 1.0, 2.0],
        kdehist: bool = True,
        kde2d: bool = True,
    ) -> None:
        self.data = data
        self.shape = data.shape
        if colnames is not None:
            if len(colnames) != self.shape[1]:
                raise ValueError(
                    f'The length of colnames, {len(colnames)}, '
                    f'needs to match the shape[1] of data {self.shape[1]}.'
                )
            self.colnames = colnames
        else:
            self.colnames = [''] * self.shape[1]
        self.limits: list[tuple[float, float]] | list[tuple[None, None]]
        if limits is not None:
            if len(limits) != self.shape[1]:
                raise ValueError(
                    'The length of limits needs to match the shape[1] of data. '
                    'A staticmethod "get_limits" may help to initialize limits.'
                )
            self.limits = limits
        else:
            self.limits = self.get_limits(self.shape[1])

        self.f_margin = f_margin
        self.nbins_hist = nbins_hist
        self.nbins_scatter = nbins_scatter
        self.npix_kde = npix_kde
        self.sigmas_contour = sigmas_contour
        self.kdehist = kdehist
        self.kde2d = kde2d

    def cornerplot(self, fig: Figure) -> npt.NDArray[np.object_]:
        '''Main function.'''

        axs = fig.subplots(self.shape[1], self.shape[1])

        for ncol in range(self.shape[1]):
            column = self.data[:, ncol]

            for nrow in range(self.shape[1]):
                ax = axs[nrow, ncol]
                if self.is_blank(ncol, nrow):
                    ax.remove_frame()
                    continue

                if self.is_diagonal(ncol, nrow):
                    self.histogram(ax, column, ncol)
                    ax.set_title(self.colnames[ncol], fontsize='small')

                row = self.data[:, nrow]
                if self.is_lowertriangle(ncol, nrow):
                    self.scatter(ax, column, row, ncol, nrow)

                if ncol > 0:
                    ax.remove_yticklabel()
                if nrow < self.shape[1] - 1:
                    ax.remove_xticklabel()
                ax.tick_params(labelsize='x-small')

        return axs

    @staticmethod
    def get_limits(length: int) -> list[tuple[float, float]] | list[tuple[None, None]]:
        '''Get initial limits of parameters.'''
        limits: list[tuple[None, None]] = []
        for _ in range(length):
            limits.append((None, None))
        return limits

    def histogram(
        self,
        ax: Axes,
        column: np.ndarray,
        ncol: int,
    ) -> None:
        '''Plot histogram.'''
        margin = (column.max() - column.min()) * self.f_margin
        _range = (column.min() - margin, column.max() + margin)
        hist, edges, *_ = ax.hist(
            column, self.nbins_hist, c='gray', alpha=0.5, density=True
        )

        if self.kdehist:
            try:
                kernel = KDE(column)
                xx = np.linspace(_range[0], _range[1], self.npix_kde)
                counts = kernel(xx, lim=self.limits[ncol])
                ax.plot(xx, counts, c='gray', lw=1.0)
            except np.linalg.LinAlgError:
                logger.exception(f'The KDE line cannot be drawn at col {ncol}.')
            # * (edges[1] - edges[0]) * len(column)

        ax.remove_yticklabel()
        ax.set_xlim(_range)

    @staticmethod
    def draw_bestfitlines(ax: Axes, best: np.ndarray, limit: np.ndarray) -> None:
        '''UNDERDEVELOPMENT; Draw best-fit lines.

        If limit, the upper/lower limits are drawn.
        '''
        bins = 0  # hack
        if len(best) == 3:
            ax.axvline(best[0], ls='--', c='black')
            ax.axvline(best[0] + best[1], ls='--', c='black')
            ax.axvline(best[0] + best[2], ls='--', c='black')
        if len(best) == 1:
            ax.axvline(best[0], ls='--', c='black')
            if limit:
                ax.arrow(
                    x=best[0],
                    y=0.14,
                    dx=-2 * bins,
                    dy=0.0,
                    color='black',
                    width=0.003,
                    head_length=bins,
                    length_includes_head=True,
                )

    def scatter(
        self,
        ax: Axes,
        column: np.ndarray,
        row: np.ndarray,
        ncol: int,
        nrow: int,
    ) -> None:
        '''Scatter plot in lower triangle panels.'''

        margin_col = (column.max() - column.min()) * self.f_margin
        range_ = [(column.min() - margin_col, column.max() + margin_col)]
        margin_row = (row.max() - row.min()) * self.f_margin
        range_ += [(row.min() - margin_row, row.max() + margin_row)]

        counts, xedges, yedges = np.histogram2d(
            column, row, self.nbins_scatter, range=range_
        )
        ax.scatter(column, row, c='gray', mec='None', alpha=0.4, scale_factor=0.5)

        # counts = gaussian_filter(counts, 0.4)
        # while counts.max() > 30:
        #     nbin = nbin + 1
        #     counts, xedges, yedges = np.histogram2d(
        #         column, row, nbin, range=range_
        #     )
        #     # counts = gaussian_filter(counts, 0.4)

        if self.kde2d:
            xx, yy = np.meshgrid(xedges, yedges)
            grids = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(np.stack([column, row]))
            counts = np.reshape(kernel(grids).T, xx.shape).T

            counts1d = np.sort(counts.ravel())
            cumsum = np.cumsum(counts1d)
            cumsum = cumsum / cumsum[-1]
            sigmas = [1.0 - (norm.cdf(v) - norm.cdf(-v)) for v in self.sigmas_contour]
            levels = [counts1d[cumsum < s][-1] for s in np.sort(sigmas)]

            # ax.contourf(
            #     counts.transpose(),
            #     levels=levels,
            #     colors='white',
            #     extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            # )
            # ax.hist2d(column, row, 12, cmap='Greys', cmin=levels[0] + 0.1)
            ax.contour(
                counts.transpose(),
                levels=levels,
                c=['black', 'black', 'black'],
                lw=[1.0, 1.5, 1.5],
                extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
                origin='lower',
            )
        ax.set_xylims(range_[0], range_[1])

    @staticmethod
    def is_diagonal(ncol: int, nrow: int) -> bool:
        return ncol == nrow

    @staticmethod
    def is_lowertriangle(ncol: int, nrow: int) -> bool:
        return ncol < nrow

    @staticmethod
    def is_blank(ncol: int, nrow: int) -> bool:
        return ncol > nrow

    def configure(
        self,
        *,
        f_margin: float | None = None,
        nbins_hist: int | None = None,
        nbins_scatter: int | None = None,
        npix_kde: int | None = None,
        sigmas_contour: list[float] | None = None,
        kdehist: bool | None = None,
        kde2d: bool | None = None,
    ) -> None:
        if f_margin is not None:
            self.f_margin = f_margin
        if nbins_hist is not None:
            self.nbins_hist = nbins_hist
        if nbins_scatter is not None:
            self.nbins_scatter = nbins_scatter
        if npix_kde is not None:
            self.npix_kde = npix_kde
        if sigmas_contour is not None:
            self.sigmas_contour = sigmas_contour
        if kdehist is not None:
            self.kdehist = kdehist
        if kde2d is not None:
            self.kde2d = kde2d

    __call__ = cornerplot
