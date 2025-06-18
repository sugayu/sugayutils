'''Illustrate noise properties of spectra.
'''

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
from numpy.random import default_rng
import numpy.typing as npt
from astropy.stats import sigma_clip
from statsmodels.tsa.stattools import acf
from sugayutils.figure import makefig, Figure
from logging import getLogger

logger = getLogger(__name__)


__all__ = ['fig_spectral_noise_property', 'draw_spectral_noise_property']


##
def fig_spectral_noise_property(data, fsave: str | Path | None = None) -> None:
    '''Illustrate noise properties of the input spectrum.'''
    fig = makefig(figsize=['large', 0.33])
    fig.subplots_adjust(left=0.07, bottom=0.23, top=0.87, wspace=0.27)

    axs0, axs1 = draw_spectral_noise_property(fig, data)
    axs0[0, 0].set_title(r'F$\mathdefault{_\nu/\mu}$Jy')
    axs0[0, 0].set_xlabel(r'wavelength$\mathdefault{/\mu}$m')
    axs0[0, 1].set_title(r'Cont-sub. S/N')
    axs0[0, 1].set_xlabel(r'wavelength$\mathdefault{/\mu}$m')
    axs0[0, 2].set_title(r'PDF')
    axs0[0, 2].remove_xyticklabels()
    axs0[0, 3].set_title(r'Auto Correlation')
    axs0[0, 3].set_xlabel(r'pixel index')

    fig.save_or_plot(fsave)


def draw_spectral_noise_property(
    fig: Figure, data, c: str = 'black'
) -> tuple[npt.NDArray[np.object_], ...]:
    '''Main function to draw the figure.'''
    axs0 = fig.subplots(2, 4, width_ratios=(1.0, 1.0, 0.08, 1.0))
    axs1 = fig.subplots(2, 3)

    # spectrum
    axs0[0, 0].step(data.wave, data.flux, c=c, lw=1.0)
    axs0[0, 0].step(data.wave, data.cont, c='orange', lw=1.0)
    axs0[0, 0].fill_between(data.wave, data.err, -data.err, c='bgray', zorder=0.5)
    height = np.nanmax(data.flux) - np.nanmin(data.flux)
    if hasattr(data, 'mask'):
        axs0[0, 0].fill_between(
            *(data.wave, 0, 1),
            where=data.mask,
            c='bgray',
            transform=axs0[0, 0].get_xaxis_transform(),
            ec='None',
            zorder=-1,
        )
    axs0[0, 0].set_ylim(
        np.nanmin(data.flux) - height * 0.05, np.nanmax(data.flux) + height * 0.05
    )

    # subtracted
    yy = data.subtract / data.err
    axs0[0, 1].step(data.wave, yy, c=c, lw=1.0)
    axs0[0, 1].axhline(1.0, ls='--', c='gray')
    axs0[0, 1].axhline(-1.0, ls='--', c='gray')
    axs0[0, 1].axhline(0.0, ls='-', c='gray', lw=1.0)
    if hasattr(data, 'mask'):
        axs0[0, 1].fill_between(
            *(data.wave, 0, 1),
            where=data.mask,
            c='bgray',
            transform=axs0[0, 1].get_xaxis_transform(),
            ec='None',
            zorder=-1,
        )
        _yy = np.copy(yy)
        _yy[~data.mask.astype(bool)] = np.nan
        axs0[0, 1].step(data.wave, _yy, c='gray', lw=1.0)
        yy = yy[~data.mask.astype(bool)]
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        data_masked, *bounds = sigma_clip(yy, sigma=5, return_bounds=True)
    if bounds[1] - bounds[0] > 40:
        bounds = (-20, 20)
        logger.warning('Too large bounds. Sigma clipping may be failed.')
    ylim = (bounds[0], bounds[1])
    # ylim = axs[1].get_ylim()
    axs0[0, 1].set_ylim(ylim)
    pos_axs1 = axs0[0, 1].get_position()

    # distribution
    axs0[0, 2].hist(
        yy,
        20,
        orientation='horizontal',
        c='gray',
        density=True,
        range=ylim,
    )
    axs0[0, 2].set_ylim(ylim)
    x = np.arange(ylim[0], ylim[1], 0.1)
    axs0[0, 2].plot(np.exp(-0.5 * (x**2)) / np.sqrt(2 * np.pi), x, ls='--', c='black')
    pos_axs2 = axs0[0, 2].get_position()
    axs0[0, 2].set_position(
        (pos_axs1.x1, pos_axs2.y0, pos_axs2.x1 - pos_axs1.x1, pos_axs2.height)
    )

    # auto correlation function
    rng = default_rng(0)
    _acf, acferr = acf(
        rng.standard_normal(len(data.subtract)),
        adjusted=True,
        # nlags=len(data) - 1,
        alpha=0.05,
    )
    ml, _, _ = axs0[0, 3].stem(
        np.arange(len(_acf)),
        data.err_autocorr[: len(_acf)],
        markerfmt=axs0[0, 3].colorful(c),
        linefmt=axs0[0, 3].colorful('gray'),
        basefmt=axs0[0, 3].colorful('black'),
    )
    ml.set_markersize(1.0)
    acferr = acferr[:, 0] - _acf
    axs0[0, 3].fill_between(
        np.arange(len(_acf)), acferr, -acferr, c='bgray', zorder=0.5
    )
    axs0[1, 0].remove_frame()
    axs0[1, 1].remove_frame()
    axs0[1, 2].remove_frame()
    axs0[1, 3].remove_frame()

    # correlations
    axs1[1, 0].scatter(yy[:-1], yy[1:], s=2**2, c='gray', mec=(1, 1, 1, 0.5), mew=0.4)
    axs1[1, 0].text(0.1, 0.9, 'Index 1', transform=axs1[1, 0].transAxes)
    axs1[1, 1].scatter(yy[:-2], yy[2:], s=2**2, c='gray', mec=(1, 1, 1, 0.5), mew=0.4)
    axs1[1, 1].text(0.1, 0.9, 'Index 2', transform=axs1[1, 1].transAxes)
    axs1[1, 2].scatter(yy[:-3], yy[3:], s=2**2, c='gray', mec=(1, 1, 1, 0.5), mew=0.4)
    axs1[1, 2].text(0.1, 0.9, 'Index 3', transform=axs1[1, 2].transAxes)
    axs1[0, 0].remove_frame()
    axs1[0, 1].remove_frame()
    axs1[0, 2].remove_frame()

    return axs0, axs1
