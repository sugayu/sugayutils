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

__all__ = ['fig_spectral_noise_property', 'draw_spectral_noise_property']


##
def fig_spectral_noise_property(data, fsave: str | Path | None = None) -> None:
    '''Illustrate noise properties of the input spectrum.'''
    fig = makefig(figsize=['large', 0.33])
    fig.subplots_adjust(left=0.07, bottom=0.23, top=0.87, wspace=0.27)

    axs = draw_spectral_noise_property(fig, data)
    axs[0].set_title(r'F$\mathdefault{_\nu/\mu}$Jy')
    axs[0].set_xlabel(r'wavelength$\mathdefault{/\mu}$m')
    axs[1].set_title(r'Cont-sub. S/N')
    axs[1].set_xlabel(r'wavelength$\mathdefault{/\mu}$m')
    axs[2].set_title(r'PDF')
    axs[2].remove_xyticklabels()
    axs[3].set_title(r'Auto Correlation')
    axs[3].set_xlabel(r'pixel index')

    fig.save_or_plot(fsave)


def draw_spectral_noise_property(
    fig: Figure, data, c: str = 'black'
) -> npt.NDArray[np.object_]:
    '''Main function to draw the figure.'''
    axs = fig.subplots(1, 4, width_ratios=(1.0, 1.0, 0.08, 1.0))

    # spectrum
    axs[0].step(data.wave, data.flux, c=c, lw=1.0)
    axs[0].step(data.wave, data.cont, c='orange', lw=1.0)
    axs[0].fill_between(data.wave, data.err, -data.err, c='bgray', zorder=0.5)
    height = np.nanmax(data.flux) - np.nanmin(data.flux)
    axs[0].set_ylim(
        np.nanmin(data.flux) - height * 0.05, np.nanmax(data.flux) + height * 0.05
    )

    # subtracted
    yy = data.subtract / data.err
    axs[1].step(data.wave, yy, c=c, lw=1.0)
    axs[1].axhline(1.0, ls='--', c='gray')
    axs[1].axhline(-1.0, ls='--', c='gray')
    axs[1].axhline(0.0, ls='-', c='gray', lw=1.0)
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        data_masked, *bounds = sigma_clip(yy, sigma=5, return_bounds=True)
    ylim = (bounds[0], bounds[1])
    # ylim = axs[1].get_ylim()
    axs[1].set_ylim(ylim)
    pos_axs1 = axs[1].get_position()

    # distribution
    axs[2].hist(
        yy,
        20,
        orientation='horizontal',
        c='gray',
        density=True,
    )
    axs[2].set_ylim(ylim)
    x = np.arange(ylim[0], ylim[1], 0.1)
    axs[2].plot(np.exp(-0.5 * (x**2)) / np.sqrt(2 * np.pi), x, ls='--', c='black')
    pos_axs2 = axs[2].get_position()
    axs[2].set_position(
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
    ml, _, _ = axs[3].stem(
        np.arange(len(_acf)),
        data.err_autocorr[: len(_acf)],
        markerfmt=axs[3].colorful(c),
        linefmt=axs[3].colorful('gray'),
        basefmt=axs[3].colorful('black'),
    )
    ml.set_markersize(1.0)
    acferr = acferr[:, 0] - _acf
    axs[3].fill_between(np.arange(len(_acf)), acferr, -acferr, c='bgray', zorder=0.5)

    return axs
