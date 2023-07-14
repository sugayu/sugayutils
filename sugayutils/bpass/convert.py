'''Convert BPASS stellar models into other input formats.
'''
import numpy as np
from astropy.io import fits


##
def convert_for_bagpipes(
    fnames: list[str],
    zmet: np.ndarray,
    fnames_starmass: list[str],
    do_include_compactmass: bool = False,
) -> fits.HDUList:
    '''Convert files into fits format used by Bagpipes.

    In the config.py of Bagpipes, users can use the example format written in the bottom of config.py.

    Args:
        fnames (list[str]): Filenames of BPASS `spectra-XXX.dat`.
        zmet (np.ndarray): Array of metallicity of BPASS data in units of solar metallicity.
            The order of the metallicity must be the same as files given in `fnames`.
        fnames_starmass (list[str]): Filenames of BPASS `starmass-XXX.dat`.
        do_include_compactmass (bool, optional): If True, include the mass of compact objects
            in the living fraction of stellar mass. Defaults to False.

    Returns:
        fits.HDUList: Fits file HDU list, including data of BPASS.
            The format is almost the same as the SB99 fits file default in Bagpipes.

    Examples:
        >>> fnames = ['spectra-bin-imf135_300.a+06.z001.dat',
                      'spectra-bin-imf135_300.a+06.z004.dat.gz',
                      'spectra-bin-imf135_300.a+06.z020.dat.gz']
        >>> zmet = np.ndarray([0.001, 0.004, 0.02]) / 0.02
        >>> fnames_starmass = [
                'starmass-bin-imf135all_100.z001.dat.gz',
                'starmass-bin-imf135all_100.z004.dat.gz',
                'starmass-bin-imf135all_100.z020.dat.gz',
                ]
        >>> hdul = convert_for_bagpipes(fnames, zmet, fnames_starmass)
        >>> hdul.writeto('bpass_2.3_bin_stellar_grids_example.fits')
    '''
    if len(fnames) != len(zmet):
        raise ValueError('Lengths of fnames and zmet must be same.')
    if len(fnames) != len(fnames_starmass):
        raise ValueError('Lengths of fnames and fnames_starmass must be same.')

    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())
    liv_mstar_frac = []

    for i, (f, z, fm) in enumerate(zip(fnames, zmet, fnames_starmass)):
        # main spectral data
        data = np.loadtxt(f)
        if i == 0:
            n_age = data.shape[1] - 1
            wavelength = data[:, 0]
        spectra = data[:, 1:]
        hdu_list.append(fits.ImageHDU(spectra.T, name=f'ZMET{z:0.4f}ZSOL'))

        # needed for computing LIV_MSTAR_FRAC after for loop
        data_sm = np.loadtxt(fm)
        if do_include_compactmass:
            remaining_mass = data_sm[:, 1] + data_sm[:, 2]
        else:
            remaining_mass = data_sm[:, 1]
        _liv_mstar_frac = remaining_mass / 1.0e6
        _liv_mstar_frac[_liv_mstar_frac > 1] = 1  # some are higher than 1
        liv_mstar_frac.append(_liv_mstar_frac)

    hdu_list.append(fits.ImageHDU(np.vstack(liv_mstar_frac).T, name='LIV_MSTAR_FRAC'))
    age = 10 ** (6 + 0.1 * (np.arange(n_age)))
    hdu_list.append(fits.ImageHDU(age, name='STELLAR_AGE_YR'))
    hdu_list.append(fits.ImageHDU(wavelength, name='WAVELENGTHS_AA'))

    return fits.HDUList(hdu_list)
