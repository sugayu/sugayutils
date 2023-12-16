'''Modified blackbody function.
'''
import numpy as np
import astropy.units as u
from astropy.modeling import models
from astropy.cosmology import Planck18 as cosmo
import astropy.constants as c


##
# TODO: Making it class as inheritance of blackbody.
class ModifiedBlackBody:
    '''Modified blackbody function.'''

    def __init__(
        self,
        Mdust: u.Quantity,
        emissibity: float,
        Tdust: u.Quantity,
        kappa0: u.Quantity = 30.0 * u.cm**2 / u.g,
        z: float = 0.0,
    ) -> None:
        self.Mdust = Mdust
        self.emissibity = emissibity
        self._Tdust: float
        self._blackbody: models.BlackBody
        self.Tdust = Tdust
        self.kappa0 = kappa0

        self._z: float
        self._d: u.Quantity
        self._blackbody_cmb: u.Quantity
        self.z = z

    @property
    def Tdust(self) -> u.Quantity:
        return self._Tdust

    @Tdust.setter
    def Tdust(self, value: u.Quantity) -> None:
        self._Tdust = value
        self._blackbody = models.BlackBody(temperature=value)

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: float) -> None:
        '''Set redshift.'''
        self._z = value
        self._blackbody_cmb = models.BlackBody(temperature=cosmo.Tcmb(z=self._z))
        if self.z == 0:
            self._d = 10.0 * u.pc
        else:
            self._d = cosmo.luminosity_distance(z=self._z)

    def execute(self, frequency: u.Quantity) -> u.Quantity:
        '''Return intrinsic SED in units identical to luminosity.'''
        _frequency = frequency.to(u.Hz, u.spectral())
        _wave = frequency.to(u.um, u.spectral())
        kappa_nu = self.kappa0 * (100 * u.um / _wave) ** self.emissibity
        sr = 4 * np.pi * u.sr
        unit = u.erg / u.s / u.Hz
        return (sr * self.Mdust * kappa_nu * self._blackbody(_frequency)).to(unit)

    def observe(self, freq_obs: u.Quantity) -> u.Quantity:
        '''Return observed SED including the effect of CMB'''
        _freq_rest = (freq_obs / (1 + self.z)).to(u.Hz, u.spectral())
        _wave_rest = (freq_obs / (1 + self.z)).to(u.um, u.spectral())
        kappa_nu = self.kappa0 * (100 * u.um / _wave_rest) ** self.emissibity

        bb = self._blackbody(_freq_rest) - self._blackbody_cmb(_freq_rest)
        flux = u.sr * (1 + self.z) * self.Mdust * kappa_nu * bb / self._d**2
        return flux.to(u.uJy)

    def get_IRluminosity(self) -> u.Quantity:
        '''Return IR luminosity including the CMB effect.'''
        wave = np.arange(8.0, 1000) * u.um
        wave_obs = wave * (1 + self.z)
        lum_nu = 4.0 * np.pi * self._d**2 * self.observe(wave_obs)
        unit = u.erg / u.s / u.um
        lum_lam = lum_nu.to(unit, u.spectral_density(wave_obs))
        return np.sum(lum_lam * 1.0 * u.um * (1 + self.z)).to(u.Lsun)

    def __call__(self, frequency: u.Quantity) -> u.Quantity:
        return self.execute(frequency)
