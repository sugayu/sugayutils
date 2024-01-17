'''Dust attenuation law.
'''
import numpy as np
import astropy.units as u


__all__ = ['DustAttenuationLaw']


##
class DustAttenuationLaw:
    '''Dust attenuation law, A(lambda)/A(V).

    F(lambda) = F_int(lambda) * 10** (-Av * A(lambda)/A(V) /2.5)
    '''

    def __init__(self) -> None:
        ...

    def Calzetti00(self, wave: u.Quantity) -> np.ndarray:
        '''Calzetti et al. (2000), which is taken from Bagpipes'''
        attenuation = np.empty_like(wave.value)
        _wave = wave.to(u.um, u.spectral())

        idx1 = _wave < 1200.0 * u.AA
        idx2 = (_wave < 6300.0 * u.AA) & (_wave >= 1200.0 * u.AA)
        idx3 = (_wave < 31000.0 * u.AA) & (_wave >= 6300.0 * u.AA)

        _fac1 = -2.156 + 1.509 / 0.12 - 0.198 / 0.12**2 + 0.011 / 0.12**3
        factor1 = 4.05 + 2.695 * _fac1
        attenuation[idx1] = (_wave.value[idx1] / 0.12) ** -0.77 * factor1
        attenuation[idx2] = 4.05 + 2.695 * (
            -2.156
            + 1.509 / _wave.value[idx2]
            - 0.198 / _wave.value[idx2] ** 2
            + 0.011 / _wave.value[idx2] ** 3
        )
        attenuation[idx3] = 2.659 * (-1.857 + 1.040 / _wave.value[idx3]) + 4.05
        attenuation /= 4.05
        return attenuation
