'''Emission and Absorption lines
'''
import astropy.units as u
from dataclasses import dataclass

##
__all__ = ['LineWavelengthAt', 'LineList']


class LineWavelengthAt:
    '''Container returning  wavelengths in the observed frame at specific redshift.

    Example:
        >>> line = LineWavelengthAt(z=1.2)
    '''

    def __init__(self, z: float = 0) -> None:
        self.z = z
        self.redshift()

    def redshift(self) -> None:
        '''Set observed-frame line wavelengths into the instance variables.

        New instannce variables are Quality class.
        '''
        linelist = LineList()
        lines = [
            attr
            for attr in dir(linelist)
            if not callable(getattr(linelist, attr)) and not attr.startswith("__")
        ]
        for line in lines:
            setattr(self, line, getattr(linelist, line) * (1 + self.z) * u.AA)


@dataclass
class LineList:
    '''Line list in vacuum.

    Optical lines are in units of Aungstrom.
    The values are taken from atomic line list.
    '''

    # vacuum in Aungstrom
    OIII4363 = 4364.436
    OIII4959 = 4960.295
    OIII5007 = 5008.240
    OII3727 = 3727.092
    OII3729 = 3729.875
    OII_doublet = (3729.875 + 3727.092) / 2
    Ha = 6564.61
    Hb = 4862.683
    Hg = 4341.684
    NII6548 = 6549.85
    NII6584 = 6585.28
    HeII4686 = 4687.02
    HeI3889 = 3889.7475084
    HeI4471 = 4472.7290973
    HeI5876 = 5877.2432990
    HeI6678 = 6679.9955989
