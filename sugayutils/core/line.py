'''Emission and Absorption lines
'''
import astropy.units as u
from dataclasses import dataclass
from .misc import listup_instancevar

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
        lines = listup_instancevar(linelist)
        for line in lines:
            setattr(self, line, getattr(linelist, line) * (1 + self.z) * u.AA)

    def asdict(self, unit: str = 'AA') -> dict:
        '''Create line dictionary.'''
        lines = listup_instancevar(self)
        dictionary = {}
        for line in lines:
            if line == 'z':
                continue
            dictionary[line] = getattr(self, line).to(unit)
        return dictionary


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
    Hdelta = 4102.892
    Hepsilon = 3971.195
    H6 = 3890.151
    NII6548 = 6549.85
    NII6584 = 6585.28
    HeII4686 = 4687.02
    HeI3889 = 3889.7475084
    HeI4471 = 4472.7290973
    HeI5876 = 5877.2432990
    HeI6678 = 6679.9955989
    HeI7067 = 7067.12521
    Lya = 1215.6700
    NIII1744 = 1744.351
    NIII1747 = 1746.823
    NIII1750 = 1749.674
    NIII1752 = 1752.160
    NIII1754 = 1753.995
    NIII1483 = 1483.321
    OIII1660 = 1660.8092
    OIII1666 = 1666.1497
    HeII1640 = 1640.42
    CIII1906 = 1906.683
    CIII1908 = 1908.734
    CIV1548 = 1548.203
    CIV1550 = 1550.777
    OIII88 = 883560
    OIII52 = 518145
    CII158 = 1577409
    NII122 = 1218976
    NII205 = 2051783
