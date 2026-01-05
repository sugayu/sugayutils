'''Emission and Absorption lines
'''

import astropy.units as u
from dataclasses import dataclass
from .misc import listup_instancevar

##
__all__ = ['LineWavelengthAt', 'LineList']


@dataclass
class LineList:
    '''Line list in vacuum.

    Optical lines are in units of Aungstrom.
    The values are taken from atomic line list.
    '''

    # vacuum in Aungstrom
    OIII4363: u.Quantity = 4364.436 * u.AA
    OIII4959: u.Quantity = 4960.295 * u.AA
    OIII5007: u.Quantity = 5008.240 * u.AA
    OII3727: u.Quantity = 3727.092 * u.AA
    OII3729: u.Quantity = 3729.875 * u.AA
    OII_doublet: u.Quantity = (3729.875 + 3727.092) / 2 * u.AA
    Ha: u.Quantity = 6564.61 * u.AA
    Hb: u.Quantity = 4862.683 * u.AA
    Hg: u.Quantity = 4341.684 * u.AA
    Hdelta: u.Quantity = 4102.892 * u.AA
    Hepsilon: u.Quantity = 3971.195 * u.AA
    H6: u.Quantity = 3890.151 * u.AA
    NII6548: u.Quantity = 6549.85 * u.AA
    NII6584: u.Quantity = 6585.28 * u.AA
    HeII4686: u.Quantity = 4687.02 * u.AA
    HeI3889: u.Quantity = 3889.7475084 * u.AA
    HeI4471: u.Quantity = 4472.7290973 * u.AA
    HeI5876: u.Quantity = 5877.2432990 * u.AA
    HeI6678: u.Quantity = 6679.9955989 * u.AA
    HeI7067: u.Quantity = 7067.12521 * u.AA
    HeI10833: u.Quantity = 10833.0 * u.AA
    SII6718: u.Quantity = 6718.29 * u.AA
    SII6733: u.Quantity = 6732.67 * u.AA
    SIII9071: u.Quantity = 9071.1 * u.AA
    SIII9533: u.Quantity = 9533.2 * u.AA
    Paa: u.Quantity = 18756.13 * u.AA
    Pab: u.Quantity = 12821.59 * u.AA
    Pag: u.Quantity = 10941.091 * u.AA
    Pad: u.Quantity = 10052.128 * u.AA
    Lya: u.Quantity = 1215.6700 * u.AA
    NIII1744: u.Quantity = 1744.351 * u.AA
    NIII1747: u.Quantity = 1746.823 * u.AA
    NIII1750: u.Quantity = 1749.674 * u.AA
    NIII1752: u.Quantity = 1752.160 * u.AA
    NIII1754: u.Quantity = 1753.995 * u.AA
    NIII1483: u.Quantity = 1483.321 * u.AA
    OIII1660: u.Quantity = 1660.8092 * u.AA
    OIII1666: u.Quantity = 1666.1497 * u.AA
    HeII1640: u.Quantity = 1640.42 * u.AA
    CIII1906: u.Quantity = 1906.683 * u.AA
    CIII1908: u.Quantity = 1908.734 * u.AA
    CIV1548: u.Quantity = 1548.203 * u.AA
    CIV1550: u.Quantity = 1550.777 * u.AA
    OIII88: u.Quantity = 883560 * u.AA
    OIII52: u.Quantity = 518145 * u.AA
    CII158: u.Quantity = 1577409 * u.AA
    NII122: u.Quantity = 1218976 * u.AA
    NII205: u.Quantity = 2051783 * u.AA


class LineWavelengthAt(LineList):
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
            setattr(self, line, getattr(linelist, line) * (1 + self.z))

    def asdict(self, unit: str = 'AA') -> dict:
        '''Create line dictionary.'''
        lines = listup_instancevar(self)
        dictionary = {}
        for line in lines:
            if line == 'z':
                continue
            dictionary[line] = getattr(self, line).to(unit)
        return dictionary
