'''Convenent flux conversion tools.
'''

from __future__ import annotations
from logging import getLogger
from dataclasses import dataclass
import numpy as np
import astropy.units as u
import astropy.constants as c

__all__ = ['LineFlux']

logger = getLogger(__name__)


##
class LineFlux:
    '''Line flux conversion.'''

    def __init__(self, flux: u.Quantity, *, at: u.Quantity) -> None:
        self._original = flux
        self.wave = at.to(u.AA, u.spectral())
        self.nu = at.to(u.Hz, u.spectral())

        if flux.unit.physical_type == self.u.ergscm2.physical_type:
            self._flux = flux
            self._Sdv = self.to_jykms(flux)
        if flux.unit.physical_type == self.u.jykms.physical_type:
            self._Sdv = flux
            self._flux = self.to_ergscm2(flux)

    def to(self, unit: u.Unit) -> u.Quantity:
        if unit.physical_type == self.u.ergscm2.physical_type:
            return self._flux.to(unit)
        if unit.physical_type == self.u.jykms.physical_type:
            return self._Sdv.to(unit)

    def to_ergscm2(self, Sdv: u.Quantity) -> u.Quantity:
        dnu = 1.0 * u.Hz
        dv = c.c * dnu / self.nu
        return (Sdv / dv * dnu).to(self.u.ergscm2)

    def to_jykms(self, flux: u.Quantity) -> u.Quantity:
        dnu = 1.0 * u.Hz
        dv = c.c * dnu / self.nu
        return (flux / dnu * dv).to(self.u.jykms)

    @dataclass
    class LineFluxUnit:
        ergscm2: u.Unit = u.erg / u.s / u.cm**2
        jykms: u.Unit = u.Jy * u.km / u.s

    u = LineFluxUnit()

    def __str__(self):
        return f'{self.to(self.u.ergscm2)}'

    def __repr__(self):
        return f'<LineFlux {self.to(self.u.ergscm2)} == {self.to(self.u.jykms)}>'
