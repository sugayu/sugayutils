'''Conversion and scaling relations from literature.
'''
import numpy as np
from typing import NamedTuple

__all__ = ['IRXbeta']


##
class IRXbeta:
    '''IRX-beta relation.'''

    class ParamIRXbeta(NamedTuple):
        BC_UV: float
        dA_FUV_dbeta: float
        beta0: float

    param_Meurer = ParamIRXbeta(1.75, 1.99, -2.23)
    param_SMC = ParamIRXbeta(1.75, 1.1, -2.23)
    param_SMC_Reddy18 = ParamIRXbeta(1.75, 1.1, -2.62)
    param_SMC_Overzier = ParamIRXbeta(1.75, 1.99, -1.96)
    param_Fudamoto20_z45 = ParamIRXbeta(1.75, 0.71, -2.62)
    param_Fudamoto20_z55 = ParamIRXbeta(1.75, 0.48, -2.62)

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    def formula_IRX(
        self,
        BC_UV: float,
        dA_FUV_dbeta: float,
        beta0: float,
    ) -> np.ndarray:
        '''IRX-beta relation.
        The initial values are from
        '''
        return BC_UV * (10.0 ** (0.4 * dA_FUV_dbeta * (self.beta - beta0)) - 1)

    def Meurer(self) -> np.ndarray:
        '''Meurer et al. (1999)'''
        return self.formula_IRX(*self.param_Meurer)

    def SMC(self) -> np.ndarray:
        '''SMC e.g., Prevot et al. (1984)'''
        return self.formula_IRX(*self.param_SMC)

    def SMC_Reddy18(self) -> np.ndarray:
        '''SMC Reddy et al. (2018), beta0 = -2.62'''
        return self.formula_IRX(*self.param_SMC_Reddy18)

    def Fudamoto20_z45(self) -> np.ndarray:
        '''Fudamoto et al. (2020) at z=4.5'''
        return self.formula_IRX(*self.param_Fudamoto20_z45)

    def Fudamoto20_z55(self) -> np.ndarray:
        '''Fudamoto et al. (2020) at z=5.5'''
        return self.formula_IRX(*self.param_Fudamoto20_z55)
