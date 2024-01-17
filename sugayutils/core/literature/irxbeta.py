'''IRX-beta relations from literature.
'''
import numpy as np
from typing import NamedTuple

__all__ = ['IRXbeta']


##
class IRXbeta:
    '''IRX-beta relation.'''

    class ParamIRXbeta(NamedTuple):
        BC: float
        dA_FUV_dbeta: float
        beta0: float

    param_Meurer_original = ParamIRXbeta(1.66 / 1.4, 1.99, -2.23)
    param_Meurer_totalIR = ParamIRXbeta(1.66, 1.99, -2.23)
    param_Calzetti00_original = ParamIRXbeta(1.11, 2.31, -2.1)
    param_Calzetti00_totalIR = ParamIRXbeta(1.11 * 1.75, 2.31, -2.1)
    param_Takeuchi12 = ParamIRXbeta(1.66, 1.58, -1.94)
    param_SMC = ParamIRXbeta(1.75, 1.1, -2.23)
    param_SMC_Reddy18 = ParamIRXbeta(1.79, 1.07, -2.62)
    param_SMC_Overzier = ParamIRXbeta(1.75, 1.99, -1.96)
    param_Fudamoto20_z45 = ParamIRXbeta(1.75, 0.71, -2.62)
    param_Fudamoto20_z55 = ParamIRXbeta(1.75, 0.48, -2.62)

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    def formula_IRX(
        self,
        BC: float,
        dA_FUV_dbeta: float,
        beta0: float,
    ) -> np.ndarray:
        '''IRX-beta relation.
        The initial values are from
        '''
        return BC * (10.0 ** (0.4 * dA_FUV_dbeta * (self.beta - beta0)) - 1)

    def Meurer_original(self) -> np.ndarray:
        '''Meurer et al. (1999)'''
        return self.formula_IRX(*self.param_Meurer_original)

    def Meurer_totalIR(self) -> np.ndarray:
        '''Meurer et al. (1999) using total IR.

        Original Meurer+99 is derived for FIR luminosity.
        This is corrected to the total IR luminosity, by setting BC_IR = 1.0.
        '''
        return self.formula_IRX(*self.param_Meurer_totalIR)

    def Calzetti00_original(self) -> np.ndarray:
        '''Calzetti et al. (2000)'''
        return self.formula_IRX(*self.param_Calzetti00_original)

    def Calzetti00_totalIR(self) -> np.ndarray:
        '''Calzetti et al. (2000) using total IR.

        Original Calzetti+00 is derived for FIR luminosity.
        This is corrected to the total IR luminosity, by setting BC_IR = 1.0.
        '''
        return self.formula_IRX(*self.param_Calzetti00_totalIR)

    def Takeuchi12(self) -> np.ndarray:
        '''Takeuchi et al. (2012)'''
        return self.formula_IRX(*self.param_Takeuchi12)

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
