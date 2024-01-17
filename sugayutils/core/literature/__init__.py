from . import dust_attenuation
from . import irxbeta
from .dust_attenuation import DustAttenuationLaw
from .irxbeta import IRXbeta

__all__ = []
__all__ += irxbeta.__all__
__all__ += dust_attenuation.__all__
