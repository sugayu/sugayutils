'''Utilities for statistics
'''

from . import correlation
from .correlation import autocorrelation
from . import kde
from .kde import KDE

__all__ = correlation.__all__
__all__ += kde.__all__
