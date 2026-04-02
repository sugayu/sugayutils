'''Noise simulations.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
from logging import getLogger
import numpy as np
from numpy.random import default_rng

__all__ = ['get_noisesource']

logger = getLogger(__name__)


##
