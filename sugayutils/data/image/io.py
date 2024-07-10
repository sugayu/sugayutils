'''I/O of 2d images.
'''
import numpy as np

from ...fitting.fit2d import FitGauss2d


##
class Image2D():
    '''2D image.
    '''
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
