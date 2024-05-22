'''Check environmental variables.
'''
import os
from pathlib import Path

if (_strpath := os.environ.get('SGY_DATADOWNLOAD_PATH', None)) is None:
    raise ValueError('The environment variable "SGY_DATADONWLOAD_PATH" is not defined.')

PATH_DOWNLOAD = Path(_strpath)

__all__ = ['PATH_DOWNLOAD']
