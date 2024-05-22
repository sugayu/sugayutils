'''Helper functions to download data from public data archives.
'''
from .environment import PATH_DOWNLOAD

if PATH_DOWNLOAD is None:
    raise ValueError('The environment variable "SGY_DATADONWLOAD_PATH" is not defined.')
