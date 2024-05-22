'''Check environmental variables.
'''
import os
from pathlib import Path

PATH_DOWNLOAD = Path(os.environ.get('SGY_DATADOWNLOAD_PATH', None))
