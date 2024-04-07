'''Logging utilities.
'''
from datetime import datetime
import inspect
from pathlib import Path
from logging import config as logconfig
from typing import Optional
import yaml

__all__ = ['mylogconfig']


##
def mylogconfig(
    level: str = 'INFO',
    print_console: bool = True,
    print_file: bool = False,
    filename: Optional[str] = None,
) -> None:
    '''My log configuration.

    Args:
        level (str, optional): Information level. Defaults to 'INFO'.
        print_console (bool, optional): If True, print log in the console.
            Defaults to True.
        print_file (bool, optional): If True, print log in a file.
            Defaults to False.
        filename (Optional[str], optional): Filename to write log. The output
            filename include datetime in addition to the input filename.
            For example, if filename is "logfile", the output filename is like
            "logfile-20240101-00h00m00s.log". If this parameter is not given
            (i.e., None), instead, the name of the script file that called
            "mylogconfig" will be used. Defaults to None.

    Examples:
        >>> mylogconfig()
    '''
    if filename is None:
        filename = Path(inspect.stack()[1].filename).stem
        if '<ipython' in filename:
            filename = 'ipython'

    config = get_default_mylogconfig()
    config = modify_logconfig(
        config,
        level=level,
        print_console=print_console,
        print_file=print_file,
        filename=filename,
    )
    logconfig.dictConfig(config)


def modify_logconfig(
    config: dict,
    level: str = 'INFO',
    print_console: bool = True,
    print_file: bool = False,
    filename: str = 'log',
) -> dict:
    '''Modify default log format to input configurations.'''
    config = config.copy()
    if not print_console:
        del config['handlers']['consoleHandler']
        config['root']['handlers'].remove('consoleHandler')
    if print_file:
        date = datetime.today().strftime('%Y%m%d-%Hh%Mm%Ss')
        path = Path(f'log/{filename}_{date}.log')
        if not path.parent.exists():
            raise FileNotFoundError('No "log/" in the current working directory.')
        config['handlers']['fileHandler']['filename'] = str(path)
    else:
        del config['handlers']['fileHandler']
        config['root']['handlers'].remove('fileHandler')
    config['root']['level'] = level
    return config


def get_default_mylogconfig() -> dict:
    '''Return default configuration of mylog format.'''
    path = Path(__file__).parent
    with open(path / 'log_config.yaml') as f:
        log_conf = yaml.safe_load(f.read())
    return log_conf
