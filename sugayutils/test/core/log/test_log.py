import logging
import pytest
from pathlib import Path
from ....core.log import log


##
def test_mylogconfig():
    with pytest.raises(FileNotFoundError) as e:
        log.mylogconfig(print_console=False, print_file=True)
    assert 'No "log/"' in str(e.value)

    p = Path('log/')
    p.mkdir()
    log.mylogconfig(print_console=False, print_file=True)
    logpath = Path(logging.root.manager.root.handlers[0].baseFilename)
    assert 'test_log' in logpath.name
    logpath.unlink()
    p.rmdir()


def test_modify_logconfig():
    config = log.get_default_mylogconfig()
    config0 = log.modify_logconfig(config)
    assert isinstance(config0, dict)
    assert 'fileHandler' not in config0['root']['handlers']
