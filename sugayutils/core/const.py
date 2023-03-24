'''Setting constants
'''
from dataclasses import dataclass
from astropy import units as u

##
__all__ = ['Colors', 'AstroConst', 'colors']


@dataclass
class Colors:
    '''Default colors
    https://jfly.uni-koeln.de/colorset/
    '''

    blue: str = '#005aff'  # '0173b2' seaborn
    orange: str = '#f6aa00'  # 'de8f05'
    green: str = '#03af7a'  # '029e73'
    red: str = '#ff4b00'  # 'd55e00'
    purple: str = '#990099'  # 'cc78bc'
    brown: str = '#804000'  # 'ca9161'
    pink: str = '#ff8082'  # 'fbafe4'
    gray: str = '#84919e'  # '949494'
    grey: str = '#84919e'  # '949494'
    yellow: str = '#fff100'  # 'ece133'
    sky: str = '#4dc4ff'  # '56b4e9'
    black: str = '#323232'
    borange: str = '#ffca80'
    bgreen: str = '#77d9a8'
    bygreen: str = '#d8f255'
    bpurple: str = '#c9ace6'
    bpink: str = '#ffcabf'
    byellow: str = '#ffff80'
    bsky: str = '#bfe4ff'
    bgray: str = '#c8c8cb'
    bgrey: str = '#c8c8cb'


@dataclass
class AstroConst:
    '''Constants for Astronomy'''

    logOHsun: u.Quantity = 8.67 * u.dimensionless_unscaled


colors = Colors()
