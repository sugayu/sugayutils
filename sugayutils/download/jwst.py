'''Download JWST data
'''
from datetime import datetime
from logging import getLogger
from pathlib import Path
import numpy as np
from astroquery.mast import Observations
from astropy.table import vstack
from .environment import PATH_DOWNLOAD

logger = getLogger(__name__)
subdir_nirspecifu = Path('JWST/NIRSpecIFU/')


##
def download_nirspecifu_rawdata(
    dryrun: bool = True, dataRights: str = 'PUBLIC', nlimit: int = 10, **kwargs
) -> None:
    '''Download raw data of JWST NIRSpec IFU.

    Args:
        dryrun (bool, optional): If True, downloads are not conducted. Defaults to True.
        dataRights (str, optional): Accessibility of the data, "PUBLIC" or "".
            Defaults to 'PUBLIC'.
        nlimit (int, optional): Number limit of the searched observations for downloads.
            Defaults to 10.
        kwargs (optional): Criteria to search observations.

    Returns:
        None: Save downloaded files.

    Examples:
        >>> download_nirspecifu_rawdata(proposal_id=1840)
    '''
    obs = Observations.query_criteria(
        instrument_name='NIRSpec/IFU', calib_level=3, dataRights=dataRights, **kwargs
    )
    if len(obs) > nlimit:
        msg = (
            f'Too many observations ({len(obs)} > {nlimit}) satisfying'
            'the given criteria.'
        )
        logger.error(msg)
        raise ValueError(msg)

    products = []
    for _obs in obs:
        p = Observations.get_product_list(_obs)
        uncal = (p['calib_level'] == 1) & (p['productType'] == ['SCIENCE'])
        miscs = (p['calib_level'] == 2) & (
            (p['productType'] == 'PREVIEW') | (p['productType'] == 'INFO')
        )
        products.append(p[np.where(uncal | miscs)])
    product_list = vstack(products)
    dpath = PATH_DOWNLOAD / subdir_nirspecifu

    logger.info(f'Number of observations: {len(obs)}')
    logger.info(f'Number of products: {len(product_list)}')
    if dryrun:
        for p in product_list['productFilename']:
            logger.info(f'{dpath / p}')
        logger.info('These data will be moved to appropriate directries later.')
        return
    else:
        logger.info(datetime.today())
        manifest = Observations.download_products(
            product_list, download_dir=dpath, flat=True
        )
        logger.debug(manifest)
        logger.info(datetime.today())

    now = datetime.today().strftime('%Y%m%d%H%M')
    product_list.write(dpath / f'downloads_{now}.ecsv', format='ascii.ecsv')
    manifest.write(dpath / f'manifest_{now}.ecsv', format='ascii.ecsv')

    for m in manifest:
        path = Path(m['Local Path'])
        info = product_list[np.where(path.name == product_list['productFilename'])]
        filtername = ''.join(info['filters'][0].split(';'))
        if path.suffix == '.fits':
            subdir = 'raw'
        elif path.suffix == '.jpg':
            subdir = 'images'
        elif path.suffix == '.png':
            subdir = 'images'
        elif path.suffix == '.json':
            subdir = 'cal'
        elif path.suffix == '.csv':
            subdir = 'cal'
        if path.exists():
            path.rename(path.parent / filtername / subdir / path.name)
