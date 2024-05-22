'''Download JWST data
'''
from datetime import datetime
from logging import getLogger
from pathlib import Path
import numpy as np
from astroquery.mast import Observations
from astropy.table import Table, vstack
from .environment import PATH_DOWNLOAD

logger = getLogger(__name__)
subdir_nirspecifu = Path('JWST/NIRSpecIFU/')
__all__ = ['download_nirspecifu_rawdata']


##
def download_nirspecifu_rawdata(
    obs=None,
    dryrun: bool = True,
    dataRights: str = 'PUBLIC',
    nlimit: int = 10,
    **kwargs,
) -> Table:
    '''Download raw data of JWST NIRSpec IFU.

    Args:
        obs: Observation table obtained by previous queries. Defaults to None.
        dryrun (bool, optional): If True, downloads are not conducted. Defaults to True.
        dataRights (str, optional): Accessibility of the data, "PUBLIC" or "".
            Defaults to 'PUBLIC'.
        nlimit (int, optional): Number limit of the searched observations for downloads.
            Defaults to 10.
        kwargs (optional): Criteria to search observations.

    Returns:
        Table: Output observation table based on the input query.
               In addition, save downloaded files.

    Examples:
        >>> download_nirspecifu_rawdata(proposal_id=1840)
    '''
    if obs is None:
        obs = Observations.query_criteria(
            instrument_name='NIRSpec/IFU',
            calib_level=3,
            dataRights=dataRights,
            **kwargs,
        )
        if len(obs) > nlimit:
            msg = (
                f'Too many observations ({len(obs)} > {nlimit})'
                'satisfying the given criteria.'
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
    if dryrun:
        logger.info(f'Number of observations: {len(obs)}')
        for nm, fil, _id in zip(obs['target_name'], obs['filters'], obs['proposal_id']):
            logger.info(f'PID{_id}: {fil} -- {nm}')
        logger.info(f'Number of products: {len(product_list)}')
        for p in product_list['productFilename']:
            if '.fits' in p:
                logger.info(f'{dpath / p}')
        logger.info('These data will be moved to appropriate directries later.')
        return obs
    else:
        dpath.mkdir(exist_ok=True)
        logger.info(f'Number of observations: {len(obs)}')
        logger.info(f'Number of products: {len(product_list)}')
        logger.info(f'START: {datetime.today()}')
        manifest = Observations.download_products(
            product_list, download_dir=dpath, flat=True
        )
        logger.debug(manifest)
        logger.info(f'END: {datetime.today()}')

    now = datetime.today().strftime('%Y%m%d%H%M')
    product_list.write(dpath / f'downloads_{now}.ecsv', format='ascii.ecsv')
    manifest.write(dpath / f'manifest_{now}.ecsv', format='ascii.ecsv')

    for m in manifest:
        path = Path(m['Local Path'])
        info = product_list[np.where(path.name == product_list['productFilename'])]
        obs_i = obs[np.where(obs['obsid'] == info['parent_obsid'][0])]
        target_name = obs_i['target_name'][0]
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
            newpath: Path = path.parent / target_name / filtername / subdir
            newpath.mkdir(parents=True, exist_ok=True)
            path.rename(path.parent / target_name / filtername / subdir / path.name)
    return obs
