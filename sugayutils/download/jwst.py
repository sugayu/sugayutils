'''Download JWST data
'''

from concurrent.futures import ThreadPoolExecutor
from threading import current_thread
import asyncio
from datetime import datetime
from logging import getLogger
from pathlib import Path
import numpy as np
from astroquery.mast import Observations
from astropy.table import Table, vstack
from .environment import PATH_DOWNLOAD

logger = getLogger(__name__)
subdir_nirspecifu = Path('JWST/NIRSpecIFU/')
subdir_nirspecmsa = Path('JWST/NIRSpecMSA/')
__all__ = ['download_nirspecifu_rawdata', 'download_nirspecmsa_calibdata']


##
global_count_thread = 0


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
        logger.info('These data will be moved to appropriate directories later.')
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

    logger.info('Arranging directory structure.')
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


def download_nirspecmsa_calibdata(
    obs=None,
    dryrun: bool = True,
    dataRights: str = 'PUBLIC',
    nlimit: int = 1000,
    nprocess: int = 2,
    **kwargs,
) -> Table:
    '''Download calibrated data of JWST NIRSpec MSA.

    Args:
        obs: Observation table obtained by previous queries. Defaults to None.
        dryrun (bool, optional): If True, downloads are not conducted. Defaults to True.
        dataRights (str, optional): Accessibility of the data, "PUBLIC" or "".
            Defaults to 'PUBLIC'.
        nlimit (int, optional): Number limit of the searched observations for downloads.
            Defaults to 1000.
        kwargs (optional): Criteria to search observations.

    Returns:
        Table: Output observation table based on the input query.
               In addition, save downloaded files.

    Examples:
        >>> download_nirspecmsa_calibdata(proposal_id=1180)
    '''
    if obs is None:
        obs = Observations.query_criteria(
            instrument_name='NIRSpec/MSA',
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
    logger.info(f'Number of observations: {len(obs)}')

    n_download = 250

    # products = []
    # for i in range(len(obs) // n_download):
    #     j0, j1 = i * n_download, (i + 1) * n_download
    #     j1 = j1 if j1 <= len(obs) else len(obs)
    #     _obs = obs[j0:j1]
    #     logger.info(f'Observations {j0}--{j1-1}: Creating product list...')
    #     products.append(get_product_list(_obs))

    obslist = []
    for i in range(len(obs) // n_download + 1):
        j0, j1 = i * n_download, (i + 1) * n_download
        if j0 == len(obs):
            break
        j1 = j1 if j1 <= len(obs) else len(obs)
        obslist.append(obs[j0:j1])
        # logger.info(f'Observations {j0}--{j1-1}: Creating product list...')
        # products.append(get_product_list(_obs))
    with ThreadPoolExecutor(10) as executor:
        products = list(executor.map(get_product_list, obslist))

    # obslist = []
    # for i in range(len(obs) // n_download + 1):
    #     j0, j1 = i * n_download, (i + 1) * n_download
    #     if j0 == len(obs):
    #         break
    #     j1 = j1 if j1 <= len(obs) else len(obs)
    #     obslist.append(obs[j0:j1])
    #     # logger.info(f'Observations {j0}--{j1-1}: Creating product list...')
    #     # products.append(get_product_list(_obs))
    # products = asyncio.run(create_product_list_async(obslist))

    # if j1 <= len(obs):
    #     logger.info(f'Objects {j0}--{j1-1}: Downloading...')

    #     # These codes are taken from Observations.get_product_list()
    #     service = Observations._caom_products
    #     params = {'obsid': ','.join(obs[j0:j1]['obsid']), 'calib_level': 3}
    #     responses = Observations._portal_api_connection.service_request_async(
    #         service, params
    #     )
    #     p = Observations._parse_result(responses)
    #     # p = Observations.get_product_list(obs[j0:j1])
    # else:
    #     logger.info(f'Objects {j0}--{len(obs)-1}: Downloading...')
    #     service = Observations._caom_products
    #     params = {'obsid': ','.join(obs[j0:]['obsid']), 'calib_level': 3}
    #     responses = Observations._portal_api_connection.service_request_async(
    #         service, params
    #     )
    #     p = Observations._parse_result(responses)
    #     # p = Observations.get_product_list(obs[j0:])
    # L2c = p['calib_level'] == 3
    # products.append(p[np.where(L2c)])
    # # product_list = p[np.where(L2c)]
    product_list = vstack(products)

    dpath = PATH_DOWNLOAD / subdir_nirspecmsa
    if dryrun:
        logger.info(f'Number of observations: {len(obs)}')
        for i, (nm, fil, _id) in enumerate(
            zip(obs['target_name'], obs['filters'], obs['proposal_id'])
        ):
            logger.info(f'PID{_id}: {fil} -- {nm}')
            if i > 100:
                logger.info('...')
                break
        logger.info(f'Number of products: {len(product_list)}')
        for i, p in enumerate(product_list['productFilename']):
            if '.fits' in p:
                logger.info(f'{dpath / p}')
            if i > 300:
                logger.info('...')
                break
        logger.info('These data will be moved to appropriate directories later.')
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

    logger.info('Arranging directory structure.')
    for m in manifest:
        path = Path(m['Local Path'])
        info = product_list[np.where(path.name == product_list['productFilename'])]
        obs_i = obs[np.where(obs['obsid'] == info['parent_obsid'][0])]
        target_name = obs_i['target_name'][0]
        filtername = ''.join(info['filters'][0].split(';'))
        if path.suffix == '.fits':
            subdir = 'product'
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
    logger.info('Download Done.')
    return obs


# products = list(pool.map(get_product_list, obslist))


async def create_product_list_async(obslist) -> list:
    tasks = [get_product_list_async(_obs) for _obs in obslist]
    return await asyncio.gather(*tasks)


def get_product_list(obs: Table) -> list:
    '''Wrapper of Observations.get_product_list()'''
    # These codes are taken from Observations.get_product_list()
    # service = Observations._caom_products
    # params = {'obsid': ','.join(obs[j0:j1]['obsid']), 'calib_level': 3}
    # responses = Observations._portal_api_connection.service_request_async(
    #     service, params
    # )
    # p = Observations._parse_result(responses)
    global global_count_thread
    logger.info(
        'Creating product list in '
        '{current_thread().getName()} #{global_count_thread}...'
    )
    p = Observations.get_product_list(obs)
    L2c = p['calib_level'] == 3
    global_count_thread += 1
    return p[np.where(L2c)]


async def get_product_list_async(obs: Table) -> list:
    '''Wrapper of Observations.get_product_list()'''
    # These codes are taken from Observations.get_product_list()
    # service = Observations._caom_products
    # params = {'obsid': ','.join(obs[j0:j1]['obsid']), 'calib_level': 3}
    # responses = Observations._portal_api_connection.service_request_async(
    #     service, params
    # )
    # p = Observations._parse_result(responses)
    logger.info(f'Creating product list in {current_thread().getName()}...')
    loop = asyncio.get_event_loop()
    responces = await loop.run_in_executor(
        None, Observations.get_product_list_async, obs
    )
    p = Observations._parse_result(responces)
    L2c = p['calib_level'] == 3
    return p[np.where(L2c)]
