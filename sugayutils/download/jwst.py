'''Download JWST data
'''

from concurrent.futures import ThreadPoolExecutor
from threading import current_thread
from queue import Queue
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
    nthread: int = 10,
    chunksize_download: int = 250,
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
        nthread (int, optional): Number of threads for downloads, but NOTE that
            the actual number of threads is roughly x2 more than nthread, beacause
            this function provides two download steps and both steps use as many threads
            as nthread. Defaults to 10.
        chunksize_download (int, optional): The chunksize used for deviding
            the product list. Defaults to 250.
        kwargs (optional): Criteria to search observations.

    Returns:
        Table: Output observation table based on the input query.
               In addition, save downloaded files.

    Examples:
        >>> download_nirspecmsa_calibdata(proposal_id=1180)
    '''
    global global_count_thread
    global_count_thread = 0
    dpath = PATH_DOWNLOAD / subdir_nirspecmsa
    dpath.mkdir(exist_ok=True)

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

    obslist = []
    for i in range(len(obs) // chunksize_download + 1):
        j0, j1 = i * chunksize_download, (i + 1) * chunksize_download
        if j0 == len(obs):
            break
        j1 = j1 if j1 <= len(obs) else len(obs)
        obslist.append(obs[j0:j1])

    if dryrun:
        product_list = get_product_list__nirspecmsa_calibdata(obslist[0])
        for i, (nm, fil, _id) in enumerate(
            zip(obs['target_name'], obs['filters'], obs['proposal_id'])
        ):
            logger.info(f'PID{_id}: {fil} -- {nm}')
            if i > 50:
                logger.info('...')
                break
        logger.info(f'Number of products in the first chunk: {len(product_list)}')
        for i, p in enumerate(product_list['productFilename']):
            if '.fits' in p:
                logger.info(f'{dpath / p}')
            if i > 100:
                logger.info('...')
                break
        logger.info('These data will be moved to appropriate directories later like:')
        return obs

    logger.info(f'START: {datetime.today()}')
    queue_product: Queue = Queue()
    queue_manifest: Queue = Queue()
    with ThreadPoolExecutor(nthread) as exe1, ThreadPoolExecutor(nthread) as exe2:
        futures_product, futures_manifest = [], []
        for obs in obslist:
            future = exe1.submit(
                get_product_list__nirspecmsa_calibdata, obs, queue_product
            )
            futures_product.append(future)

        for _ in obslist:
            future = exe2.submit(
                download_products__nirspecmsa_calibdata,
                dpath,
                queue_product,
                queue_manifest,
            )
            futures_manifest.append(future)

        for _ in obslist:
            _product_list, _manifest = queue_manifest.get()
            for m in _manifest:
                arrange_dirs__nirspecmsa_calibdata(obs, _product_list, m)

        product_list = vstack([future.result() for future in futures_product])
        manifest = vstack([future.result() for future in futures_manifest])
    logger.info(f'END: {datetime.today()}')

    assert queue_product.empty() is True
    assert queue_manifest.empty() is True

    logger.info('====Download results====')
    logger.info(f'Number of observations: {len(obs)}')
    logger.info(f'Number of products: {len(product_list)}')
    now = datetime.today().strftime('%Y%m%d%H%M')
    product_list.write(dpath / f'downloads_{now}.ecsv', format='ascii.ecsv')
    manifest.write(dpath / f'manifest_{now}.ecsv', format='ascii.ecsv')
    logger.info(
        f'List written in {dpath}/downloads_{now}.ecsv and /manifest_{now}.ecsv'
    )
    logger.info('Download Done.')
    return obs


def get_product_list__nirspecmsa_calibdata(
    obs: Table, queue: Queue | None = None
) -> Table:
    '''Wrapper of Observations.get_product_list()'''
    global global_count_thread
    p = Observations.get_product_list(obs)
    logger.info(
        'Created product list in ' '{current_thread().getName()} #{global_count_thread}'
    )

    L2c = p['calib_level'] == 3
    global_count_thread += 1
    _p = p[np.where(L2c)]
    if queue is not None:
        queue.put(_p)
    return _p


def download_products__nirspecmsa_calibdata(
    dpath: Path, queue_product: Queue, queue_manifest: Queue
) -> Table:
    '''Download throught queue'''
    product_list = queue_product.get()
    manifest = Observations.download_products(
        product_list, download_dir=dpath, flat=True
    )
    queue_manifest.put((product_list, manifest))
    return manifest


def arrange_dirs__nirspecmsa_calibdata(
    obs: Table, product_list: Table, manifest: Table
) -> None:
    path = Path(manifest['Local Path'])
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
