'''Download JWST data
'''

from concurrent.futures import ThreadPoolExecutor, Future
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
    downloader = NIRSpecMSADownloader()

    if obs is None:
        downloader.query_criteria(kwargs)
        if downloader.n_obs > nlimit:
            msg = (
                f'Too many observations ({downloader.n_obs} > {nlimit})'
                'satisfying the given criteria.'
            )
            logger.error(msg)
            raise ValueError(msg)
    else:
        downloader.obs = obs
    assert isinstance(downloader.obs, Table)
    logger.info(f'Number of observations: {downloader.n_obs}')

    obslist = downloader.chunk_observations(chunksize_download)

    if dryrun:
        product_list = downloader.get_product_list(obslist[0])
        obs = downloader.obs
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
                logger.info(f'{downloader.dpath / p}')
            if i > 100:
                logger.info('...')
                break
        logger.info('These data will be moved to appropriate directories later like:')
        return obs

    logger.info(f'START: {datetime.today()}')
    with ThreadPoolExecutor(nthread) as exe1, ThreadPoolExecutor(nthread) as exe2:
        downloader.run_get_product_list(obslist, exe1)
        downloader.run_download_products(exe2)
        downloader.run_arrange_dirs()
        product_list, manifest = downloader.result()
    logger.info(f'END: {datetime.today()}')

    downloader.assert_emptyqueue()

    logger.info('====Download results====')
    logger.info(f'Number of observations: {len(obs)}')
    logger.info(f'Number of products: {len(product_list)}')
    now = datetime.today().strftime('%Y%m%d%H%M')
    downloader.write_results(now)
    logger.info(
        f'List written in {downloader.dpath}/downloads_{now}.ecsv and /manifest_{now}.ecsv'
    )
    logger.info('Download Done.')
    return obs


class NIRSpecMSADownloader:
    '''Download MSA data in multi-thread.'''

    path_download = PATH_DOWNLOAD
    subdir_nirspecmsa = Path('JWST/NIRSpecMSA/')
    dataRights = 'PUBLIC'
    instrument_name = 'NIRSpec/MSA'

    def __init__(self) -> None:
        self.count_thread: int = 0
        self.dpath = self.path_download / self.subdir_nirspecmsa
        self.dpath.mkdir(exist_ok=True)
        self.obs: Table | None = None
        self.product_list: Table | None = None
        self.manifest: Table | None = None
        self.queue_product: Queue = Queue()
        self.futures_product: list[Future] = []
        self.queue_manifest: Queue = Queue()
        self.futures_manifest: list[Future] = []

    def query_criteria(
        self, insttrument_name=instrument_name, dataRights=dataRights, **kwargs
    ) -> Table:
        '''Return results of observations query.'''
        self.obs = Observations.query_criteria(
            instrument_name=insttrument_name,
            calib_level=3,
            dataRights=dataRights,
            **kwargs,
        )

    def chunk_observations(self, size: int) -> list:
        '''Make a list of chunks of queried observations.'''
        if self.obs is None:
            raise ValueError('obs is None.')

        obslist = []
        for i in range(self.n_obs // size + 1):
            j0, j1 = i * size, (i + 1) * size
            if j0 == self.n_obs:
                break
            j1 = j1 if j1 <= self.n_obs else self.n_obs
            obslist.append(self.obs[j0:j1])
        return obslist

    @property
    def n_obs(self) -> int:
        if self.obs is None:
            raise ValueError('obs is None.')
        return len(self.obs)

    def run_get_product_list(
        self, obslist: list[Table], exe: ThreadPoolExecutor
    ) -> None:
        '''Run Observations.get_product_list in units of observation chunks.'''
        for obs in obslist:
            future = exe.submit(self.get_product_list, obs)
            self.futures_product.append(future)

    def get_product_list(self, obs_chunk: Table) -> Table:
        '''Wrapper of Observations.get_product_list()'''
        product = Observations.get_product_list(obs_chunk)
        logger.info(
            'Created product list in '
            '{current_thread().getName()} #{global_count_thread}'
        )

        L2c = product['calib_level'] == 3
        self.count_thread += 1
        product = product[np.where(L2c)]
        return product

    def run_download_products(self, exe: ThreadPoolExecutor) -> None:
        '''Download data in units of observation chuncks.'''
        if not self.futures_product:
            raise ValueError('Product list is not ready.')
        for _ in self.futures_product:
            future = exe.submit(self.download_products)
            self.futures_manifest.append(future)

    def download_products(self) -> Table:
        '''Download throught queue'''
        product_list = self.queue_product.get()
        manifest = Observations.download_products(
            product_list, download_dir=self.dpath, flat=True
        )
        self.queue_manifest.put((product_list, manifest))
        return manifest

    def run_arrange_dirs(self) -> None:
        '''Arrange directories in threads.'''
        for _ in self.futures_manifest:
            _product_list, _manifest = self.queue_manifest.get()
            for m in _manifest:
                self.arrange_dirs(_product_list, m)

    def arrange_dirs(self, product_list: Table, manifest: Table) -> None:
        '''Arrange directory structure for downloaded data.'''
        assert self.obs is not None

        path = Path(manifest['Local Path'])
        info = product_list[np.where(path.name == product_list['productFilename'])]
        obs_i = self.obs[np.where(self.obs['obsid'] == info['parent_obsid'][0])]
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

    def result(self) -> tuple[Table, Table]:
        '''Wait results of product_list and manifest as results of download.'''
        self.product_list = vstack([future.result() for future in self.futures_product])
        self.manifest = vstack([future.result() for future in self.futures_manifest])
        return self.product_list, self.manifest

    def assert_emptyqueue(self) -> None:
        assert self.queue_product.empty() is True
        assert self.queue_manifest.empty() is True

    def write_results(self, tag: str) -> None:
        if self.product_list is None:
            raise ValueError('Product list is not ready.')
        if self.manifest is None:
            raise ValueError('Manifest is not ready: Download has not been finished.')
        self.product_list.write(
            self.dpath / f'downloads_{tag}.ecsv', format='ascii.ecsv'
        )
        self.manifest.write(self.dpath / f'manifest_{tag}.ecsv', format='ascii.ecsv')
