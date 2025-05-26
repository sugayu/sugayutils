'''Download JWST data
'''

from concurrent.futures import ThreadPoolExecutor, Future
from threading import current_thread
from queue import Queue
from datetime import datetime
from logging import getLogger
import logging
from pathlib import Path
import numpy as np
from astropy.table import Table, vstack
from astroquery.mast import ObservationsClass, Observations
from astroquery.exceptions import RemoteServiceError
import requests  # type:ignore
from .environment import PATH_DOWNLOAD

logger = getLogger(__name__)
subdir_nirspecifu = Path('JWST/NIRSpecIFU/')
__all__ = ['download_nirspecifu_rawdata', 'download_nirspecmsa_calibdata']


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
    nthread_p: int = 10,
    nthread_d: int = 20,
    chunksize_download: int = 50,
    overwrite: bool = False,
    savetag: str = '',
    skip_existdata: bool = True,  # Not used now
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
        nthread_p (int, optional): Number of threads to get a product list.
            Defaults to 10.
        nthread_d (int, optional): Number of threads to download data. Defaults to 20.
        chunksize_download (int, optional): The chunksize used for deviding
            the product list. Defaults to 50.
        overwrite (bool, optional): If True, the data will be overwrite. Otherwise,
            the exisiting data will not be overwriten by the downloaded data and
            warning will be reported. Defaults to False.
        savetag (str, optional): Tag used to modify the save directory.
        skip_existdata (bool, optional): If True, the data already downloaded will be
            skiped. Defaults to True.
        kwargs (optional): Criteria to search observations.

    Returns:
        Table: Output observation table based on the input query.
               In addition, save downloaded files.

    Examples:
        >>> download_nirspecmsa_calibdata(proposal_id=1180)
    '''
    pool_maxsize = nthread_d + nthread_p + 2  # Needed for multi-thread downloads
    downloader = NIRSpecMSADownloader(savetag=savetag, pool_maxsize=pool_maxsize)

    if obs is None:
        downloader.query_criteria(**kwargs)
        if downloader.n_obs > nlimit:
            msg = (
                'Too many observations satisfying the given criteria'
                f'({downloader.n_obs} > {nlimit}) .'
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
        for i, _obs in enumerate(obs):
            _id, fil, name = _obs["proposal_id"], _obs["filters"], _obs["target_name"]
            logger.info(f'PID{_id}: {fil} -- {name}')
            if i > 50:
                logger.info('...')
                break
        logger.info(f'Number of all products in the first chunk: {len(product_list)}')
        for i, p in enumerate(product_list['productFilename']):
            if '.fits' in p:
                logger.info(f'{downloader.dpath_downloading / p}')
            if i > 100:
                logger.info('...')
                break
        logger.info('These data will be moved to appropriate directories later.')
        return obs

    # Ignore INFO level logger in manifest to prevent cache message.
    current_level = logging.getLogger("astroquery").getEffectiveLevel()
    logging.getLogger("astroquery").setLevel(logging.WARNING)

    # Main task
    logger.info(f'START: {datetime.today()}')
    with ThreadPoolExecutor(nthread_p) as exe1, ThreadPoolExecutor(nthread_d) as exe2:
        downloader.run_get_product_list(obslist, exe1)
        downloader.run_download_products(exe2)
        product_list, manifest = downloader.result()
        downloader.run_arrange_dirs()
    logger.info(f'END: {datetime.today()}')

    # Back to the default level
    logging.getLogger("astroquery").setLevel(current_level)

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
    subdir_tempdownload = 'tmp_download'
    dataRights = 'PUBLIC'
    instrument_name = 'NIRSpec/MSA'
    max_retries = 3

    def __init__(
        self, savetag: str = '', skip_existdata: bool = True, pool_maxsize: int = 0
    ) -> None:
        self.savetag = savetag + '_' if savetag else ''
        self.skip_existdata = skip_existdata
        self.count_thread: int = 0
        self.dpath = self.path_download / self.subdir_nirspecmsa
        self.dpath_downloading = self.dpath / self.subdir_tempdownload
        self.dpath.mkdir(exist_ok=True)
        self.dpath_downloading.mkdir(exist_ok=True)
        self.obs: Table | None = None
        self.product_list: Table | None = None
        self.manifest: Table | None = None
        self.queue_product: Queue = Queue()
        self.futures_product: list[Future] = []
        self.queue_manifest: Queue = Queue()
        self.futures_manifest: list[Future] = []

        self.mast = ObservationsClass()
        if pool_maxsize:
            self.modify_sessions(pool_maxsize, self.max_retries)
        # self.mast._parse_result = self._parse_result_modified

    def query_criteria(
        self, instrument_name=instrument_name, dataRights=dataRights, **kwargs
    ) -> None:
        '''Return results of observations query.'''
        self.obs = self.mast.query_criteria(
            instrument_name=instrument_name,
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
            future = exe.submit(self.get_product_list, obs, True)
            self.futures_product.append(future)

    def get_product_list(
        self, obs_chunk: Table, queue_up: bool = False, retries: int = max_retries
    ) -> Table:
        '''Wrapper of Observations.get_product_list()'''
        tries = 1
        while tries <= retries:
            tries += 1
            try:
                product = self.mast.get_product_list(obs_chunk)
            except (TimeoutError, RemoteServiceError) as e:
                if tries > retries:
                    logger.exception(
                        f'Number of tries has exceeeded the max retries ({retries}) '
                        f'in {current_thread().name}.'
                    )
                    raise e
                logger.info(
                    'Retry to Create product list in ' f'{current_thread().name}'
                )

        count_thread = self.count_thread
        self.count_thread += 1
        logger.info(
            'Created product list in ' f'{current_thread().name} #{count_thread}'
        )

        L2c = product['calib_level'] == 3
        excude_csvandasn = product['productType'] != 'INFO'
        product = product[np.where(L2c & excude_csvandasn)]
        if queue_up:
            self.queue_product.put((count_thread, product))
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
        count_thread, product_list = self.queue_product.get()
        # if self.skip_existdata:
        #     product_list = self.check_exist(product_list)

        manifest = self.mast.download_products(
            product_list,
            download_dir=self.dpath_downloading,
            flat=True,
            verbose=False,
        )

        logger.info(f'Downloaded products in {current_thread().name} #{count_thread}')
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
        # elif path.suffix == '.json':
        #     subdir = 'cal'
        # elif path.suffix == '.csv':
        #     subdir = 'cal'
        if path.exists():
            dname = self.savetag + target_name
            newdpath = self.dpath / dname / filtername / subdir
            newdpath.mkdir(parents=True, exist_ok=True)
            newpath = newdpath / path.name
            if newpath.exists():
                logger.warning(
                    f'The downloaded data, {newpath}, already exists. '
                    f'The downloaded data was not be moved from {newpath}.'
                )
            else:
                path.rename(newpath)

    def check_exist(self, observations: Table) -> bool:
        '''Check whether the data exists.'''
        obs = observations[0]
        fname = Path(obs['dataURL'][5:]).name
        target_name = obs['target_name']
        dname = self.savetag + target_name
        filtername = ''.join(obs['filters'][0].split(';'))
        subdir = 'product'
        newdpath = self.dpath / dname / filtername / subdir / fname
        return newdpath.exists()

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

    def _change_poolsize(self, maxsize: int) -> None:
        '''Change DEFAULT_POOLSIZE in the request package.

        Default pool_maxsize is 10, but this is fewer than expected.
        This value should be as many as the number of threads used in downloads.

        NOTE:
            This is deprecated because it's meaningless to change global variables
            after defining the class that uses the variables as default arguments.
        '''
        if not isinstance(maxsize, int):
            raise TypeError(f'Maxsize ({maxsize}) must be int.')
        requests.adapters.DEFAULT_POOLSIZE = maxsize
        # This is Omake.
        requests.adapters.DEFAULT_RETRIES = 3

    def modify_sessions(self, maxsize: int, max_retries: int) -> None:
        self.mast._session.adapters['https://']._pool_maxsize = maxsize
        self.mast._session.adapters['https://']._pool_connections = maxsize
        self.mast._session.adapters['https://'].max_retries = max_retries
        self.mast._session.adapters['https://'].init_poolmanager(
            maxsize, maxsize, max_retries
        )
        self.mast._session.adapters['http://']._maxsize = maxsize
        self.mast._session.adapters['http://']._pool_connections = maxsize
        self.mast._session.adapters['http://'].max_retries = max_retries
        self.mast._session.adapters['http://'].init_poolmanager(
            maxsize, maxsize, max_retries
        )

    # def _parse_result_modified(self, responses, *, verbose=False) -> Table:
    #     '''Same as _portal_api_connection._parse_result'''
    #     connection = self.mast._portal_api_connection
    #     result_list = []

    #     # loading the columns config
    #     col_config = None
    #     if connection._current_service:
    #         col_config = connection._column_configs.get(connection._current_service)
    #         connection._current_service = None  # clearing current service

    #     for resp in responses:
    #         result = resp.json()

    #         # check for error message
    #         if result['status'] == "ERROR":
    #             raise RemoteServiceError(
    #                 result.get('msg', "There was an error with your request.")
    #             )

    #         result_table = _json_to_table(result, col_config)
    #         result_list.append(result_table)

    #     all_results = vstack(result_list)

    #     # Check for no results
    #     if not all_results:
    #         logger.warning("Query returned no results.")
    #     return all_results
