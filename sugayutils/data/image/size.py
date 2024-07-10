'''Size analyses.
'''
import numpy as np


##
class FitImage(abc.ABC):
    '''Fitting Image.'''

    def __init__(
        self,
        image: np.ndarray,
        grids: tuple[np.ndarray, np.ndarray] | None = None,
        upsampling_rate: int = 3,
        psf_upsampling_rate: int = 1,
        psf: np.ndarray | None = None,
        filt: str = '',
    ) -> None:
        self.image = image
        self.shape = image.shape
        self.shape_psf_upsampling = tuple([s * psf_upsampling_rate for s in self.shape])
        self.error = np.nanstd(image) * 2.0
        self.upsampling_rate = upsampling_rate
        self.psf_upsampling_rate = psf_upsampling_rate
        if grids is None:
            sampling_rate = upsampling_rate * psf_upsampling_rate
            yy = self.gridding_upsample(np.arange(self.shape[0]), sampling_rate)
            xx = self.gridding_upsample(np.arange(self.shape[1]), sampling_rate)
            self.x, self.y = np.meshgrid(xx, yy)
        else:
            self.x, self.y = grids
        self.psf = psf
        self.model = self.define_model()
        self.bound0, self.bound1 = self.get_bounds()
        self.nwalkers = 12
        self.ndim = len(self.bound0)
        self.nsteps = 3000
        self.progressbar = False

        self._filter = filt

    def run_fitting(self) -> None:
        '''Run fitting.'''
        init = self.initialize_param()

        rng = default_rng(222)
        norm = rng.standard_normal((self.nwalkers, self.ndim))
        init = init + init * 0.0001 * norm

        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.calculate_log_probability,
        )
        sampler.run_mcmc(init, self.nsteps, progress=self.progressbar)
        self.sampler = sampler

    def calculate_log_probability(self, params):
        '''Calculate log probability of model(theta; data).'''
        log_prior = self.calculate_log_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        log_likelihood = self.calcurate_log_likelihood(params)
        return log_prior + log_likelihood

    def calculate_log_prior(self, params):
        '''Calcurate values from the prior probability distribution in log'''
        _params = np.array(params)
        if np.all(self.bound0 < _params) and np.all(_params < self.bound1):
            return 0.0
        return -np.inf

    def calcurate_log_likelihood(self, params):
        '''Calculate log likelihood'''
        model = self.model(params)
        chi = (self.image - model) / self.error

        return -0.5 * np.nansum(abs(chi) ** 2)

    @abc.abstractmethod
    def define_model(self):
        '''Define model.'''
        pass

    @abc.abstractmethod
    def get_bounds(self):
        '''Return bounding parameters.'''
        pass

    @abc.abstractmethod
    def initialize_param(self):
        '''Estimate initial parameters.'''
        pass

    @staticmethod
    def down_sampling(image: np.ndarray, shape_to: tuple[int, ...]) -> np.ndarray:
        '''Down-sampling of a data cube, moved from Tokult.

        This is to reconstruct the image more-finely-resampled when the sampling
        rate is not sufficient.

        Args:
            image (np.ndarray): 2d dimage.
            shape_to (tuple[int, int]): Shape of the resampled cube.

        Returns:
            np.ndarray: Resampled data cube.
        '''
        shape_from = image.shape
        if shape_from == shape_to:
            return image
        nbins = [f // t for f, t in zip(shape_from, shape_to)]
        image_out = image.reshape(shape_to[0], nbins[0], shape_to[1], nbins[1])
        return image_out.mean(axis=(1, 3))

    @staticmethod
    def gridding_upsample(grid: np.ndarray, rate_upsampling: int) -> np.ndarray:
        '''Make grids up-sampling, moved from Tokult.

        Args:
            grid (np.ndarray):
            rate_upsampling (int):
        '''
        if rate_upsampling == 1:
            return grid
        nbins_to = len(grid) * rate_upsampling
        return np.linspace(grid[0] - 0.5, grid[-1] + 0.5, (nbins_to) * 2 + 1)[1:-1:2]


class FitGauss2d(FitImage):
    '''2D Gaussian Fitting.'''

    def define_model(self) -> Callable:
        return self.model_gauss

    def model_gauss(self, params) -> np.ndarray:
        '''Fit Gaussian model.'''
        cont, flux, center_y, center_x, radius = params
        r = radius
        gauss = self.gauss2d(self.x, self.y, flux, (center_y, center_x), r)

        gauss = self.down_sampling(gauss, self.shape_psf_upsampling)
        if self.psf is not None:
            psf = self.psf[1:, 1:] if self.psf.shape[0] % 2 == 1 else self.psf
            gauss = convolve(gauss, psf, mode='same')
        gauss = self.down_sampling(gauss, self.shape)

        return gauss + cont

    @staticmethod
    def gauss2d(
        x: np.ndarray,
        y: np.ndarray,
        flux: float,
        center: tuple[float, float],
        radius: float,
    ) -> np.ndarray:
        '''2d Gauss function.'''
        peak = flux / (2 * np.pi * radius**2)
        exp_y = np.exp(-0.5 * ((y - center[0]) / radius) ** 2)
        exp_x = np.exp(-0.5 * ((x - center[1]) / radius) ** 2)

        # This is patch for flux conservation.
        gauss = peak * exp_x * exp_y
        if radius < (dx := x[0, 1] - x[0, 0]) / 2:
            dy = y[1, 0] - y[0, 0]
            idx = gauss == gauss.max()
            max_pix_value = flux / dx / dy - gauss[~idx].sum()
            gauss[idx] = max_pix_value
        return gauss

    def get_bounds(self):
        '''Return bounding parameters.'''
        bound0 = np.array([-1.0, 0.0, 10.0, 10.0, 0.0])
        bound1 = np.array([1.0, 100.0, 20.0, 20.0, 3.0])
        return bound0, bound1

    def initialize_param(self):
        '''Estimate initial parameters.

        ----
        cont: [MJy(/Sr)]
        flux: [MJy(/Sr)] (integrated unit of image)
        center_y: [pix]
        center_x: [pix]
        radius: [pix]
        '''
        cont = 1e-5
        center_y = np.mean(self.y)
        center_x = np.mean(self.x)
        radius = 2.0
        flux = np.nanmax(self.image) * (2 * np.pi * radius**2)
        return np.array([cont, flux, center_y, center_x, radius])


class FitExponential2d(FitImage):
    '''Exponential fitting'''

    n = 1  # Sersic index
    b_n = gammaincinv(2.0 * n, 0.5)

    def define_model(self) -> Callable:
        return self.model_exponential

    def model_exponential(self, params) -> np.ndarray:
        '''Fit Exponential model.'''
        cont, flux, center_y, center_x, r_e = params
        r = r_e
        fluxmap = self.exponential(self.x, self.y, flux, (center_y, center_x), r)

        fluxmap = self.down_sampling(fluxmap, self.shape_psf_upsampling)
        if self.psf is not None:
            psf = self.psf[1:, 1:] if self.psf.shape[0] % 2 == 1 else self.psf
            fluxmap = convolve(fluxmap, psf, mode='same')
        fluxmap = self.down_sampling(fluxmap, self.shape)

        return fluxmap + cont

    def exponential(
        self,
        x: np.ndarray,
        y: np.ndarray,
        flux: float,
        center: tuple[float, float],
        r_e: float,
    ) -> np.ndarray:
        '''2d exponential function.'''
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        I_e = flux / 3.0773 / r_e**2  # numerical solution
        exp = I_e * np.exp(-self.b_n * ((r / r_e) - 1))

        # This is patch for flux conservation.
        if r_e < (dx := x[0, 1] - x[0, 0]) / 2:
            dy = y[1, 0] - y[0, 0]
            idx = exp == exp.max()
            max_pix_value = flux / dx / dy - exp[~idx].sum()
            exp[idx] = max_pix_value
        return exp

    def get_bounds(self):
        '''Return bounding parameters.'''
        bound0 = np.array([-1.0, 0.0, 10.0, 10.0, 0.0])
        bound1 = np.array([1.0, 100.0, 20.0, 20.0, 9.0])
        return bound0, bound1

    def initialize_param(self):
        '''Estimate initial parameters.

        ----
        cont: [MJy(/Sr)]
        flux: [MJy(/Sr)] (integrated unit of image)
        center_y: [pix]
        center_x: [pix]
        radius: [pix]
        '''
        cont = 1e-5
        center_y = np.mean(self.y)
        center_x = np.mean(self.x)
        radius = 2.0
        flux = np.nanmax(self.image) * (2 * np.pi * radius**2)
        return np.array([cont, flux, center_y, center_x, radius])
