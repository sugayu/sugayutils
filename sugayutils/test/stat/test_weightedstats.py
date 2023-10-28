import numpy as np
from ...stat import weightedstats as wstats


x = np.arange(51)
sigma = 4.0
gauss = np.exp(-0.5 * (x - 25.0) ** 2 / sigma**2)


def test_moment():
    assert np.isclose(wstats.moment(x, gauss, order=1), 0.0)


def test_variance():
    assert np.isclose(wstats.variance(x, gauss), sigma**2)


def test_std():
    assert np.isclose(wstats.std(x, gauss), sigma)


def test_skewness():
    assert np.isclose(wstats.skewness(x, gauss), 0.0)


def test_kurtosis():
    assert np.isclose(wstats.kurtosis(x, gauss), 0.0)


def test_clipped_index1d():
    assert np.count_nonzero(wstats._clipped_index1d(x, gauss, 3.1)) == 6 * sigma + 1
