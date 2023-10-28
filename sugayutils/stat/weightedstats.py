'''Weighted statistics, useful for moment calcurations of emission lines.
'''
from __future__ import annotations
import numpy as np
import astropy.units as u


##
def moment(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    order: int,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted n-th moment about the weighted average.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        order (int): Order of the moment.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | np.ndarray | u.Quantity: Weighted n-th moment.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> mom2 = weightedstats.moment(wavelength, weights=flux, order=2)
    '''
    if sigma is not None:
        idx = _clipped_index1d(values, weights, sigma)
        _values, _weights = values[idx], weights[idx]
    else:
        _values, _weights = values, weights

    average = _average(_values, weights=_weights, axis=axis)
    return np.average((_values - average) ** order, weights=_weights, axis=axis)


def variance(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted variance.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | np.ndarray | u.Quantity: Weighted variance.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> variance = weightedstats.variance(wavelength, weights=flux)
    '''
    return moment(values, weights, order=2, axis=axis, sigma=sigma)


def std(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted standard deviation.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | np.ndarray | u.Quantity: Weighted standard deviation.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> sigma = weightedstats.std(wavelength, weights=flux)
    '''
    return np.sqrt(variance(values, weights, axis=axis, sigma=sigma))


def gauss_fwhm(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted fwhm computed from standard deviation assuming Gaussian.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | np.ndarray | u.Quantity: Weighted fwhm.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> fwhm = weightedstats.gauss_fwhm(wavelength, weights=flux)
    '''
    return (
        2.0 * np.sqrt(2.0 * np.log(2.0)) * std(values, weights, axis=axis, sigma=sigma)
    )


def skewness(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted skewness.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | u.Quantity: Weighted skewness.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> skewness = weightedstats.skewness(wavelength, weights=flux)
    '''
    norm = std(values, weights, axis=axis) ** 3
    mom3 = moment(values, weights, order=3, axis=axis, sigma=sigma)
    return mom3 / norm


def kurtosis(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
    sigma: float | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Weighted kurtosis.

    This function adopts the denifition that the Gauss function has a kurtosis of zero,
    by subtracting 3 from normalized 4th moments: E(X-mu)/sigma**4 - 3

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.
        sigma (float | None): Size of sigma clipping. This works only for 1d input values.
            Defaults to None.

    Returns:
        float | u.Quantity: Weighted kurtosis.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> kurtosis = weightedstats.skewness(wavelength, weights=flux)
    '''
    norm = variance(values, weights, axis=axis) ** 2
    mom4 = moment(values, weights, order=4, axis=axis, sigma=sigma)
    return mom4 / norm - 3.0


def _average(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> float | np.ndarray | u.Quantity:
    '''Average wrapper to conserve dimenstions of the array if axis is specified.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.

    Returns:
        np.ndarray | u.Quantity: Agerage array.
    '''
    average = np.average(values, weights=weights, axis=axis)
    if axis is not None:
        _axis = (axis,) if isinstance(axis, int) else axis
        shape_averaged = (1 if i in _axis else s for i, s in enumerate(values.shape))
        average = average.reshape(tuple(shape_averaged))
    return average


def _clipped_index1d(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    sigma: float | np.ndarray | u.Quantity,
) -> np.ndarray:
    '''Return index after sigma clipping.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        sigma (float | np.ndarray | u.Quantity): Number of standard deviation
            for sigma clipping.

    Returns:
        np.ndarray : Clipped bool index. This array is 1d.
    '''
    index0 = np.zeros_like(values).astype(bool)
    index1 = np.ones_like(values).astype(bool)

    while not np.all(index0 == index1):
        index0 = index1
        avg = _average(values[index0], weights[index0])
        _std = std(values[index0], weights[index0])
        index1 = np.abs(values - avg) < sigma * _std

    return index1


# Write a test!!
