'''Weighted statistics, useful for moment calcurations of emission lines.
'''
from __future__ import annotations
import numpy as np
import astropy.units as u


##
def variance(
    values: np.ndarray | u.Quantity,
    weights: np.ndarray | u.Quantity,
    axis: int | tuple[int, ...] | None = None,
) -> float | u.Quantity:
    '''Weighted variance.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.

    Returns:
        float | u.Quantity: Weighted variance.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> variance = weightedstats.variance(wavelength, weights=flux)
    '''
    average = np.average(values, weights=weights, axis=axis)
    if axis is not None:
        _axis = (axis,) if isinstance(axis, int) else axis
        shape_averaged = (1 if i in _axis else s for i, s in enumerate(values.shape))
        average = average.reshape(tuple(shape_averaged))
    return np.average((values - average) ** 2, weights=weights, axis=axis)


def std(
    values: np.ndarray,
    weights: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
) -> float:
    '''Weighted standard deviation.

    Args:
        values (np.ndarray | u.Quantity): Input values, e.g., wavelengths.
        weights (np.ndarray | u.Quantity): Weights, e.g., flux.
        axis (int | tuple[int, ...] | None, optional): Axes along which to average values.
            Defaults to None.

    Returns:
        float | u.Quantity: Weighted standard deviation.

    Examples:
        >>> from sugayutils.stat import weightedstats
        >>> sigma = weightedstats.std(wavelength, weights=flux)
    '''
    return np.sqrt(variance(values, weights, axis=axis))


# Write a test!!
