"""
pyRw/autocorr.py

Utility functions for calculating
the integrated autocorrelation time.

"""

import numpy as np
from numba import njit


@njit
def autocorrelation(x, max_lag=None):
    """
    Compute the normalized autocorrelation function of a 1D array.

    Inputs:
        x   :   list or numpy.ndarray
            A 1d numpy array to calculate the autocorrelation of.

        max_lag   :   int
            Maximum lag in overlapping copies of x

    Returns:
        acf     :    numpy.ndarray
            The autocorrelation function normalised appropriately
            for the estimation of the integrated autocorrelation time,
            see ref [1].

    References:
        [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    x = np.asarray(x).flatten()  # Ensure 1D
    x -= np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    acf = np.correlate(x, x, mode="full")[n - 1 : n + max_lag]
    acf /= np.arange(n, n - max_lag - 1, -1)  # unbiased normalization
    acf /= acf[0]  # dividing by the variance to normalize to ACF[0] == 1
    return acf


@njit
def integrated_autocorrelation_time(x, max_lag=None):
    """
    Estimate the integrated autocorrelation time of a 1D array.
    Uses simple windowing: stop summing when ACF becomes negative.

    Inputs:
        x   :   list or numpy.ndarray

    Returns:
        tau_int :   float
            Estimate of the integrated autocorrelation time, according
            to ref [1].

    References:
        [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation
        [2] http://stackoverflow.com/q/14297012/190597
    """
    x = np.asarray(x).flatten()
    acf = autocorrelation(x, max_lag)
    acf_tail = acf[1:]  # exclude lag-0

    W = 0
    for i in range(len(acf_tail)):
        if acf_tail[i] < 0:
            break
        W = i + 1

    tau_int = 1 + 2 * np.sum(acf_tail[:W])
    return tau_int
