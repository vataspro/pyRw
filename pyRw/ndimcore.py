"""
pyRw/core.py

Core functions for computation of the
n-dimensional parameter multiple histogram reweighting.

Where optimal numba jit compilation and
vectorisation has been implemented.

"""

import numpy as np
from numba import njit, guvectorize


# Calculation of logsumexp for a 1D array
@njit
def logsumexp1d(a):
    """
    Numerically stable calculation of
    log(e^a0 + e^a1 + ... + e^an)

    Inputs:
        a   :   numpy.ndarray
            Set of numbers to apply the logsumexp function to.

    Returns:
        float

    References:
        [1] https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    megisto = np.max(a)
    exp_sum = 0.0
    for i in range(a.shape[0]):
        exp_sum += np.exp(a[i] - megisto)
    return megisto + np.log(exp_sum)


# Single histogram reweighting
@guvectorize(
        "float64[:,:], float64[:], float64[:], float64[:,:], float64[:]", 
        "(m,l),(m),(l),(o,l)->(o)"
)
# m the total number of measurements
# l the number of parameters / observables
# o the number of target points
def ndSingleHistogramReweight(J, Q, kappa_0, kappa, newQ):
    for k in range(kappa.shape[0]):
        num = 0.0
        den = 0.0
        d_kappa = kappa[k] - kappa_0
        for i in range(J.shape[0]):
            w = np.exp(np.sum(d_kappa * J[i]))
            num += Q[i] * w
            den += w

        newQ[k] = num / den


# Single histogram reweighting with logsumexp
@guvectorize(
        "float64[:,:], float64[:], float64[:], float64[:,:], int32, float64[:]",
         "(m,l),(m),(l),(o,l),()->(o)",
)
# m number of measurements
# l the number of parameters
# o the number of target points
def ndSingleHistogramReweightLogsumexp(J, Q, kappa_0, kappa, n, newQ):
    """
    Single histogram reweighting: interpolate the
    value of the observable Q, using the measured values
    of E and Q sampled at beta_0 at the target beta values.

    Inputs:
        E : numpy.ndarray [m]
            The measured values of the energy (or Euclidean action).
        Q : numpy.ndarray [m]
            The measured values of the observable to interpolate.
        beta_0 : float
            The value of beta at the source ensemble.
        beta: numpy.ndarray [k]
            The target beta values to interpolate at.

    Outputs:
        newQ : numpy.ndarray [k]
            The interpolated values of the observable at the
            target beta values.

    """
    for k in range(kappa.shape[0]):
        num = 0.0
        den = 0.0
        d_kappa = kappa[k] - kappa_0
        mask = Q > 0.0
        num = logsumexp1d(n * np.log(Q[mask]) + np.sum(d_kappa * J[mask], axis=1))
        den = logsumexp1d(np.sum(d_kappa * J[mask], axis=1))

        newQ[k] = np.exp(num - den)

