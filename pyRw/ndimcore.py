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
    Single histogram reweighting with logsumexp: interpolate the
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


@guvectorize(
        "float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:]",
    "(m),(m),(k,q),(m,q),(l,q)->(k)",
)
# m number source ensembles
# l number of measurements
# k number of target kappas
# q dimensionality of parameter space
def ndGetLogZ(logZ, logN, kappa, kappa0, J, newLogZ):
    for k in range(kappa.shape[0]):
        c = np.empty(J.shape[0])
        for i in range(J.shape[0]):
            c[i] = -logsumexp1d(logN - logZ - np.sum((kappa[k] - kappa0) * J[i], axis=1))
        newLogZ[k] = logsumexp1d(c)


def ndItersolve(logN, kappa0, J, tol=1e-10, max_iter=50000, verbose=True):

    #if not (len(logN) == len(betas)):
    #    raise ValueError("logN and betas should have the same length")

    f = np.zeros(kappa0.shape[0])

    for i in range(max_iter):
        f_old = np.copy(f)
        ndGetLogZ(f_old, logN, kappa0, kappa0, J, f)

        # As the free energy is defined up to a constant,
        # some value has to be fixed,
        # otherwise the solver does not coverge,
        # exhibiting 'drifting' behaviour.
        # This can be seen as a 'gauge fixing'
        f -= np.mean(f)  # Could also fix with 'f -= f[0] - 1'

        # Check convergence up to gauge
        delta = f - f_old
        delta -= np.mean(delta)
        err = np.linalg.norm(delta)

        if verbose:
            print(f"Iter {i}: error = {err:.6e}")

        if err < tol:
            break
    else:
        raise RuntimeError("Iteration did not converge")

    return f


#@njit
@guvectorize(
        "float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:], int32, float64[:]",
        "(m),(m),(m,q),(k,q),(l,q),(l),(),(k)")
# m number of source ensembles
# q dimensionality of parameter
# l number of measurements
def ndGetQn(logZ, logN, kappa0, kappa, J, Q, n, newQ):
    newLogZ = np.empty_like(newQ)

    Qsafe = np.empty_like(Q) # probably want to mask here
    Qsafe[:] = np.where(Q == 0, 1e-10, Q)

    for k in range(kappa.shape[0]):
        c = np.empty(J.shape[0])
        for i in range(J.shape[0]):
            c[i] = -logsumexp1d(logN - logZ - np.sum((kappa[k] - kappa0) * J[i], axis=1))
        newLogZ[k] = logsumexp1d(c)
        newQ[k] = logsumexp1d(n * np.log(Qsafe) + c) - newLogZ[k]

        newQ[k] = np.exp(newQ[k])

