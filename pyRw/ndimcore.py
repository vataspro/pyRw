"""
pyRw/ndimcore.py

Core functions for computation of the
n-dimensional parameter multiple histogram reweighting.

Where optimal numba jit compilation and
vectorisation has been implemented.

"""

import numpy as np
from pyRw.core import logsumexp1d
from numba import njit, guvectorize


# Naive single histogram reweighting
@guvectorize(
    "float64[:,:], float64[:], float64[:], float64[:,:], float64[:]",
    "(m,l),(m),(l),(o,l)->(o)",
)
def ndSingleHistogramReweight(J, Q, kappa_0, kappa, newQ):
    """
        Naive single histogram reweighting without the use
        of logsumexp.
        May fail if the actions are different orders of magnitude!


    Inputs:

        J : list of 1d arrays (m, l)
            The conjugate observables contributing to the action.

        Q : 1d array (m)
            The measured values of the observable.

        kappa_0 : 1d array (l)
            The value of the vector parameter at the source ensemble.

        kappa : 2d array (o, l)
            Values of the parameter to interpolate at.


    Dimensions:
        m : total number of measurements
        l : parameter dimension
        o : number of target points

    Outputs :
        newQ : 1d array (o)
            Interpolated value of the observable.

        newQ :

    """

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
            c[i] = -logsumexp1d(
                logN - logZ - np.sum((kappa[k] - kappa0) * J[i], axis=1)
            )
        newLogZ[k] = logsumexp1d(c)


# TODO: This code is essentially duplicating what is done in pyRw.core
# Replace with a generalised itersolve function, which will take a specific iterfn
# argument
def ndItersolve(logN, kappa0, J, tol=1e-10, max_iter=50000, verbose=True):
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


# Faster kernel if all values are positive
@njit
# @guvectorize(
#        "float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:], int32, float64[:]",
#        "(m),(m),(m,q),(k,q),(l,q),(l),(),(k)")
## m number of source ensembles
## q dimensionality of parameter
## l number of measurements
def ndGetQn_(logZ, logN, kappa0, kappa, J, Q, n, newQ):
    newLogZ = np.empty_like(newQ)

    Qsafe = np.empty_like(Q)  # probably want to mask here
    Qsafe[:] = np.where(Q == 0, 1e-10, Q)

    for k in range(kappa.shape[0]):
        c = np.empty(J.shape[0])
        for i in range(J.shape[0]):
            c[i] = -logsumexp1d(
                logN - logZ - np.sum((kappa[k] - kappa0) * J[i], axis=1)
            )
        newLogZ[k] = logsumexp1d(c)
        newQ[k] = logsumexp1d(n * np.log(Qsafe) + c) - newLogZ[k]

        newQ[k] = np.exp(newQ[k])


# Slower kernel that can deal with negative values
@njit
def ndGetQn(logZ, logN, kappa0, kappa, J, Q, n, newQ):
    Nk = kappa.shape[0]  # target values
    Ncfg = Q.shape[0]

    # Mask and index
    mask_p = Q > 0.0
    mask_m = Q < 0.0  # ~mask_p

    idx_p = np.where(mask_p)[0]  # positive obs values
    idx_m = np.where(mask_m)[0]

    Np = idx_p.shape[0]
    Nm = idx_m.shape[0]

    for k in range(Nk):  # for each target kappa
        dk = kappa[k] - kappa0

        # Compute weight for each configuration
        c = np.empty(Ncfg)
        for i in range(Ncfg):
            # energy shift per histogram
            shift = np.sum(dk * J[i], axis=1)
            c[i] = -logsumexp1d(logN - logZ - shift)

        # Sector partition functions
        c_p = c[idx_p]
        c_m = c[idx_m]

        logZp = logsumexp1d(c_p) if Np > 0 else -np.inf
        logZm = logsumexp1d(c_m) if Nm > 0 else -np.inf

        # Total Z
        if logZp > logZm:
            logZtot = logZp + np.log1p(np.exp(logZm - logZp))
        else:
            logZtot = logZm + np.log1p(np.exp(logZp - logZm))

        # Sector expectation values
        # positive sector
        if Np > 0:
            tmp_p = np.empty(Np)
            for i in range(Np):
                idx = idx_p[i]
                tmp_p[i] = n * np.log(Q[idx]) + c[idx]
            logEp = logsumexp1d(tmp_p) - logZp
        else:
            logEp = -np.inf

        # negative sector
        if Nm > 0:
            tmp_m = np.empty(Nm)
            for i in range(Nm):
                idx = idx_m[i]
                tmp_m[i] = n * np.log(-Q[idx]) + c[idx]
            logEm = logsumexp1d(tmp_m) - logZm
        else:
            logEm = -np.inf

        # Recombine with weights
        # numerator = Zp * Ep - Zm * Em
        term_p = logZp + logEp
        term_m = logZm + logEm

        # stable signed subtraction
        if term_p > term_m:
            num = np.exp(term_p) * (1.0 - np.exp(term_m - term_p))
        else:
            num = -np.exp(term_m) * (1.0 - np.exp(term_p - term_m))

        newQ[k] = num / np.exp(logZtot)
