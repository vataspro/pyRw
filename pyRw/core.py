"""
pyRw/core.py

Core functions for computation of the
multiple histogram reweighting.

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


@guvectorize(
    "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]",
    "(m),(m),(m),(k),(l)->(k)",
)
def getLogZ(logZ, logN, betas, beta, Es, newLogZ):
    """
    Given the values of logZ, where Z is the free energy,
    at (m) values of β (betas), interpolate the value of logZ at
    (k) points (beta). This requires measurements of the energy
    (or Euclidean action).

    Applying this iteratively can be used to numerically solve
    for the value of logZ for a given histogram.

    Inputs:
        logZ    :   numpy.ndarray   [m]
            The known values of logZ at β values betas.
        logN    :   numpy.ndarray   [m]
            The number of measurements at each β value
            in betas.
        betas   :   numpy.ndarray   [m]
            The values of β which the system has been sampled at.
        beta    :   numpy.ndarray   [k]
            The valuez of β to interpolate logZ at.
        Es      :   numpy.ndarray   [l]
            The measured values of the energy (or Euclidean action).
            The length [l] is the sum of the number of the measurements
            at each of the known β values in betas:
                l = Σ_{i=0}^{m-1} N_i
            Please ensure that this has been flattened correctly in
            agreement with the inputted betas and logN arrays.

    Outputs:
        newLogZ :   numpy.ndarray   [k]
            The interpolated values of

    References:
        [1] Newman, M. E. J. & Berkema, G.T., "Monte Carlo Methods in
            Statistical Physics", Chapter 8.
        [2] https://www.mv.helsinki.fi/home/rummukai/lectures/montecarlo_oulu/lectures/mc_notes4.pdf

    """
    for k in range(beta.shape[0]):
        c = np.empty_like(Es)
        for i in range(Es.shape[0]):
            c[i] = -logsumexp1d(logN - logZ + (beta[k] - betas) * Es[i])
        newLogZ[k] = logsumexp1d(c)


@guvectorize(
    "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64[:]",
    "(m),(m),(m),(k),(l),(l),()->(k)",
)
def getQn(logZ, logN, betas, beta, Es, Q, n, newQ):
    """
    Estimate the n-th moment <Q^n> of an observable at β values beta
    by using multiple histogram reweighting.

    Inputs:
        logZ    :   numpy.ndarray   [m]
            The known values of logZ at β values betas.
        logN    :   numpy.ndarray   [m]
            The number of measurements at each β value
            in betas.
        betas   :   numpy.ndarray   [m]
            The values of β which the system has been sampled at.
        beta    :   numpy.ndarray   [k]
            The valuez of β to interpolate logZ at.
        Es      :   numpy.ndarray   [l]
            The measured values of the energy (or Euclidean action).
        Qs      :   numpy.ndarray   [l]
            The measured values of the observable.
            The length [l] is the sum of the number of the measurements
            at each of the known β values in betas:
                l = Σ_{i=0}^{m-1} N_i
            Please ensure that this has been flattened correctly in
            agreement with the inputted betas and logN arrays. The indexing
            of the Es and Qs arrays should also be consistent with each other.
        n       :   float
            The order of the moment of the observable to calculate. For the
            mean value set n=1.

    Outputs:
        newQ    :   numpy.ndarray   [k]
            The values of the n-th order moment <Q^n> at the β values beta.


    References:
        [1] Newman, M. E. J. & Berkema, G.T., "Monte Carlo Methods in
        Statistical Physics", Chapter 8.
    """

    newLogZ = np.empty_like(newQ)

    Qsafe = np.empty_like(Q)
    Qsafe[:] = np.where(Q == 0, 1e-10, Q)

    for k in range(beta.shape[0]):
        c = np.empty_like(Es)
        for i in range(Es.shape[0]):
            c[i] = -logsumexp1d(logN - logZ + (beta[k] - betas) * Es[i])
        newLogZ[k] = logsumexp1d(c)
        newQ[k] = logsumexp1d(n * np.log(Qsafe) + c) - newLogZ[k]

        newQ[k] = np.exp(newQ[k])


def itersolve(logN, betas, E, tol=1e-10, max_iter=50000, verbose=True):
    """
    Iterative solve for logZ.

    Inputs:
        logN    :   1d list or numpy.ndarray
            The number of measurements at each β value
            in betas.
        betas   :   1d list or numpy.ndarray
            The values of β which the system has been sampled at.
        Es      :   numpy.ndarray
            The measured values of the energy (or Euclidean action).

    Returns:
        logZ    :   1d numpy.ndarray
            The solution for logZ from the iterative solver.
    """

    if not (len(logN) == len(betas)):
        raise ValueError("logN and betas should have the same length")

    f = np.zeros_like(betas)

    for i in range(max_iter):
        f_old = np.copy(f)
        getLogZ(f_old, logN, betas, betas, E, f)

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
