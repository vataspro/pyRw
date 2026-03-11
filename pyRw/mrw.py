"""
pyRw/mrw.py

Implements multiple reweighting wrapper class.

It is advised to use the package via the MultiRw
interface class.

"""

import numpy as np
import pyRw.autocorr
import pyRw.core
import pyRw.utils
from copy import deepcopy


class MultiRw:
    """

    Multiple Histogram Reweighting wrapper class.

    In order to avoid common pitfalls in
    the core vectorised and jit compiled functions
    of the package this safe wrapper class is the
    suggested inerface for pyRw.


    Arguments:
        betas    :  list or 1d np.ndarray
            The values of β at which the ensembles have been sampled.
        E        :  list of lists or list of 1d arrays or 2d array
            The observed values of the energy (or Euclidean action)
            at each beta.
        logZ     :  1d list or np.ndarray
            Set value of logZ -- if not set it is calculated iteratively.
        autocorr :  bool
            Calculate autocorrelation and implement binning on the
            measured data. The autocorrelation time is that of the
            action or energy.

    """

    def __init__(
        self,
        betas,
        E,
        logZ=None,
        autocorr=False,
        verbose=True,
        max_iter=100000,
        tol=1e-10,
    ):
        # Guard
        pyRw.utils.ensureValidObservableShape(E)
        if len(E) != len(betas):
            raise ValueError("Provided betas and E dimension mismatch.")

        self.autocorr = autocorr
        self.verbose = verbose

        # Autocorrelation
        if self.autocorr:
            self.texp = [pyRw.autocorr.integrated_autocorrelation_time(x) for x in E]
            self.nskips = np.ceil(self.texp).astype(int)
            self.E = pyRw.utils.binObservable(E, self.nskips)
        else:
            self.E = E

        # Initialise variables
        self.betas = np.array(betas)
        self.logN = np.array([np.log(len(e)) for e in self.E])

        # LogZ computation or loading
        if logZ is None:
            self.logZ = pyRw.core.itersolve(
                self.logN,
                self.betas,
                np.concatenate(self.E),
                verbose=self.verbose,
                max_iter=max_iter,
                tol=tol,
            )
        else:
            if not len(betas) == len(logZ):
                raise ValueError("logZ and betas arrays do not have the same length.")
            self.logZ = np.array(logZ)

    def reweight(self, Q, beta, n=1):
        """
        Obtain the reweighted value of the n-th moment
        of the observable Q at β values beta.

        Inputs:
            Q   :   list of lists or list of 1d arrays or 2d array
                The observed values of the obaservable Qat each beta.
                Should have the same shape as the energy (or Euclidean
                action) E.
            n   :   float
                The order of the moment to return. The standard value
                n=1 returns the expectation value of Q at beta.

        Returns:
            q   :   1d np.ndarray
                The n-th moment of the observable Q <Q^n> calculated at
                β value(s) b.

        """

        # Setup
        pyRw.utils.ensureValidObservableShape(Q)
        pyRw.utils.checkObservableNotNegative(Q)
        if self.autocorr:
            Q_ = pyRw.utils.binObservable(Q, self.nskips)
        else:
            Q_ = deepcopy(Q)
        pyRw.utils.ensureSameShape2d(self.E, Q_)

        # Reweight
        b = np.array(beta)
        q = np.empty_like(beta)
        q = pyRw.core.getQn(
            self.logZ,
            self.logN,
            self.betas,
            b,
            np.concatenate(self.E),
            np.concatenate(Q_),
            n,
            q,
        )

        return q


"""
Run the SimpleRw program to boostrap and reweight
an observable and its susceptibility.

    Inputs:
        betas   :   1d list
                The beta values of the ensembles
        action  :   2d list
                The action measurements at each beta value.
        observable : 2d list
                Observable measurements. Same shape as action.
        target_betas : 1d np.ndarray
                Target beta values to reweight at.
        num_bootstraps : int
                Number of bootstrap samples used in the calculation.
        volume  :   int
                Lattice volume Nt*Nx*Ny*Nz
        tau     :   1d list
                Integrated autocorrelation time for each ensemble.

    Returns:
        mean_obs :  1d np.ndarray
                mean value of the observable
        error_obs :  1d np.ndarray
                bootstrap error of the observable
        mean_susc :  1d np.ndarray
                mean value of the susceptibility
        error_susc :  1d np.ndarray
                bootstrap error of the susceptibility
"""


def SimpleRw(
    betas,
    action,
    observable,
    target_betas,
    num_bootstraps,
    volume,
    tau=None,
    verbose=False,
):
    # Calculate autocorrelation times
    if tau is None:
        tau = [
            pyRw.autocorr.integrated_autocorrelation_time(observable[i])
            for i in range(len(betas))
        ]

    # Resize samples for autocorrelation
    bs_sizes = [len(action[i]) // (2 * int(np.ceil(tau[i]))) for i in range(len(betas))]

    # Bootstrap observable and susceptibility
    bs_samples_susc = []
    bs_samples_obs = []
    for _ in range(num_bootstraps):
        # Aggregate sample action and observable measurements
        action_ = []
        observable_ = []
        for i in range(len(betas)):
            bs_idx = np.random.randint(0, len(action[i]), bs_sizes[i])
            action_.append(np.array(action[i][bs_idx]))
            observable_.append(np.array(observable[i][bs_idx]))

        # Reweight
        mrw = MultiRw(betas, action_, verbose=verbose)
        x = mrw.reweight(observable_, target_betas)
        x2 = mrw.reweight(observable_, target_betas, n=2)

        # Collect results
        bs_samples_susc.append(volume * (x2 - x**2))
        bs_samples_obs.append(x)

    # Calculate bootstrap mean and error
    mean_obs = np.mean(bs_samples_obs, axis=0)
    error_obs = np.std(bs_samples_obs, axis=0)
    mean_susc = np.mean(bs_samples_susc, axis=0)
    error_susc = np.std(bs_samples_susc, axis=0)

    return mean_obs, error_obs, mean_susc, error_susc
