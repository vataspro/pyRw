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

    def __init__(self, betas, E, logZ=None, autocorr=False, verbose=True):
        # Guard
        pyRw.utils.ensureValidObservable(E)
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
        self.logN = np.array([len(e) for e in E])

        # LogZ computation or loading
        if logZ is None:
            self.logZ = pyRw.core.itersolve(
                self.logN, self.betas, np.concatenate(self.E), verbose=self.verbose
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
        pyRw.utils.ensureValidObservable(Q)
        pyRw.utils.checkObservableNotNegative(Q)
        if self.autocorr:
            Q_ = pyRw.utils.binObservable(Q, self.nskips)
        else:
            Q_ = np.copy(Q)
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
