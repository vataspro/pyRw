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
import pyRw.ndimcore
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

        # Check for negative observable values
        if np.all(np.concatenate(Q_) > 0.0) or ((n % 2) == 0):
            # positive observable
            q = pyRw.core.getQn(
                self.logZ,
                self.logN,
                self.betas,
                b,
                np.concatenate(self.E),
                np.abs(np.concatenate(Q_)),
                n,
                q,
            )
        else:
            if n != 1:
                raise ValueError(
                    "Cannot indirectly estimate odd higher moments of a non-postive valued observable"
                )
            # observable contains negative values
            q = pyRw.core.getQn_(
                self.logZ,
                self.logN,
                self.betas,
                b,
                np.concatenate(self.E),
                np.concatenate(Q_),
                q,
            )

        return q


def BootstrapRw(
    betas,
    action,
    observable,
    target_betas,
    num_bootstraps,
    Ns=[1],
    tau=None,
    verbose=False,
):
    """
    Run the BootstrapRw  program to boostrap and reweight
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
    # Calculate autocorrelation times
    if tau is None:
        tau = [
            pyRw.autocorr.integrated_autocorrelation_time(observable[i])
            for i in range(len(betas))
        ]

    # Resize samples for autocorrelation
    bs_sizes = [len(action[i]) // (2 * int(np.ceil(tau[i]))) for i in range(len(betas))]

    # Bootstrap observable moments
    # List of dictionaries, each list element
    # holding the moments
    target_values = []

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

        # Interpolate moments of observable
        target_values.append({})
        for n in Ns:
            target_values[-1][n] = mrw.reweight(observable_, target_betas, n=n)

    return target_values


class BootstrapRwSaver:
    def __init__(self, betas, action, num_bootstraps, tau=None, verbose=False):
        # List contains all instances of MultiRws
        self.mrw = []
        self.bs_idx = []

        self.betas = betas

        # Calculate autocorrelation times
        if tau is None:
            tau = [
                pyRw.autocorr.integrated_autocorrelation_time(action[i])
                for i in range(len(betas))
            ]

        # Resize samples for autocorrelation
        bs_sizes = [
            len(action[i]) // (2 * int(np.ceil(tau[i]))) for i in range(len(betas))
        ]

        # Get LogZ for every bs sample
        for _ in range(num_bootstraps):
            # Aggregate sample action and observable measurements
            action_ = []
            bs_idx_ = []
            # observable_ = []
            for i in range(len(betas)):
                bs_idx_.append(np.random.randint(0, len(action[i]), bs_sizes[i]))
                action_.append(np.array(action[i][bs_idx_[-1]]))

            # Reweight
            self.mrw.append(MultiRw(betas, action_, verbose=verbose))
            # and save bs indices
            self.bs_idx.append(bs_idx_)

    def reweight(self, observable, target_betas, Ns=[1]):
        target_values = []
        for sample_num, idx in enumerate(self.bs_idx):
            observable_ = [
                np.array(np.array(observable[i])[idx[i]])
                for i in range(len(self.betas))
            ]
            # for i in len(betas):
            #    observable_.append(np.array(observable[i][idx]))
            # Interpolate moments of observable
            target_values.append({})
            for n in Ns:
                target_values[-1][n] = self.mrw[sample_num].reweight(
                    observable_, target_betas, n=n
                )

        return target_values


class ndMrw:
    """
    Multi-histogram reweighting for n-dimensional parameter.

    J should be provided as a list of 2d arrays, with
    the first (outer) list indexed by the sample source;
    the second indexed by the measurement number;
    the last index is the dimension of the vector parameter.

    E.g. # with A_i, B_i : 1d vectors of length number of measurements
    j1 = np.vstack([A_1, B_1]) # sample from first kappa_1
    j2 = np.vstack([S_2, B_2]) # sample from second kappa_2
    ...

    J = [j1, j2]

    """

    def __init__(self, kappa0, J):
        # TODO: Add check on J shape:
        # All J[i] should be enforced as 2d arrays; all should have same J[i].shape[1]
        self.J = np.hstack(J).T

        # TODO: Add check on kappa0:
        # Should be 2d array with kappa0.shape[1] == J[i].shape[1]
        self.kappa0 = kappa0

        self.logN = np.array([np.log(j.shape[0]) for j in J])
        self.logZ = pyRw.ndimcore.ndItersolve(self.logN, kappa0, self.J)

    def reweight(self, Q, kappa, n=1):
        # TODO: Add check on kappa shape:
        # should be a 2d array with kappa.shape[1] = kappa0.shape[1]
        newQ = np.empty(kappa.shape[0])

        # if all observable values are positive use faster kernel
        if np.all(Q >= 0):
            pyRw.ndimcore.ndGetQn_(
                self.logZ, self.logN, self.kappa0, kappa, self.J, Q, n, newQ
            )
        else:
            pyRw.ndimcore.ndGetQn(  # slower kernel if observable is negative valued
                self.logZ, self.logN, self.kappa0, kappa, self.J, Q, n, newQ
            )
        return newQ
