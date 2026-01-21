"""
pyRw/bootstrap.py

Implements a class for bootstrapping the results of MultiRw.

Uses democratic resampling.

"""

import numpy as np
import pyRw.autocorr
from pyRw.mrw import MultiRw


# Performs one bootstrap
def bootstrap_indices(arr):
    boot_idx = []

    for arr_ in arr:
        len_ = len(arr_)

        idx = np.random.randint(0, len_, size=len_)
        boot_idx.append(idx)

    return boot_idx


def apply_bootstrap(arr, bs_idxs):
    bs_sample = []

    for arr_, idx in zip(arr, bs_idxs):
        arr_ = np.asarray(arr_)
        bs_sample.append(arr_[idx])

    return bs_sample


class bootstrapRw:
    """

    Bootstrap Multi Histogram Reweighting.

    This class calls the MultiRw class on resampled
    energies and observables.

    """

    def __init__(
        self,
        betas,
        E,
        Q,
        target_b=None,
        nboot=10,
        autocorr=True,
        max_iter=100000,
        tol=1e-10,
    ):
        self.autocorr = autocorr

        # Autocorrelation
        if self.autocorr:
            self.texp = [pyRw.autocorr.integrated_autocorrelation_time(x) for x in E]
            self.nskips = np.ceil(self.texp).astype(int)
            self.E = pyRw.utils.binObservable(E, self.nskips)
        else:
            self.E = E

        self.betas = np.array(betas)

        if target_b is None:
            target_b = np.linspace(min(betas), max(betas), 100)

        self.OBS = []
        # Resample and reweight
        for _ in range(nboot):
            bs_idx = bootstrap_indices(self.E)
            E_ = apply_bootstrap(self.E, bs_idx)
            Q_ = apply_bootstrap(Q, bs_idx)
            mrw = MultiRw(self.betas, E_, max_iter=max_iter, tol=tol)

            q = mrw.reweight(Q_, target_b)

            self.OBS.append(q)


# Bootstrap sample
# Repeat nboot times:
# - reweight on sample
# Get bootstrap mean, sdev
