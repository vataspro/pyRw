"""

tests/test_core.py

Tests for the core part of the
pyRw code.

"""

import numpy as np
import pyRw.core
import pytest


def testSumlogexp():
    # Returns correct value for zeros
    N = 10
    assert all([np.log(n) == pyRw.core.logsumexp1d(np.zeros(n)) for n in range(1, N)])

    # This sum breaks normal numpy, but not logsumexp
    with pytest.warns(RuntimeWarning, match="overflow"):
        assert np.inf == np.log(np.exp(1) + np.exp(10**4))

    # This sum goes to 1000 up to numerical accuracy
    assert np.allclose(1000, pyRw.core.logsumexp1d(np.array([1, 1000])))

    # Passing nan returns nan
    assert np.isnan(pyRw.core.logsumexp1d(np.array([np.nan])))

    # Passing inf returns nan
    # This is not ideal, as np.inf -> np.inf would be preffered.
    # This arises due to np.inf - np.inf
    # Removing this behaviour would however slow down the kernel.
    assert np.isnan(pyRw.core.logsumexp1d(np.array([np.inf])))


def testGetLogZ():
    # Compare getLogZ with naive result
    n = 3  # number of ensembles
    K = 4  # number of target beta values

    betas = np.array([1.0, 1.1, 1.2])
    E = [[1.0, 1.01], [1.5, 1.49, 1.51], [2.1, 2.01]]

    targetb = np.array([1.08, 1.14, 0.96, 1.22])

    # Naive implementation of reweighting
    Z = []
    for k in range(K):
        SUM = 0
        for i in range(n):
            for s in range(len(E[i])):
                SUM_ = 0
                for j in range(n):
                    SUM_ += len(E[j]) * 1 * np.exp((targetb[k] - betas[j]) * E[i][s])
                SUM += 1 / SUM_
        Z.append(SUM)

    # pyRw implementation
    logZ = np.zeros(n)
    logN = np.log([len(e) for e in E])
    newLogZ = np.empty(K)
    newLogZ = pyRw.core.getLogZ(logZ, logN, betas, targetb, np.concatenate(E), newLogZ)

    assert np.allclose(newLogZ, np.log(Z))
