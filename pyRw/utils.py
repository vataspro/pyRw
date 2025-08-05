"""
pyRw/utils.py

An assortment of small utility functions.

"""

import numpy as np
from warnings import warn


# Ensure that the passed object is a list of lists
def ensureValidObservableShape(x):
    # if not type(x) in (list, ndarray):
    if not ((type(x) is list) or ((type(x) is np.ndarray) and len(x.shape) == 2)):
        raise ValueError("Passed object is not a list or 2d array")

    if not all(
        [   # if it contains anything that isn't a list of lists and/or arrays
            (type(x_) is list) or ((type(x_) is np.ndarray) and (len(x_.shape) == 1))
            for x_ in x
        ]
    ) or (len(x) == 0) or all([len(x_) == 0 for x_ in x]):
        raise ValueError(
            "Passed object is neither a list of lists or a list of 1d arrays"
        )

    return True


# Check that a 2d list has no negative numbers
# TODO custom warning
def checkObservableNotNegative(x):
    if any([any(np.array(x_) < 0) for x_ in x]):
        warn("An observable has a negative value!")
    return True


# Ensure that two lists of lists are the same shape
def ensureSameShape2d(a, b):
    ensureValidObservableShape(a, b)
    if not same_shape2d(a, b):
        raise ValueError("The Observable is not the same shape as the actions")


# Are these two lists of lists are the same shape?
def same_shape2d(a, b):
    ensureValidObservableShape(a)
    ensureValidObservableShape(b)
    if len(a) != len(b):
        return False
    return all(len(sub_a) == len(sub_b) for sub_a, sub_b in zip(a, b))


# Simple skip binning
def binObservable(Q, binSizes):
    ensureValidObservableShape(Q)
    if not len(Q) == len(binSizes):
        raise ValueError("Observable and bin sizes are not the same length")

    return [bin1d(q, binsize) for (q, binsize) in zip(Q, binSizes)]

# 1d binning
def bin1d(q, binsize):
    return q[::binsize]

