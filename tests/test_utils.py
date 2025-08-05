"""

tests/test_utils.py

This file contains basic tests for
utility functions used in pyRw.

"""

import numpy as np
import pyRw.utils
import warnings
import pytest

# TODO binObservable


def testEnsureValidObservable():
    # 2D array passes
    x = np.random.random([7, 5])
    assert pyRw.utils.ensureValidObservableShape(x)

    # List of lists passes
    x = x.tolist()
    assert pyRw.utils.ensureValidObservableShape(x)

    # List of Arrays passes
    x = [np.array(x_) for x_ in x]
    assert pyRw.utils.ensureValidObservableShape(x)

    # Invalid observable datatypes raise an error
    invalid = [
        list(range(10)),  # 1d lists are not allowed
        "a",  # chars are not allowed
        np.pi,  # test 1
        0,  # test 0
        np.random.random([5, 4, 3]),
        [],
        {}
    ]  # nd <= 2

    for x in invalid:
        with pytest.raises(ValueError):
            pyRw.utils.ensureValidObservableShape(x)


def testCheckObservableNotNegative():
    # Positive observable does not raise warning
    x = np.random.random([3, 6])
    valid = [x, np.zeros_like(x)]  # neither does a zero observable
    for v in valid:
        assert pyRw.utils.checkObservableNotNegative(v)

    # Negative observable raises a warning
    invalid = [np.copy(x) - 1]
    for x in invalid:
        # Check that a warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Always trigger warnings
            pyRw.utils.checkObservableNotNegative(x)

            # Check if any warnings were raised
            flag = any(issubclass(warn.category, Warning) for warn in w)

        assert flag

def testSame_shape2d():
    # simple data type returns false
    invalid_1d = ['a', 0, np.pi, float, [], [[], []], {}]
    for x in invalid_1d:
        with pytest.raises(ValueError):
            pyRw.utils.same_shape2d(x, x)

    # Check that valid 2d types pass
    valid_2d = [ [[1], []], 
                 [[1], [1]],
                 [4*list(range(5)), []],
                 [4*list(range(5)), [3*list(range(5))]]
                ]

    for x in valid_2d:
        assert pyRw.utils.same_shape2d(x, x)

    # Cross them to make sure that they don't cross pass
    for x in valid_2d:
        for x_ in valid_2d:
            if not x == x_:
                assert not pyRw.utils.same_shape2d(x, x_)

def testBinObservable():

    # Check that a 1d array does not pass
    with pytest.raises(ValueError):
        pyRw.utils.binObservable(np.arange(10), [2])

    # Check that a 2d array with a wrong skips breaks
    with pytest.raises(ValueError):
        pyRw.utils.binObservable(np.random.random([5, 3]), [1, 1])

    # Check resulting sizes consistent
    NUM_ENSEMBLES = 5
    NUM_MEASUREMENTS = 5
    assert NUM_ENSEMBLES >= NUM_MEASUREMENTS # meta test

    X = np.random.random([NUM_ENSEMBLES, NUM_MEASUREMENTS])
    skips = list(range(1, NUM_ENSEMBLES+1))

    # The expected length of a binned is ceil( (len(x) / binsize) )
    binned_X = pyRw.utils.binObservable(X,  skips)
    binned_X_lengths = [len(x) for x in binned_X]
    expected_lengths = [int(np.ceil( NUM_MEASUREMENTS / x )) for x in skips]

    for explen, binlen in zip(expected_lengths, binned_X_lengths):
        assert explen == binlen


