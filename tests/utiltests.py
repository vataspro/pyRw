"""

tests/utiltests.py

This file contains basic tests for
utility functions used in pyRw.

"""

import numpy as np
import pyRw.utils
import warnings

# TODO same_shape2d
#      binObservable


def testEnsureValidObservable():
    # 2D array passes
    x = np.random.random([7, 5])
    assert pyRw.utils.ensureValidObservable(x)

    # List of lists passes
    x = x.tolist()
    assert pyRw.utils.ensureValidObservable(x)

    # List of Arrays passes
    x = [np.array(x_) for x_ in x]
    assert pyRw.utils.ensureValidObservable(x)

    # Invalid observable datatypes raise an error
    invalid = [
        list(range(10)),  # 1d lists are not allowed
        "a",  # chars are not allowed
        np.pi,  # test 1
        0,  # test 0
        np.random.random([5, 4, 3]),
    ]  # nd <= 2

    for x in invalid:
        try:
            flag = pyRw.utils.ensureValidObservable(x)
        except:
            flag = False
            raise
        assert not flag


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


if __name__ == "__main__":
    testEnsureValidObservable()
    testCheckObservableNotNegative()
