# pyRw

pyRw is a Python code for multiple histogram reweighting.

Where possible, it has been vectorised using numpy.

## Description

pyRw loads a list of MC chain actions and observables in
the `MultiRw` class. It uses histogramming to reweight
the observable.

## Requirements

Requires scipy and matplotlib.

## Usage

Run the basic test in `src/tests/basicTest.py` using `python -m
src.tests.basicTest` in the pyRw home direction (where this README lives).

To use, simply create an instance of the `MultiRw` class and call the
expectation value. The actions have been sampled at `Beta` and are evaluated
at `beta`, which can be a scalar or a `np.ndarray`.

``` python
multirw = MultiRw(actions, Betas) # You can also increase the bins with bins=250

rw_vals = multirw.expval(beta, lambda x : x**2) # e.g. to get <S^2> 

```

## Reference

- This is an implementation of the method as found in [the lecture notes by Kari
Rummukainen](https://www.mv.helsinki.fi/home/rummukai/lectures/montecarlo_oulu/lectures/mc_notes4.pdf).
