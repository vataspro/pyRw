.. _usage:

Usage
=====

Installation
------------


To install **pyRw**, clone the git repository:

.. code-block:: console

   $ git clone https://github.com/vataspro/pyRw.git 

Then, from the project home directory install via pip:

.. code-block:: console

   $ pip install -e .


.. _quick-start-section:

Quick Start
-----------

Usage of **pyRw** from the end user perspective
is expected to be mainly via the classes found in
``pyRw.mrw``. These are documented in :ref:`user-interface`.

These safely implement the multihistogram reweighting
method, utilising the functions from ``pyRw.core`` for
the single parameter case or ``pyRw.ndimcore`` for the
case of a multidimensional parameter space.

In order to perform multi histogram reweighting the
the :class:`pyRw.mrw.MultiRw` class is the suggested
method.

Example usage:

.. code-block:: Python

   mrw = MultiRw(betas, E)
   x = mrw.reweight(X, target_betas)


The function :func:`pyRw.mrw.BootstrapRw` offers an interface
repeats the calculation multiple times, using bootstrap
sampling with replacement on the input.

