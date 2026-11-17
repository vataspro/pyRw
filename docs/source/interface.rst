.. _user-interface:

User Interface
==============


This section outlines the functions and classes
the end user of **pyRw** is expected to interact with.


For details on the internals please refer to :ref:`core-module`


Multi histogram reweighting
---------------------------

For a straight forward implementation of multi histogram reweighting,
the **MultiRw** class is used:


.. autoclass:: pyRw.mrw.MultiRw
   :members:


Example usage:

.. code-block:: Python

   mrw = MultiRw(betas, E)
   x = mrw.reweight(X, target_betas)


``MultiRw`` can calculate the moment :math:`\langle x^n \rangle`,
by setting the parameter ``n``.
For example, in order to calculate the variance:

.. code-block:: Python

   mrw = MultiRw(betas, E)
   x = mrw.reweight(X, target_betas)
   x2 = mrw.reweight(X, target_betas, n=2)
   variance = x2 - x**2


Multi histogram reweighting with bootstrap resampling
-----------------------------------------------------

In order to repeat the calculation using the bootstrap
method, the ``BootstrapRw`` function can be used, calculating
any requested higher moments and returning them in a dictionary:

.. autofunction:: pyRw.mrw.BootstrapRw


