.. _core-module:

Core
====


This section contains documentation for the core functions
in **pyRw**. Users are not advised to interact with these
directly and instead encouraged use the interface objects 
in :ref:`user-interface`.

All internal kernels are *just-in-time* compiled with **numba**.


Utilities
---------

.. autofunction:: pyRw.core.logsumexp1d


.. autofunction:: pyRw.core.itersolve


.. autofunction:: pyRw.core.SingleHistogramReweightLogsumexp


Single parameter internals
--------------------------

.. autofunction:: pyRw.core.getLogZ


.. autofunction:: pyRw.core.getQn


