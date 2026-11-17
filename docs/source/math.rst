.. _math-section:

Mathematical formulation
========================


Reweighting is a numerical technique, complementary to Monte Carlo 
sampling techniques, that allows for the interpolation and extrapolation 
of the value of observables at values of the input parameters 
that differ from the sampling parameters.

This greatly reduces the numerical cost when calculating 
properties of the system in an almost continuous scan 
of the parameters, for example near a phase transition.

Wherever the term "energy" appears in the context of statistical systems 
(e.g. the Ising Model), the lattice field theory equivalent is the 
Euclidean action.

Much of this section is based on the paedogogical chapter in
:cite:`Newman1998-tj`.

Single histogram reweighting
----------------------------


Single histogram reweighting, first presented in
:cite:`Ferrenberg:1988yz`, is a method used to interpolate
the results of Markov Chain Monte Carlo data at different parameter
values than the ones sampled at.

The probability that a system, e.g. an Ising Model, at some inverse 
temperature :math:`\beta` is in a state with energy :math:`E` is:

.. math::
   :label: singlerw-prob

    p_\beta(E) = \frac{1}{Z(\beta)}N(E) \exp(-\beta E)

where :math:`N(E)` enumerates the states with energy :math:`E` 
and :math:`Z(\beta)` is the partition function, defined as:

.. math::
   :label: singlerw-partition-function-definition
   
    Z(\beta) = \sum_{E}N(E)\exp(-\beta E).


Reweighting utilises the fact that one can express the 
probability that the system be in a state of energy :math:`E`
when at a different temperature, given by :math:`\beta'`, as follows:

.. math::
   :label: singlerw-pdf-interpolation-formula

    p_{\beta'} (E) = 
    \frac{p_\beta(E)\exp[(\beta-\beta')E]}
    {\sum_{E}p_\beta(E)\exp[(\beta-\beta')E]}.


This is directly observable by replacing :math:`p_\beta(E)` with 
the expression above, leading to the same expression but 
with :math:`\beta'` replacing :math:`\beta`.

Estimators
~~~~~~~~~~

Sampling from a known probability density function :math:`p(x)`
can be used to numerically calculate an integral using the following 
prescription:

.. math::
   :label: estimators-prescription

   \int f(x)\mathrm{d}x =
   \int \frac{f(x)}{p(x)}p(x)\mathrm{d}x =
   \langle \frac{f(x)}{p(x)} \rangle_X


Note that in order for this prescription to work, 
even assuming that we can sample the distribution 
infinitely many times, 
:math:`p(x)` must be non-zero on the support of :math:`f`. 

The formula above is actually the way that estimators are defined
when using Monte Carlo methods - for an observable :math:`O`, 
we estimate its value from :math:`N` measurements as:

.. math::
   :label: estimator-approximation

    O_N = \frac{\sum_{i=0}^N O_i \frac{\exp(-\beta E_i)}{p_i}}
    {\sum_{j=0}^N \frac{\exp(-\beta E_j)}{p_j}}

where the numerator is an approximation of the integral:

.. math::
   :label: estimators-numerator-integral

    \int O(E)N(E)\exp(-\beta E) \mathrm{d}E =
    \langle O(E)N(E)\frac{\exp(-\beta E)}{p(\beta, E)} \rangle

and the denominator is an approximation of the partition function,
defined as:

.. math::
   :label: estimators-partition-function-definition

   Z = \int N(E) \exp(-\beta E) \mathrm{d}E.

The estimator for :math:`Q_N` becomes the usual mean when setting 
:math:`p_i = \frac{1}{Z}\exp(-\beta E)`, which is the 
target distribution of most Monte Carlo sampling algorithms.

We can obtain the formula for reweighting immediately by 
choosing :math:`p_i` to be the distribution at a different 
:math:`\beta`, say :math:`\beta_0`, giving:

.. math::
    :label: estimators-target-pdf

    p_i = \frac{1}{Z(\beta')}\exp(-\beta_0 E)

and replacing into the formula for the estimator :eq:`estimator-approximation`
(cancelling out the arising :math:`Z(\beta_0)` terms):


.. math::
   :label: singlerw-interpolation-formula

    O_N = \frac{\sum_{i=0}^N O_i \exp[(\beta_0-\beta) E_i)]}
    {\sum_{j=0}^N \exp[(\beta_0-\beta) E_j]}.

This is exactly the formula for the probability density function
:eq:`estimators-target-pdf` 
above, but expressed in the form of an estimator, 
which can be directly used in numerical calculation.


Valid range of interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given measurements at a parameter value :math:`\beta_0`, 
the requirement that the energy histograms at :math:`\beta_0`
and some target parameter value :math:`\beta` overlap 
can be expressed as follows:

.. math::
   :label: interpolation-range

    |E(\beta_0) - E(\beta)| \le \sigma_E


where :math:`\sigma_E` is the standard deviation of 
the measured values of :math:`E` sampled at :math:`\beta_0`.



Multi dimensional parameter space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter space of the simulated model may have more than one dimension.
For example, for the Ising model, we have the energy:

.. math::
    :label: ising-energy
    
    E = -J \sum_{\langle ij \rangle} s_i s_j 
        + B \sum_i s_i


If the energy of the system can be expressed 
based on two coupling parameters, say :math:`\alpha` 
and :math:`\beta` with conjugate variables (observables) 
:math:`A` and :math:`B` such that:

.. math::
   :label: multiparam-energy

     -\beta E_{tot} = a A + b B

note the signs - this causes what naively seems like 
a sign flip in the calculation - 
then reweighting can be used in exactly the same way 
to estimate the value at a new value of 
parameters :math:`a'` and :math:`b'`:

.. math::
   :label: multiparam-rw-interpolation

   \langle O \rangle = \frac
   {\langle O(A, B) \exp[(\alpha-\alpha')A + (\beta-\beta')B]\rangle_{\alpha, \beta}}
   {\langle \exp[(\alpha-\alpha')A + (\beta-\beta')B]\rangle_{\alpha, \beta}}

