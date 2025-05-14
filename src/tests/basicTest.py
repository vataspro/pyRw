import numpy as np
import src.multirw as mrw
from src.utils import testutils as tu
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Values of beta to get MC values for
    betas = [1.0, 2.0,  3.0]

    # Plot <x^n> for different distributions
    n = 2

    # Number of bins to use when histogramming
    bins = 250

    # Run Metropolis chains
    N = 10000
    x = []
    for beta in betas:
        x.append(tu.runMC(N, beta))

    
    # Setup the multiple reweighting
    multirw = mrw.MultiRw([np.array(x_)**2 for x_ in x], betas, bins=bins)
    multirw.setLogZ()

    # Plot the reweight
    plotBeta = np.linspace(0.8, 3.5, 10)
    rw_vals = np.array([multirw.expval(b, multirw.logZ) for b in plotBeta])

    plt.plot(plotBeta, rw_vals, 'k', label='pyRw')


    # Numerical intergration of <x^2>
    # Range of sigma values
    betas2 = np.linspace(1., 3., 50)
    sigmas = 1/np.sqrt(2*betas2)
    expvals = [tu.expval_xn(sigma) for sigma in sigmas]
    theory = sigmas**2  # analytical result

    # Plot 
    plt.plot(betas2, expvals, label=r'Numerical $\langle x^2 \rangle$', lw=2)
    plt.plot(betas2, theory, '--', label=r'Analytical $\sigma^2$', lw=2)

    # Plot <x**2> directly from the MC chains
    errs = np.array([tu.sampleError(ens) for ens in x])
    plt.errorbar(betas, [np.mean(ens**2) for ens in x], yerr=errs,
                    marker='o', ls='none', label='MCMC')

    # Monte Carlo using numpy normal
    mc = [np.mean(np.random.normal(scale=1/np.sqrt(2*b), size=10000)**2) for b in plotBeta]
    plt.plot(plotBeta, mc, label='MC on gaussian')
    
    plt.legend()
    plt.title("pyRw comparisson plot")
    plt.show()





