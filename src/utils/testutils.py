import numpy as np
from scipy.integrate import quad

"""
* runMC

 Runs an MC chain to sample a 1d Gaussian
 with σ = 1/np.sqrt(2*β)

"""
def runMC(N, beta, x0=0, eps=1e-1, Nsweep=100):

    # At x0=0 the chain is thermalised
    x = [x0]

    # Run Metropolis
    for i in range(N - 1):
        xold = x[-1]
        for _ in range(Nsweep):
            xnew = xold + eps * np.random.uniform(-1, 1)
            if not np.exp(-beta * (xnew**2 - xold**2)) > np.random.random():
                xnew = xold
            xold = xnew

        x.append(xold)

    # Consider that there is no autocorrelation
    # thanks to skiping Nsweep configs

    return np.array(x)


"""
* sampleError

Returns the error
of uncorrelated measurements

"""
def sampleError(xval):
    x = np.atleast_1d(xval)
    err = np.std(x - np.mean(x))/np.sqrt(len(x))
    return err


"""
* Normal distribution tools for numerical integral 
 estimation

 These functions are use to compare with the reweighting
 results.

"""
# 
# Normal distribution PDF
def normal_pdf(x, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))

# Integrand for <x^n>
def integrand(x, sigma, n=2):
    return x**n * normal_pdf(x, sigma)

# Compute <x^n> numerically
def expval_xn(sigma, n=2):
    result, error = quad(integrand, -np.inf, np.inf, args=(sigma,))
    return result






