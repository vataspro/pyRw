from scipy.optimize import fsolve
import src.utils.histutils as hu
from typing import Callable
import numpy as np

"""
* MultiRw

 Multiple histogram reweighting class.

 Inputs:
    xs : a list of MC chains (scalar lists)
    betas :  list of the betas corresponding to each MC chain
    bins : number of bins in the histogramming

 Notes:
    If you don't like the look of your reweight, consider increasing
    or decreasing the number of bins.
"""
# Mutli Reweight Class
# Give it a list of histograms
class MultiRw:
    def __init__(self, xs, betas, bins=100):

        # Protect against wrong number of betas
        if len(xs) != len(betas):
            print("Number of betas is not the same as the number of histograms")
            exit(-1)
        # Protect against passing nothing
        if len(betas) < 1:
            print("No ensembles passed!")
            exit(-1)

        # Get the histograms
        self.multiHistogram = hu.MultiHistogram(hu.MakeNiceHistograms(xs, bins=bins))
        self.betas = np.array(betas)

        # Get the action values
        # We are using the first histogram as we have set them all to have the 
        # same bins using MakeNiceHistogram
        rawActionValues = self.multiHistogram.histograms[0].binEdges
        dS = rawActionValues[1]-rawActionValues[0]
        self.actionValues = rawActionValues[:-1] + 0.5*dS
        

        # Initialize logZ
        self.logZ = np.zeros(self.multiHistogram.numHistograms)
        self.setLogZ()

    # This is the density -- vectorised in xval
    def Eta(self, logZ, xval):
        x = np.array(xval).reshape(-1)

        # sum over the histograms
        numerator = self.multiHistogram.sumHist(x)
        # again, sum over the histograms only
        denominator =  np.sum(self.multiHistogram.histogramLengths \
                     * np.exp(-self.betas*x[:, None] - logZ), axis=1).reshape(-1)
    
        return numerator / denominator

    # Solving the equation gives the optimal logZ
    def equation(self, logZ):
        return np.exp(logZ) \
            - np.sum(self.Eta(logZ, self.actionValues) \
              * np.exp(-self.betas[:, None] * self.actionValues), axis=1)

    def setLogZ(self):
        self.logZ = fsolve(self.equation, x0=self.logZ)

    # Get the expval of the action
    def expval(self, 
               beta, 
               logZ : np.ndarray, 
               o : Callable[[np.ndarray], np.ndarray]):
        """
        Get the expectation value of an observable
        at inverse temperature beta.

        The observable is a function of the action.

        Inputs:
            o : function of self.actionValues, the observable
         beta : scalar or 1D array

        """
        beta = np.atleast_1d(beta)
        x = self.actionValues
        O = o(self.actionValues)

        eta = self.Eta(logZ, x)  # shape (n_x,)
        weights = np.exp(-beta[:, None] * x[None, :])  # shape (n_beta, n_x)

        numerator = np.sum(O[None, :] * eta[None, :] * weights, axis=1)
        denominator = np.sum(eta[None, :] * weights, axis=1)

        result = numerator / denominator
        return result if result.shape != (1,) else result[0]
