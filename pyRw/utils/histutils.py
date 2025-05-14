import numpy as np

"""
* Histogram
 
 The histogram class takes an MC chain and
 histograms it.

 It can then get the density or absolute number
 of states at a value xval.


"""
class Histogram:
    def __init__(self, x : list, bins=100, binRange=None):
        # Histogram the MC chain
        self.valueDensities, self.binEdges = np.histogram(x, bins=bins, range=binRange)
        self.chainLen = len(x)

        self.valueDensities = np.array(self.valueDensities) / self.chainLen
        # This make the density be the density
        # rather than number of hits in the bin

    # Get the density of states at xval
    # This is a vectorised implementation, but scalar also works
    def density(self, xval):
        idx = np.searchsorted(self.binEdges, xval, side="right") - 1
        
        # Use vectorized logic
        valid = (idx >= 0) & (idx < len(self.valueDensities))
        # Return the results
        result = np.zeros_like(xval, dtype=self.valueDensities.dtype)
        result[valid] = self.valueDensities[idx[valid]]

        # Return array to array and scalar to scalar
        return result if result.shape else result.item()

    # Get the actual number density in the xval bin
    def hist(self, xval):
        return self.chainLen * self.density(xval)


"""
* MultiHistogram

 Contains multiple histograms and
 can sum over them at some value x.

 Can also get the absolute density 
 at some point x.
 
 Give it a list of Histogram
 objects.

"""
class MultiHistogram:
    def __init__(self, histograms : list):
        # Get the histograms
        self.numHistograms = len(histograms)
        self.histogramLengths = np.array([histogram.chainLen for histogram in histograms])
        self.histograms = histograms

    def sumHist(self, xval):
        x = np.array(xval)
        #return np.sum([histogram.hist(xval) for histogram in self.histograms], axis=0)
        return np.sum([histogram.hist(x[:, None]) for histogram in self.histograms], axis=0).reshape(-1)

    def density(self, xval):
        return np.sum([histogram.hist(xval) / histogram.chainLen
                         for histogram in self.histograms])


"""
* MakeNiceHistograms

 Utility function.
 
 Makes a list of hists for a list of MC chains.
 
 The total range of the histograms is set to be the
 same for all histograms, and is simply the smallest
 to the largest values measured in all histograms.
 
"""
# Utility that makes a list of hists for a list of lists
#  with all hists using the same bins
# gotten from the range of all the chains.
def MakeNiceHistograms(xs, bins=100):
    binRange = np.min(xs), np.max(xs)
    return [Histogram(x, bins=bins, binRange=binRange) for x in xs]