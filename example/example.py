"""
example/example.py

An example usage of the MultiRw class.

The example data is sampled from a 2D Ising
Model around the critical temperature. In this
example the magnetic susceptibility is reweighted
and compared with the raw data at various
square lattice sizes.

"""

import matplotlib.pyplot as plt
import json
import numpy as np
from pyRw.mrw import MultiRw

# Plot colours
colors = ["#D61F0B", "#D6860B", "#C2D60B", "#1F0BD6"][::-1]

# Load data
with open("example/data.json") as f:
    data = json.load(f)

# Lattice dimensions (Ns**2 lattice points)
Ns = list(data.keys())

# Loop over different lattice sizes
for i, N in enumerate(Ns):
    datum = data[N]

    # Energy and betas
    E = [datum[key]["energy"] for key in datum]
    T = [float(key) for key in datum]
    betas = 1 / np.array(T)

    # Take the absolute value of the magnetisation
    Q = [np.abs(datum[key]["magnetisation"]) for key in datum]

    # Set up reweight (with autocorrelation)
    mrw = MultiRw(betas, E, autocorr=True)

    # β values to reweight at
    b = np.linspace(np.min(betas), np.max(betas), 100)

    # Get the expectation value <Q>
    q = mrw.reweight(Q, b, n=1)
    # Get the second order moment <Q^2>
    q2 = mrw.reweight(Q, b, n=2)

    # Magnetic Susceptibility
    susc = float(N) ** 2 * (q2 - q * q)

    # Plot the measured vs reweighted Magnetic Susceptibilities
    plt.plot(b, susc, c=colors[i])
    plt.plot(betas, [float(N) ** 2 * np.var(q_) for q_ in Q], "o", c=colors[i], label=N)

# Plot finalisation
plt.title("Volume Depenence for the Magnetic Susceptibility of the 2D Ising Model")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle \chi \rangle$")
plt.legend()
plt.show()
