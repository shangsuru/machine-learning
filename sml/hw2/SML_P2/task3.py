import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp, pi
import tools


bin_size = 2

data = np.loadtxt("dataSets/nonParamTrain.txt")

bins, hist = tools.buildHistogram(data, bins=int((np.amax(data) - np.amin(data)) / bin_size), density=True, bin_count=True)

fig, axs = plt.subplots()
axs.bar(bins[:-1], hist, width=bins[1]-bins[0])
axs.set_title("histogram with bin size of " + str(bin_size))
plt.show()