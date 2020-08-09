import numpy as np
from math import sqrt, exp, pi, log

def parseData(filename):
    file = open(filename)
    lines = file.readlines()
    data = []
    for line in lines:
        x_y = line.strip().split()
        data.append([float(x_y[0]), float(x_y[1])])
    return np.array(data)


def expectation(data):
    sum = 0
    count = 0
    for datapoint in data:
        sum += datapoint
        count += 1
    return sum / count

def parzen_window_kernel(u):
    if np.linalg.norm(u) <= 0.5:
        return 1
    return 0


def gaussian_kernel(u):
    return 1/sqrt(2 * pi) * exp(-0.5 * (u ** 2))


def kernel_density_at_x(data_set, x, kernel, h):
    if len(data_set[0].shape) > 1:
        d = data_set[0].shape[0]
    else:
        d = 1
    V = h ** d
    result = 0
    for data in data_set:
        result += kernel(np.linalg.norm(x - data) / h)
    return result * (1 / (len(data_set) * V))


def kernel_denstiy_estimation(data_set, kernel, h, interval=None, precision=None):
    if not precision:
        precision = h
    if interval:
        max = interval[1]
        current_bin = interval[0]
    else:
        max = np.amax(data_set)
        current_bin = np.amin(data_set)
    bins = []
    hist = []
    while current_bin <= max:
        bins.append(current_bin)
        hist.append(kernel_density_at_x(data_set, current_bin, kernel, h))
        current_bin += precision
    return np.array(bins), np.array(hist)



def buildHistogram(data_set, bins, density=False, bin_count=False):
    max = np.amax(data_set)
    current_bin = np.amin(data_set)
    if bin_count:
        bins = (max - current_bin) / bins
    data_set.sort()
    bins_array = []
    hist = []
    bins_array.append(current_bin)
    current_bin += bins
    currentIndex = 0
    while current_bin <= max:
        count = 0
        while currentIndex < len(data_set) and data_set[currentIndex] <= current_bin:
            currentIndex += 1
            count += 1
        bins_array.append(current_bin)
        hist.append(count)
        current_bin += bins
    hist[-1] += len(data_set) - currentIndex # count of elements in the last interval
    bins_array = np.array(bins_array)
    hist = np.array(hist)
    if density:
        hist = hist / (len(data_set) * bins)
    return bins_array, hist


def hist_prob(histogram, point):
    x = histogram["bins"]
    y = histogram["hist"]
    if point < x[0] or point > x[-1]:
        return 0.
    i = 0
    for interval in np.nditer(x.T):
        if point < interval:
            break
        i += 1
    if i < len(x):
        return y[i]
    else:
        return y[i - 1]


def log_likelihood_hist(data, hist):
    sum = 0
    for point in data:
        sum += log(hist_prob(hist, point))
    return sum


def log_likelihood(data, px):
    sum = 0
    for sample in data:
        sum += log(px(sample))
    return sum