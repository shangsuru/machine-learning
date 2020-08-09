from math import pi, sqrt, log
import numpy as np

# means, variances and weights contain the initial parameters for each component inside an array


def gaussian(x, y, mean, variance):
    return np.exp(-0.5 * ((x - mean[0])**2/variance[0] + (y - mean[1])**2/variance[1])) / (2 * pi * sqrt(variance[0] * variance[1]))


def multi_gaussian(x, y):
    return sum([gaussian(x, y, means[i], variances[i], weights[i]) for i in range(4)])


def estimation_maximization(x, y, means, variances, weights):

    def e_step(comp):
        return (weights[comp] * gaussian(x, y, means[comp], variances[comp])) / sum([weights[i] * gaussian(x, y, means[i], variances[i]) for i in range(4)])

    def m_step(responsibilities):
        Nj = sum(responsibilities)
        N = len(x)
        mean = 1/Nj * np.array(responsibilities.dot(x),
                               responsibilities.dot(y))
        variance = 1/Nj * \
            np.array(responsibilities.dot(
                (x - mean[0])**2), responsibilities.dot((y - mean[1])**2))
        weight = Nj / N
        return mean, variance, weight

    log_likelihood = 0

    for i in range(30):
        responsibilities_comp1 = e_step(0)
        responsiblities_comp2 = e_step(1)
        responsibilities_comp3 = e_step(2)
        responsibilities_comp4 = e_step(3)

        mean_comp1, variance_comp1, weight_comp1 = m_step(
            responsibilities_comp1)
        mean_comp2, variance_comp2, weight_comp2 = m_step(
            responsibilities_comp2)
        mean_comp3, variance_comp3, weight_comp3 = m_step(
            responsibilities_comp3)
        mean_comp4, variance_comp4, weight_comp4 = m_step(
            responsibilities_comp4)

        means = [mean_comp1, mean_comp2, mean_comp3, mean_comp4]
        variances = [variance_comp1, variance_comp2,
                     variance_comp3, variance_comp4]
        weights = [weight_comp1, weight_comp2, weight_comp3, weight_comp4]

        for i in range(len(x)):
            log_likelihood += log(multi_gaussian(x[i], y[i]))
            print(i, log_likelihood)


def read_data(file):
    with open(file) as f:
        data = f.readlines()
    x = [float((row.strip().split('  ')[0])) for row in data]
    y = [float((row.strip().split('  ')[1])) for row in data]
    return x, y


x, y = read_data("dataSets/gmm.txt")
initial_means = [[2, 2], [0.5, 1], [0.5, 1.5], [1, 0]]
initial_variances = [[1, 1], [1, 1], [1, 1], [1, 1]]
initial_weights = [0.25, 0.25, 0.25, 0.25]

x = np.asarray(x)
y = np.asarray(y)
estimation_maximization(x, y, initial_means,
                        initial_variances, initial_weights)
