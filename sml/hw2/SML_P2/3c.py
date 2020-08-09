import tools
import numpy as np
import matplotlib.pyplot as plt

k = 2
data = np.loadtxt('dataSets/nonParamTrain.txt')


def getKNN(x, k, data, dist_f):
    dtype = [('pos', np.float64), ('dist', np.float64)]
    res = []
    for sample in data:
        res.append((sample, dist_f(x, sample)))
    res = np.array(res, dtype=dtype)
    res = np.sort(res, order='dist')
    return res[:k]


def KNN_estimation(x, k, data, dist_f):
    knn = getKNN(x, k, data, dist_f)
    V = knn[-1]['dist'] * 2
    return k / (len(data) * V)


def dist_f(x, y): return abs(x - y)


xs = np.linspace(-4, 8, 100)
ys = []

for x in xs:
    ys.append(KNN_estimation(x, k, data, dist_f))

plt.fill_between(xs, ys, label='K = ' + str(k), alpha=0.35)
ll = tools.log_likelihood(data, lambda x: KNN_estimation(x, k, data, dist_f))
print('Log-Likelihood with K = ' + str(k) + ': ' + str(ll))

k = 8

ys = []

for x in xs:
    ys.append(KNN_estimation(x, k, data, dist_f))

plt.fill_between(xs, ys, label='K = ' + str(k), alpha=0.35)
ll = tools.log_likelihood(data, lambda x: KNN_estimation(x, k, data, dist_f))
print('Log-Likelihood with K = ' + str(k) + ': ' + str(ll))

k = 35

ys = []

for x in xs:
    ys.append(KNN_estimation(x, k, data, dist_f))

plt.fill_between(xs, ys, label='K = ' + str(k), alpha=0.35)
ll = tools.log_likelihood(data, lambda x: KNN_estimation(x, k, data, dist_f))
print('Log-Likelihood with K = ' + str(k) + ': ' + str(ll))


plt.legend()
plt.show()
